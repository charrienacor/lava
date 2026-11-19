"""Export the resemblyzer voice encoder to ONNX — dev-side one-time tool.

The speaker gate (mcp/asr_server.py) used to embed speech with resemblyzer,
which drags in torch (~600 MB installed) + librosa/numba/scipy — the reason
`pip install` dies on a poorly connected RPi. The encoder itself is tiny:
a 3-layer LSTM (40 -> 256) with a linear projection, ~1.4 M parameters.

This script (run on a machine that HAS resemblyzer + torch, e.g. the dev
Mac) produces the two artifacts the RPi runtime needs:

    models/voice_encoder.onnx   the LSTM encoder, ~6 MB, opset 17
    models/mel_basis.npy        the 40 x 201 Slaney mel filterbank, 32 KB

mcp/voice_encoder.py reimplements resemblyzer's preprocessing in pure numpy
and runs the ONNX model via onnxruntime (already a pipecat base dependency),
so the RPi install drops torch entirely. Embeddings are numerically faithful
to resemblyzer's — the enrolled voiceprint.json stays valid (verify with
--verify, which needs resemblyzer present).

Usage:
    .venv/bin/python tools/export_voice_encoder.py            # export
    .venv/bin/python tools/export_voice_encoder.py --verify   # export + check
                                                              # against resemblyzer
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "models"
_ONNX_PATH = _MODELS / "voice_encoder.onnx"
_MEL_PATH = _MODELS / "mel_basis.npy"

# resemblyzer hparams (resemblyzer/hparams.py) — keep in sync.
SAMPLING_RATE = 16000
MEL_WINDOW_LENGTH = 25  # ms
MEL_WINDOW_STEP = 10    # ms
MEL_N_CHANNELS = 40

# The exact filterbank resemblyzer's wav_to_mel_spectrogram produces:
# librosa.feature.melspectrogram defaults (htk=False, norm='slaney',
# fmin=0, fmax=sr/2) over an n_fft=400 / hop=160 STFT.
N_FFT = int(SAMPLING_RATE * MEL_WINDOW_LENGTH / 1000)  # 400


def export() -> None:
    import torch
    from resemblyzer import VoiceEncoder

    _MODELS.mkdir(exist_ok=True)

    encoder = VoiceEncoder(device="cpu", verbose=False)
    encoder.eval()

    # A partial utterance: (batch, n_frames, mel channels). Both batch and
    # time are dynamic — utterance length varies per turn.
    dummy = torch.randn(1, 160, MEL_N_CHANNELS)
    torch.onnx.export(
        encoder,
        (dummy,),
        str(_ONNX_PATH),
        input_names=["mels"],
        output_names=["embeds"],
        dynamic_axes={"mels": {0: "batch", 1: "frames"}, "embeds": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the legacy TorchScript exporter — no onnxscript needed
    )
    size_mb = _ONNX_PATH.stat().st_size / 1e6
    print(f"wrote {_ONNX_PATH} ({size_mb:.1f} MB)")

    # The mel filterbank, byte-identical to what librosa computes on the
    # resemblyzer call — the RPi runtime loads this instead of librosa.
    import librosa

    basis = librosa.filters.mel(sr=SAMPLING_RATE, n_fft=N_FFT, n_mels=MEL_N_CHANNELS)
    np.save(_MEL_PATH, basis.astype(np.float32))
    print(f"wrote {_MEL_PATH} ({basis.shape[0]}x{basis.shape[1]} float32)")

    # Torch vs ONNX parity on random inputs (the recordings check runs in
    # --verify; this catches a broken export immediately).
    import onnxruntime as ort

    session = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
    mels = torch.randn(3, 160, MEL_N_CHANNELS).numpy().astype(np.float32)
    with torch.no_grad():
        expected = encoder(torch.from_numpy(mels)).numpy()
    got = session.run(None, {session.get_inputs()[0].name: mels})[0]
    diff = float(np.max(np.abs(expected - got)))
    assert diff < 1e-4, f"ONNX export deviates from torch: max abs diff {diff}"
    print(f"torch vs ONNX parity: max abs diff {diff:.2e} — OK")


def verify() -> None:
    """End-to-end: resemblyzer vs the numpy+ONNX runtime the RPi will use."""
    import json
    import wave

    sys.path.insert(0, str(_ROOT / "mcp"))
    from voice_encoder import OnnxVoiceEncoder

    from resemblyzer import VoiceEncoder, preprocess_wav

    ref = VoiceEncoder(device="cpu", verbose=False)
    port = OnnxVoiceEncoder()

    # 1) mel spectrogram parity on real audio
    wavs = []
    for path in sorted((_ROOT / "data" / "recordings").glob("*.wav")):
        with wave.open(str(path)) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        wavs.append((path.name, pcm.astype(np.float32) / 32768.0, rate))

    from resemblyzer import audio as r_audio
    from voice_encoder import wav_to_mel_spectrogram

    name, wav, rate = wavs[0]
    pre = preprocess_wav(wav, source_sr=rate)
    mel_ref = r_audio.wav_to_mel_spectrogram(pre)
    mel_new = wav_to_mel_spectrogram(pre)
    rel = np.max(np.abs(mel_ref - mel_new)) / max(float(np.max(np.abs(mel_ref))), 1e-9)
    print(f"mel parity on {name}: max rel diff {rel:.2e}")
    assert rel < 1e-3, "mel spectrogram deviates from resemblyzer's"

    # 2) full-pipeline embedding parity + gate similarity vs the live voiceprint
    vp = json.loads((_ROOT / "data" / "voiceprint.json").read_text())
    stored = np.asarray(vp["embedding"], dtype=np.float32)

    print(f"\n{'recording':<32} {'ref vs port':>12} {'port vs print':>14}")
    for name, wav, rate in wavs:
        ref_emb = ref.embed_utterance(preprocess_wav(wav, source_sr=rate))
        port_emb = port.embed_utterance(wav, source_sr=rate)
        cos_rp = float(np.dot(ref_emb, port_emb))
        cos_vp = float(np.dot(port_emb, stored))
        print(f"{name:<32} {cos_rp:>12.6f} {cos_vp:>14.4f}")
    print(
        "\nref vs port ≈ 1.0 → the ONNX runtime is numerically faithful;\n"
        "port vs print  → similarity each recording scores against the\n"
        "existing voiceprint.json (the gate threshold is 0.50)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="also check parity")
    args = parser.parse_args()
    export()
    if args.verify:
        verify()
