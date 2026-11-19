"""Speaker-embedding encoder for the ASR gate — numpy + ONNX, no torch.

Drop-in replacement for the resemblyzer path this server used before: the
same pretrained LSTM weights, the same preprocessing, the same 256-d
embeddings — but the network runs via onnxruntime (already a pipecat base
dependency) and the preprocessing is pure numpy. The RPi install therefore
needs no torch, librosa, or scipy — the ~600 MB that broke `pip install`
on a weak connection.

Artifacts, built once on the dev machine by tools/export_voice_encoder.py:

    models/voice_encoder.onnx   the LSTM encoder, ~6 MB
    models/mel_basis.npy        the 40 x 201 Slaney mel filterbank, 32 KB

Everything below is a faithful port of resemblyzer 0.1.4 (hparams.py,
audio.py, voice_encoder.py): volume normalization to -30 dBFS, webrtcvad
silence trimming, a 25 ms/10 ms 40-band mel spectrogram, 1.6 s partial
utterances at 1.3/s (0.75 min coverage), the mean of partial embeddings,
and L2 normalization. Embeddings match resemblyzer's to ~1e-6 cosine
(checked by tools/export_voice_encoder.py --verify), so a voiceprint
enrolled under resemblyzer stays valid as-is.
"""

from pathlib import Path

import numpy as np
from loguru import logger

# ---- hparams (resemblyzer/hparams.py, kept in sync) ----------------------
SAMPLING_RATE = 16000
MEL_WINDOW_LENGTH = 25  # ms
MEL_WINDOW_STEP = 10    # ms
MEL_N_CHANNELS = 40
PARTIALS_N_FRAMES = 160     # 1600 ms
VAD_WINDOW_LENGTH = 30   # ms
VAD_MOVING_AVERAGE_WIDTH = 8
VAD_MAX_SILENCE_LENGTH = 6
AUDIO_NORM_TARGET_DBFS = -30.0

N_FFT = int(SAMPLING_RATE * MEL_WINDOW_LENGTH / 1000)     # 400
HOP_LENGTH = int(SAMPLING_RATE * MEL_WINDOW_STEP / 1000)  # 160
INT16_MAX = (2**15) - 1

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _ROOT / "models" / "voice_encoder.onnx"
_MEL_BASIS_PATH = _ROOT / "models" / "mel_basis.npy"

try:
    import webrtcvad
except ImportError:  # trimming degrades gracefully; the gate still works
    webrtcvad = None
    logger.warning("webrtcvad not installed — silence trimming disabled")


# ---- preprocessing (resemblyzer/audio.py) --------------------------------

_warned_resample = False


def _resample(wav: np.ndarray, source_sr: int | None) -> np.ndarray:
    """16 kHz is the only rate this pipeline produces (browser mic, WAVs);
    anything else gets an approximate linear resample with a loud warning."""
    global _warned_resample
    if source_sr is None or source_sr == SAMPLING_RATE:
        return wav
    if not _warned_resample:
        logger.warning(
            f"audio at {source_sr} Hz — linear resampling to {SAMPLING_RATE} Hz "
            "(resemblyzer used a high-quality resampler; enroll at 16 kHz)"
        )
        _warned_resample = True
    n = int(len(wav) * SAMPLING_RATE / source_sr)
    return np.interp(
        np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav
    ).astype(np.float32)


def normalize_volume(
    wav: np.ndarray, target_dBFS: float, increase_only: bool = False
) -> np.ndarray:
    """resemblyzer normalize_volume, verbatim (numpy only)."""
    if len(wav) == 0:
        return wav
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    if rms <= 0:
        # Digital silence (every sample exactly zero — e.g. the browser's
        # echo canceller muting speech that overlaps the bot's own voice):
        # log10(0) is -inf and the gain NaN, which would poison the waveform
        # and trip the divide-by-zero/invalid-value warnings. There is
        # nothing to normalize; trim_long_silences rejects the segment
        # downstream as "no usable speech".
        return wav
    wave_dBFS = 20 * np.log10(rms / INT16_MAX)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav: np.ndarray) -> np.ndarray:
    """resemblyzer trim_long_silences, minus the scipy/webrtcvad originals.

    Same algorithm: webrtcvad (mode 3) flags 30 ms windows, a width-8 moving
    average smooths the flags, voiced regions dilate by ±3 windows — only
    the dilation is hand-rolled (a ±3 OR) instead of scipy.ndimage.
    """
    if webrtcvad is None:
        return wav
    samples_per_window = (VAD_WINDOW_LENGTH * SAMPLING_RATE) // 1000  # 480
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]
    if len(wav) == 0:
        return wav

    # int16 little-endian PCM, as the VAD requires
    pcm = np.round(wav * INT16_MAX).astype("<i2").tobytes()
    vad = webrtcvad.Vad(mode=3)
    voice_flags = np.array(
        [
            vad.is_speech(
                pcm[i * samples_per_window * 2 : (i + 1) * samples_per_window * 2],
                sample_rate=SAMPLING_RATE,
            )
            for i in range(len(wav) // samples_per_window)
        ],
        dtype=float,
    )

    def moving_average(array: np.ndarray, width: int) -> np.ndarray:
        array_padded = np.concatenate(
            (np.zeros((width - 1) // 2), array, np.zeros(width // 2))
        )
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1 :] / width

    audio_mask = np.round(
        moving_average(voice_flags, VAD_MOVING_AVERAGE_WIDTH)
    ).astype(bool)

    # dilate voiced regions by VAD_MAX_SILENCE_LENGTH // 2 = ±3 windows
    # (scipy.ndimage.binary_dilation with a ones(7) structure)
    dilated = audio_mask.copy()
    for shift in range(1, VAD_MAX_SILENCE_LENGTH // 2 + 1):
        dilated[shift:] |= audio_mask[:-shift]
        dilated[:-shift] |= audio_mask[shift:]

    return wav[np.repeat(dilated, samples_per_window)]


def preprocess_wav(wav: np.ndarray, source_sr: int | None = None) -> np.ndarray:
    """resemblyzer preprocess_wav for in-memory waveforms (no file loading):
    resample to 16 kHz, normalize volume, trim long silences."""
    wav = np.asarray(wav, dtype=np.float32)
    wav = _resample(wav, source_sr)
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_DBFS, increase_only=True)
    return trim_long_silences(wav)


# ---- mel spectrogram (resemblyzer audio.wav_to_mel_spectrogram) ----------

_window: np.ndarray | None = None


def _hann_periodic() -> np.ndarray:
    """Periodic Hann — librosa's window (scipy get_window semantics). Note
    np.hanning is the symmetric variant and would bias every bin slightly."""
    global _window
    if _window is None:
        _window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    return _window


def wav_to_mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """(n_frames, 40) mel power spectrogram — numerically identical to
    resemblyzer's librosa call (n_fft=400, hop=160, 40 Slaney mels,
    center-padded, hann window, |STFT|^2)."""
    basis = _MEL_BASIS
    padded = np.pad(wav, (N_FFT // 2, N_FFT // 2), mode="constant")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    if n_frames < 1:
        raise ValueError("audio too short for one spectrogram frame")
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(padded[idx] * _hann_periodic(), axis=1)) ** 2
    return (power @ basis.T).astype(np.float32)


_MEL_BASIS: np.ndarray = np.zeros((MEL_N_CHANNELS, 0), dtype=np.float32)


def load_mel_basis(path: Path | None = None) -> None:
    global _MEL_BASIS
    path = Path(path) if path else _MEL_BASIS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"mel filterbank missing: {path} — run tools/export_voice_encoder.py "
            "on the dev machine and copy the models/ folder over"
        )
    _MEL_BASIS = np.load(path)


# ---- partial slicing (resemblyzer VoiceEncoder.compute_partial_slices) ----


def compute_partial_slices(
    n_samples: int, rate: float = 1.3, min_coverage: float = 0.75
):
    """resemblyzer's slicing, verbatim: 1.6 s partials advancing 16000/rate
    samples per step; the trailing partial is dropped if under 75% covered."""
    assert 0 < min_coverage <= 1

    samples_per_frame = int(SAMPLING_RATE * MEL_WINDOW_STEP / 1000)  # 160
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((SAMPLING_RATE / rate) / samples_per_frame))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= PARTIALS_N_FRAMES, "The rate is too low"

    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - PARTIALS_N_FRAMES + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + PARTIALS_N_FRAMES])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (
        last_wav_range.stop - last_wav_range.start
    )
    if coverage < min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


# ---- the encoder ----------------------------------------------------------

class OnnxVoiceEncoder:
    """The resemblyzer voice encoder as an ONNX runtime session (CPU).

    Same weights as resemblyzer's pretrained.pt — the ONNX graph is exported
    from them (tools/export_voice_encoder.py) — so embeddings are the same
    up to float noise.
    """

    def __init__(self, model_path: Path | None = None, mel_basis_path: Path | None = None):
        import onnxruntime as ort  # pipecat base dependency — always present

        model_path = Path(model_path) if model_path else _MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"voice encoder model missing: {model_path} — export it on the "
                "dev machine (tools/export_voice_encoder.py) and copy models/"
            )
        load_mel_basis(mel_basis_path)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed_utterance(
        self, wav: np.ndarray, source_sr: int | None = None,
        rate: float = 1.3, min_coverage: float = 0.75,
    ) -> np.ndarray:
        """L2-normalized 256-d embedding of one utterance — the same value
        resemblyzer's VoiceEncoder.embed_utterance returns for the same audio.

        Raises ValueError for audio with no usable speech (too short or all
        silence after trimming) — callers treat that as a gate reject.
        """
        wav = preprocess_wav(wav, source_sr)
        if len(wav) == 0:
            raise ValueError("no usable speech after silence trimming")

        wav_slices, mel_slices = compute_partial_slices(
            len(wav), rate, min_coverage
        )
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        mel = wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        partial_embeds = self._session.run(
            None, {self._input_name: mels.astype(np.float32)}
        )[0]

        raw_embed = np.mean(partial_embeds, axis=0)
        return (raw_embed / np.linalg.norm(raw_embed, 2)).astype(np.float32)
