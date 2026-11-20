# voice/ — the voiceprint bundle

Everything about *whose voice Lava listens to*: enrolling it, computing it,
and the tooling around it. The speaker gate itself lives in the ASR server
(`mcp/asr_server.py`, `transcribe` + `enroll` tools) — this folder is what
feeds it.

```
record_voice.py          mic → 16 kHz WAV samples (data/recordings/)
enroll_voice.py          samples → asr server's enroll tool → data/voiceprint.json
voice_encoder.py         audio → 256-d fingerprint (numpy preprocessing + ONNX)
export_voice_encoder.py  dev machine only: resemblyzer → voice/models/voice_encoder.onnx
```

Artifacts (not in this folder, on purpose — they are data, not code):

- `data/voiceprint.json` — the enrolled print (256 floats + owner name from
  `data/profile.yaml`). Delete it → the gate returns to fail-open.
- `voice/models/voice_encoder.onnx` + `voice/models/mel_basis.npy` — the pretrained
  resemblyzer LSTM exported to ONNX (~6 MB) and the mel filterbank. Built
  once; re-enrolling never regenerates them.

## Re-enroll

```bash
.venv/bin/python voice/record_voice.py --samples 3      # or bring your own WAVs
.venv/bin/python voice/enroll_voice.py data/recordings/voice_*.wav
```

No restart needed — `asr_server.py` reloads the print by file mtime.

On the Pi setup: the ASR server runs on the Pi, so enroll **on the Pi** (its
mic is what she actually hears), or enroll on the Mac and copy
`data/voiceprint.json` over.

## The data flow, end to end

```
WAV → preprocess (−30 dBFS, webrtcvad silence trim)
    → 40-band mel spectrogram (mel_basis.npy)
    → 1.6 s partials @ 1.3/s
    → voice_encoder.onnx (onnxruntime, CPU) → 256-d partial embeddings
    → mean, L2-normalize → voiceprint.json
live speech → same encoder → cosine vs the stored print
    ≥ ASR_SPEAKER_MIN_SIM (0.50) → Whisper transcribes
    <  → VoiceNotRecognizedFrame → "Unknown voice" toast
```
