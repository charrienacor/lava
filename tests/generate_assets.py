"""(Re)generate the benchmark WAV assets in tests/assets/.

The latency benchmark feeds these exact files as fake microphone input, so
every student / machine measures identical audio. Run this once after Piper
is installed (it happens automatically on the first ./run.sh):

    .venv/bin/python tests/generate_assets.py

The WAVs are committed to the repo — you only need this if they are missing
or if you deliberately want to re-record them.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_servers.common import resample_int16  # noqa: E402

PHRASES = {
    "no_tool": "Tell me a fun fact about the ocean.",
    "tool_time": "What time is it in Manila right now?",
    "tool_weather": "What's the weather like in Quezon City today?",
    "barge": "Anyway, what about tomorrow then?",
}
SAMPLE_RATE = 16000
OUT = Path(__file__).resolve().parent / "assets"


def main():
    from piper import PiperVoice

    OUT.mkdir(exist_ok=True)
    voice = PiperVoice.load("models/piper/en_US-hfc_female-medium.onnx")
    manifest = {"sample_rate": SAMPLE_RATE, "phrases": {}}
    for name, text in PHRASES.items():
        pcm = b"".join(ch.audio_int16_bytes for ch in voice.synthesize(text))
        pcm16 = resample_int16(np.frombuffer(pcm, np.int16),
                               voice.config.sample_rate, SAMPLE_RATE).tobytes()
        with wave.open(str(OUT / f"{name}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16)
        manifest["phrases"][name] = text
        print(f"  {name}.wav  {len(pcm16) / (SAMPLE_RATE * 2):.2f}s  {text!r}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(PHRASES)} WAVs + manifest.json to {OUT}")


if __name__ == "__main__":
    main()
