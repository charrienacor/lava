"""Piper text-to-speech MCP server."""

import base64
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from piper import PiperVoice

mcp = MCPServer("tts")

VOICE = (
    Path(__file__).resolve().parent
    / "tts_model"
    / "en_US-hfc_female-medium.onnx"
)

voice = PiperVoice.load(str(VOICE))

# Set TTS_DEBUG_WAV=1 to save the last reply to /tmp/last_tts.wav
DEBUG_WAV = os.environ.get("TTS_DEBUG_WAV") == "1"


def log(msg: str) -> None:
    # stderr only: stdout is the MCP channel and must stay clean
    print(f"[tts_server] {msg}", file=sys.stderr, flush=True)


def _warm_up() -> None:
    """One throwaway synthesis at load: onnxruntime allocates its arenas and
    espeak loads its data on the FIRST run, which is several times slower
    than every later one — on a Pi that cost landed on the first reply."""
    t = time.monotonic()
    for _ in voice.synthesize("Ready."):
        pass
    log(f"voice warm in {time.monotonic() - t:.2f}s")


_warm_up()


@mcp.tool()
def speak(text: str) -> str:
    """Convert text to Base64 WAV audio."""
    t_start = time.monotonic()
    log(f"request received: {len(text)} chars: {text[:60]!r}")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        voice.synthesize_wav(text, wav)
        sample_rate = wav.getframerate()

    data = buffer.getvalue()
    synth_s = time.monotonic() - t_start

    # 16-bit mono: 2 bytes per sample, minus the 44-byte WAV header
    audio_s = max(len(data) - 44, 0) / 2 / sample_rate
    rtf = synth_s / audio_s if audio_s else float("inf")
    log(
        f"synth {synth_s:.2f}s | audio {audio_s:.2f}s | "
        f"real-time factor {rtf:.2f} (must be < 1.0) | {len(data)} bytes"
    )

    if DEBUG_WAV:
        Path("/tmp/last_tts.wav").write_bytes(data)

    return json.dumps({
        "audio_b64": base64.b64encode(data).decode("utf-8"),
        "sample_rate": sample_rate,
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
