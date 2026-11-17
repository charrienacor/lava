
"""Piper text-to-speech MCP server."""

import base64
import io
import json
import wave
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from piper import PiperVoice

mcp = MCPServer("tts")

VOICE = (
    Path(__file__).resolve().parents[1]
    / "voices"
    / "en_US-hfc_female-medium.onnx"
)

voice = PiperVoice.load(str(VOICE))

@mcp.tool()
def speak(text: str) -> str:
    """Convert text to Base64 WAV audio."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        voice.synthesize_wav(text, wav)
        sample_rate = wav.getframerate()
    return json.dumps({
        "audio_b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": sample_rate,
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")
