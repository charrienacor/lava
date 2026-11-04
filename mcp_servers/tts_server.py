"""TTS MCP server.

Exposes text-to-speech as MCP tools over streamable HTTP. Backend: Piper
(Piper VITS voices, local and fully offline).

The agent calls `synthesize` once per sentence, so the first sentence plays while
the LLM is still writing the rest.

Run:  python -m mcp_servers.tts_server            (port 8002 by default)
"""

import asyncio
import os
from contextlib import asynccontextmanager

from loguru import logger
from mcp.server.mcpserver import MCPServer

from mcp_servers.common import MODELS_DIR, b64encode, now_ms, setup_logging

PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-hfc_female-medium")
# Phoneme length scale (< 1 speaks faster, > 1 slower). On a Pi 4, 0.95 trims
# both synthesis time and playback time ~5% with no audible artefacts.
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("TTS_PORT", "8002"))


class Piper:
    name = "piper"

    def __init__(self):
        from piper import PiperVoice, SynthesisConfig
        from piper.download_voices import download_voice

        voice_dir = MODELS_DIR / "piper"
        voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = voice_dir / f"{PIPER_VOICE}.onnx"
        if not model_path.exists():
            logger.info(f"Downloading Piper voice {PIPER_VOICE}")
            download_voice(PIPER_VOICE, voice_dir)
        self._voice = PiperVoice.load(model_path)
        self.sample_rate = self._voice.config.sample_rate
        self.voice = PIPER_VOICE
        self._syn = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE) if PIPER_LENGTH_SCALE != 1.0 else None
        # One voice, one synthesis at a time: PiperVoice is not documented as
        # thread-safe, and concurrent calls do happen (a timer announcement
        # colliding with a turn's pipelined sentence).
        self._lock = asyncio.Lock()

    def _run(self, text: str) -> bytes:
        return b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, syn_config=self._syn)
        )

    async def synthesize(self, text: str, voice: str | None) -> tuple[bytes, int]:
        async with self._lock:
            return await asyncio.to_thread(self._run, text), self.sample_rate


_engine: Piper | None = None
_lock = asyncio.Lock()

# Exact-text synthesis cache. The bridge asks for the same short lines over and
# over ("Mm-hm.", timer announcements per session, the greeting); a cache hit
# skips Piper entirely. FIFO eviction.
_cache: dict[str, tuple[bytes, int]] = {}
_CACHE_MAX = 64


async def _synthesize_cached(text: str, voice: str | None) -> tuple[bytes, int, str]:
    """Returns (pcm, sr, source). Synthesizes with Piper (or serves from cache)."""
    key = text.strip()
    hit = _cache.get(key)
    if hit is not None:
        return hit[0], hit[1], "cache"
    engine = await get_engine()
    pcm, sr = await engine.synthesize(text, voice or None)
    if len(_cache) >= _CACHE_MAX and key not in _cache:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (pcm, sr)
    return pcm, sr, engine.name


async def get_engine() -> Piper:
    global _engine
    async with _lock:
        if _engine is None:
            logger.info("Loading TTS backend: piper")
            engine = await asyncio.to_thread(Piper)
            await engine.synthesize("Ready.", None)  # warm up the model
            _engine = engine
        return _engine


@asynccontextmanager
async def lifespan(_server):
    await get_engine()
    yield {}


mcp = MCPServer(
    "tts",
    instructions="Text-to-speech. Call `synthesize` with one sentence of text.",
    lifespan=lifespan,
)


@mcp.tool()
async def synthesize(text: str, voice: str = "") -> dict:
    """Convert text to speech.

    Args:
        text: Text to speak, ideally one sentence.
        voice: Optional Piper voice id (e.g. "en_US-hfc_female-medium"); defaults
               to PIPER_VOICE.

    Returns raw 16-bit little-endian mono PCM, base64-encoded, plus its sample rate.
    """
    t0 = now_ms()
    pcm, sr, src = await _synthesize_cached(text, voice)
    server_ms = now_ms() - t0
    audio_secs = len(pcm) / 2 / sr if sr else 0.0
    rtf = (server_ms / 1000) / audio_secs if audio_secs else 0.0
    logger.info(f"[{src}] {server_ms:.0f} ms -> {audio_secs:.2f}s audio (RTF {rtf:.2f}): {text!r}")
    return {
        "audio_pcm16_b64": b64encode(pcm),
        "sample_rate": sr,
        "backend": src,
        "voice": voice or PIPER_VOICE,
        "audio_secs": round(audio_secs, 3),
        "server_ms": round(server_ms, 1),
        "rtf": round(rtf, 3),
    }


@mcp.tool()
async def tts_info() -> dict:
    """Report the active TTS backend and voice."""
    engine = await get_engine()
    return {"backend": engine.name, "voice": engine.voice}


async def _selftest():
    import wave

    engine = await get_engine()
    t0 = now_ms()
    pcm, sr = await engine.synthesize("Hello! This is a test of the text to speech server.", None)
    print(f"{engine.name}: {now_ms() - t0:.0f} ms, {len(pcm) / 2 / sr:.2f}s audio @ {sr} Hz")
    with wave.open("tts_selftest.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print("wrote tts_selftest.wav")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        asyncio.run(_selftest())
    else:
        setup_logging("tts")
        logger.info(f"TTS MCP server on http://{HOST}:{PORT}/mcp (backend=piper)")
        mcp.run("streamable-http", host=HOST, port=PORT)
