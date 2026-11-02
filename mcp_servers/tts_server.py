"""TTS MCP server.

Exposes text-to-speech as MCP tools over streamable HTTP. Backend: Piper
(Piper VITS voices, local and fully offline).

The agent calls `synthesize` once per sentence, so the first sentence plays while
the LLM is still writing the rest.

Run:  python -m mcp_servers.tts_server            (port 8002 by default)
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager

import httpx
from loguru import logger
from mcp.server.mcpserver import MCPServer

from mcp_servers.common import (
    MODELS_DIR, b64encode, now_ms, setup_logging, wav_to_pcm16,
)

TTS_BACKEND = os.getenv("TTS_BACKEND", "piper")
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-hfc_female-medium")
# Phoneme length scale (< 1 speaks faster, > 1 slower). On a Pi 4, 0.95 trims
# both synthesis time and playback time ~5% with no audible artefacts.
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
# Groq-hosted TTS (Orpheus): English-only, needs internet, separate free-tier
# quota. WAV is the only supported response format and input is capped at 200
# characters, so longer text is split and the PCM concatenated.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "hannah")
GROQ_TTS_MAX_CHARS = int(os.getenv("GROQ_TTS_MAX_CHARS", "180"))
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

    def _run(self, text: str) -> bytes:
        return b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, syn_config=self._syn)
        )

    async def synthesize(self, text: str, voice: str | None) -> tuple[bytes, int]:
        return await asyncio.to_thread(self._run, text), self.sample_rate


class GroqTTS:
    """Groq-hosted neural TTS (Orpheus). English-only, needs internet."""

    name = "groq"

    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (get a free key at console.groq.com)")
        # One pooled client keeps the TLS connection warm between sentences.
        self._client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        self.model = GROQ_TTS_MODEL
        self.voice = GROQ_TTS_VOICE
        self.sample_rate = 44100

    def _chunks(self, text: str) -> list[str]:
        """Split text into Orpheus-sized pieces at sentence/clause/space
        boundaries (the API caps input at 200 characters)."""
        text = text.strip()
        if len(text) <= GROQ_TTS_MAX_CHARS:
            return [text] if text else []
        pieces: list[str] = []
        buf = ""
        for part in re.split(r"(?<=[.!?;:,])\s+", text):
            if len(buf) + len(part) + 1 <= GROQ_TTS_MAX_CHARS:
                buf = f"{buf} {part}".strip()
                continue
            if buf:
                pieces.append(buf)
            while len(part) > GROQ_TTS_MAX_CHARS:
                cut = part.rfind(" ", 0, GROQ_TTS_MAX_CHARS)
                cut = cut if cut > 0 else GROQ_TTS_MAX_CHARS
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            buf = part
        if buf:
            pieces.append(buf)
        return pieces

    async def _request(self, text: str, voice: str | None) -> bytes:
        resp = await self._client.post("/audio/speech", json={
            "model": self.model,
            "input": text,
            "voice": voice or self.voice,
            "response_format": "wav",
        })
        if resp.status_code >= 400:
            # Groq's own body is actionable (e.g. model_terms_required with a
            # consent URL) - surface it instead of a bare status code.
            raise RuntimeError(f"groq tts {resp.status_code}: {resp.text[:200]}")
        return resp.content

    async def synthesize(self, text: str, voice: str | None) -> tuple[bytes, int]:
        parts = self._chunks(text)
        if not parts:
            return b"", self.sample_rate
        wavs = await asyncio.gather(*[self._request(p, voice) for p in parts])
        pcm_all, sr = [], self.sample_rate
        for wav in wavs:
            pcm, sr = wav_to_pcm16(wav)
            self.sample_rate = sr
            pcm_all.append(pcm)
        return b"".join(pcm_all), sr


BACKENDS = {"piper": Piper, "groq": GroqTTS}
# When the primary engine fails (quota, network), fall back so speech never
# dies mid-sentence. Piper is always available offline.
TTS_FALLBACK = os.getenv("TTS_FALLBACK", "piper" if TTS_BACKEND != "piper" else "")
_engines: dict[str, object] = {}
_lock = asyncio.Lock()

# Exact-text synthesis cache. The bridge asks for the same short lines over and
# over ("Mm-hm.", timer announcements per session, the greeting); a cache hit
# skips Piper entirely. FIFO eviction.
_cache: dict[str, tuple[bytes, int]] = {}
_CACHE_MAX = 64


async def _synthesize_cached(text: str, voice: str | None) -> tuple[bytes, int, str]:
    """Returns (pcm, sr, source). Tries TTS_BACKEND, then TTS_FALLBACK."""
    key = text.strip()
    hit = _cache.get(key)
    if hit is not None:
        return hit[0], hit[1], "cache"
    last_err: Exception | None = None
    backends = [TTS_BACKEND] + ([TTS_FALLBACK] if TTS_FALLBACK and TTS_FALLBACK != TTS_BACKEND else [])
    for backend in backends:
        try:
            engine = await get_engine(backend)
            pcm, sr = await engine.synthesize(text, voice or None)
            if len(_cache) >= _CACHE_MAX and key not in _cache:
                _cache.pop(next(iter(_cache)))
            _cache[key] = (pcm, sr)
            return pcm, sr, engine.name
        except Exception as e:
            last_err = e
            logger.warning(f"[{backend}] synthesize failed: {e}")
    assert last_err is not None
    raise last_err


async def get_engine(name: str):
    async with _lock:
        if name not in _engines:
            if name not in BACKENDS:
                raise ValueError(f"Unknown TTS backend '{name}'. Choose from {list(BACKENDS)}")
            logger.info(f"Loading TTS backend: {name}")
            engine = await asyncio.to_thread(BACKENDS[name])
            await engine.synthesize("Ready.", None)  # warm up the model / connection
            _engines[name] = engine
        return _engines[name]


@asynccontextmanager
async def lifespan(_server):
    await get_engine(TTS_BACKEND)
    yield {}


mcp = MCPServer(
    "tts",
    instructions="Text-to-speech. Call `synthesize` with one sentence of text.",
    lifespan=lifespan,
)


@mcp.tool()
async def synthesize(text: str, voice: str = "", backend: str = "") -> dict:
    """Convert text to speech.

    Args:
        text: Text to speak, ideally one sentence.
        voice: Optional voice id for the backend (e.g. "en_US-hfc_female-medium").
        backend: Optional override (currently only "piper").

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
        "voice": voice or (GROQ_TTS_VOICE if src == "groq" else PIPER_VOICE),
        "audio_secs": round(audio_secs, 3),
        "server_ms": round(server_ms, 1),
        "rtf": round(rtf, 3),
    }


@mcp.tool()
async def tts_info() -> dict:
    """Report the active TTS backend, voice and the backends available."""
    engine = await get_engine(TTS_BACKEND)
    return {"backend": engine.name, "voice": engine.voice, "available": list(BACKENDS)}


async def _selftest():
    import wave

    engine = await get_engine(TTS_BACKEND)
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
        logger.info(f"TTS MCP server on http://{HOST}:{PORT}/mcp (backend={TTS_BACKEND})")
        mcp.run("streamable-http", host=HOST, port=PORT)
