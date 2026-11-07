"""ASR MCP server.

Exposes speech-to-text as MCP tools over streamable HTTP. Backend: Groq
Whisper-large-v3-turbo (SOTA accuracy, free tier).

Run:  python -m mcp_servers.asr_server            (port 8001 by default)
"""

import asyncio
import os
from contextlib import asynccontextmanager

import httpx
from loguru import logger
from mcp.server.mcpserver import MCPServer

from mcp_servers.common import (
    b64decode,
    now_ms,
    pcm16_to_wav,
    setup_logging,
    wav_duration,
)

ASR_BACKEND = os.getenv("ASR_BACKEND", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
ASR_NO_SPEECH_MAX = float(os.getenv("ASR_NO_SPEECH_MAX", "0.8"))
ASR_LOGPROB_MIN = float(os.getenv("ASR_LOGPROB_MIN", "-1.3"))
HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("ASR_PORT", "8001"))
ASR_SAMPLE_RATE = 16000


@asynccontextmanager
async def lifespan(_server):
    await get_engine(ASR_BACKEND)
    yield {}


mcp = MCPServer(
    "asr",
    instructions="Speech-to-text. Call `transcribe` with a base64 16-bit mono WAV.",
    lifespan=lifespan,
)


class GroqWhisper:
    name = "groq"

    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set (get a free key at console.groq.com)"
            )
        self._client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            # Read timeout: observed Groq latencies run 0.6-3.3 s, with
            # free-tier spikes past 8 s (the old ceiling) that surfaced as
            # blank "transcribe failed:" errors. 12 s rescues the borderline
            # clips; the 45 s turn deadline still bounds the worst case.
            timeout=httpx.Timeout(12.0, connect=5.0),
        )
        self.model = GROQ_ASR_MODEL

    async def transcribe(self, wav: bytes, language: str) -> tuple[str, dict | None]:
        resp = await self._client.post(
            "/audio/transcriptions",
            files={"file": ("speech.wav", wav, "audio/wav")},
            data={
                "model": self.model,
                "language": language,
                "response_format": "verbose_json",
                "temperature": "0",
            },
        )
        resp.raise_for_status()
        j = resp.json()
        segs = j.get("segments") or []
        quality = None
        if segs:
            quality = {
                "no_speech_prob": max(s.get("no_speech_prob", 0.0) for s in segs),
                "avg_logprob": min(s.get("avg_logprob", 0.0) for s in segs),
            }
        return j.get("text", "").strip(), quality

    async def warmup(self):
        try:
            await self._client.get("/models")
        except Exception as e:
            logger.warning(f"Groq warmup failed: {e}")


BACKENDS = {"groq": GroqWhisper}
_engines: dict[str, object] = {}
_lock = asyncio.Lock()


async def get_engine(name: str):
    async with _lock:
        if name not in _engines:
            if name not in BACKENDS:
                raise ValueError(
                    f"Unknown ASR backend '{name}'. Choose from {list(BACKENDS)}"
                )
            logger.info(f"Loading ASR backend: {name}")
            engine = await asyncio.to_thread(BACKENDS[name])
            await engine.warmup()
            _engines[name] = engine
        return _engines[name]


@mcp.tool()
async def transcribe(
    audio_wav_b64: str, language: str = ASR_LANGUAGE, backend: str = ""
) -> dict:
    """Transcribe speech audio to text.

    Args:
        audio_wav_b64: Base64-encoded 16-bit PCM WAV file (mono, any sample rate).
        language: ISO-639-1 language code, e.g. "en" or "tl".
        backend: Optional override (currently only "groq").
    """
    t0 = now_ms()
    try:
        wav = b64decode(audio_wav_b64)
        # Header-only read: the old full decode (numpy, whole clip) existed
        # solely to derive the clip length for the log line.
        audio_secs, _sr = wav_duration(wav)
        engine = await get_engine(backend or ASR_BACKEND)
        text, quality = await engine.transcribe(wav, language)
    except Exception as e:
        # httpx timeout exceptions stringify to '' - name them explicitly so
        # the log line (and the apology trigger) says WHY it failed.
        detail = str(e) or type(e).__name__
        if isinstance(e, httpx.TimeoutException):
            detail = f"{type(e).__name__} (Groq request timed out)"
        elif isinstance(e, httpx.HTTPStatusError):
            detail = f"HTTP {e.response.status_code} from Groq"
        logger.error(f"transcribe failed: {detail}")
        return {
            "text": "",
            "error": f"transcribe failed: {detail[:200]}",
            "audio_secs": 0.0,
            "server_ms": round(now_ms() - t0, 1),
        }
    server_ms = now_ms() - t0
    q = dropped = ""
    if quality:
        q = (
            f" no_speech={quality['no_speech_prob']:.2f} "
            f"logprob={quality['avg_logprob']:.2f}"
        )
        if (
            quality["no_speech_prob"] >= ASR_NO_SPEECH_MAX
            or quality["avg_logprob"] <= ASR_LOGPROB_MIN
        ):
            dropped = " DROPPED (degraded audio)"
            text = ""
    logger.info(
        f"[{engine.name}] {server_ms:.0f} ms for {audio_secs:.2f}s audio{q}: {text!r}{dropped}"
    )
    return {
        "text": text,
        "no_speech_prob": quality["no_speech_prob"] if quality else None,
        "avg_logprob": quality["avg_logprob"] if quality else None,
        "backend": engine.name,
        "model": engine.model,
        "audio_secs": round(audio_secs, 3),
        "server_ms": round(server_ms, 1),
    }


@mcp.tool()
async def asr_info() -> dict:
    """Report the active ASR backend, model and the backends available."""
    engine = await get_engine(ASR_BACKEND)
    return {"backend": engine.name, "model": engine.model, "available": list(BACKENDS)}


async def _selftest():
    """Sanity check without a microphone: transcribe a second of silence."""
    engine = await get_engine(ASR_BACKEND)
    wav = pcm16_to_wav(bytes(ASR_SAMPLE_RATE * 2), ASR_SAMPLE_RATE)
    t0 = now_ms()
    text, _ = await engine.transcribe(wav, ASR_LANGUAGE)
    print(f"{engine.name}: {now_ms() - t0:.0f} ms -> {text!r}")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        asyncio.run(_selftest())
    else:
        setup_logging("asr")
        logger.info(
            f"ASR MCP server on http://{HOST}:{PORT}/mcp (backend={ASR_BACKEND})"
        )
        mcp.run(
            "streamable-http",
            host=HOST,
            port=PORT,
            max_request_body_size=16 * 1024 * 1024,
        )
