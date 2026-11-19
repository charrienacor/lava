"""Groq Whisper speech-to-text MCP server — owner-voice gated.

Two tools:

- transcribe: Whisper large-v3-turbo, but only speech matching the enrolled
  voiceprint (data/voiceprint.json) is transcribed; other voices — family,
  flatmates, TV — return the not-owner marker (NOT "") so they never become
  Lava's turns and the UI can toast "Not your voice" instead of mistaking a
  rejected voice for real silence. With no voiceprint yet the gate is
  fail-open, so the agent still works.
- enroll: add a speech segment to the owner's voiceprint. Called 2-3 times
  with different samples via agent/enroll_voice.py, never by the LLM (it
  cannot supply audio).

The voiceprint is labeled with the user's name from data/profile.yaml.
"""

import base64
import io
import json
import os
import sys
import threading
import time
import warnings
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from groq import (  # noqa: E402
    APIConnectionError,
    APITimeoutError,
    DefaultHttpxClient,
    Groq,
    GroqError,
    RateLimitError,
)
from loguru import logger  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

# The gate's encoder is a sibling module: the resemblyzer weights exported to
# ONNX + numpy preprocessing — no torch on the RPi (see voice_encoder.py).
# asr_server.py is launched as a script from the project root, so its own
# folder goes on sys.path to make that import work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_encoder import OnnxVoiceEncoder  # noqa: E402

mcp = MCPServer("asr")
# Voice-agent budget: fail fast instead of hanging on a dead connection.
# 3 retries (default 2) ride out transient blips; 15 s timeout caps the wait
# when the network drops mid-request. Override with ASR_TIMEOUT_SECS /
# ASR_MAX_RETRIES if needed.
GROQ_TIMEOUT = float(os.environ.get("ASR_TIMEOUT_SECS", "15"))
GROQ_MAX_RETRIES = int(os.environ.get("ASR_MAX_RETRIES", "3"))
# keepalive_expiry=None: httpx's default drops an idle connection after 5 s,
# and a voice turn is almost always further apart than that — so nearly every
# utterance paid a fresh TCP + TLS handshake to Groq before its upload began.
groq = Groq(
    timeout=GROQ_TIMEOUT,
    max_retries=GROQ_MAX_RETRIES,
    http_client=DefaultHttpxClient(
        limits=httpx.Limits(
            max_connections=8, max_keepalive_connections=4, keepalive_expiry=None
        )
    ),
)

_ROOT = Path(__file__).resolve().parents[1]
PROFILE_YAML = _ROOT / "data" / "profile.yaml"
VOICEPRINT_JSON = _ROOT / "data" / "voiceprint.json"

# Voiceprint match threshold: cosine similarity (embeddings are
# L2-normalized, so this is a dot product) above which speech is accepted
# as the owner's. Same-speaker segments typically score >0.80, other
# speakers <0.65; 0.75 is a middle value — validate with real voices and
# override with ASR_SPEAKER_MIN_SIM if needed.
MIN_SIMILARITY = float(os.environ.get("ASR_SPEAKER_MIN_SIM", "0.50"))
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-large-v3-turbo")

# The gate's reject marker, returned instead of a transcript. Deliberately
# NOT "" — the empty string means "owner spoke, Whisper heard nothing" (real
# silence), and only the marker becomes a "Not your voice" toast. Keep in
# sync with agent/asr.py NOT_OWNER.
_NOT_OWNER = "not-owner"

# Speculative transcription: the Whisper upload starts at the same time as the
# speaker-gate embedding instead of after it, so the gate's cost (mel + LSTM —
# tens to hundreds of ms on a Pi) hides under the network round trip. The
# price: a rejected segment (another voice, the TV) still reaches Groq and
# spends free-tier audio seconds; its transcript is discarded. ASR_SPECULATIVE=0
# restores gate-then-upload.
SPECULATIVE = os.environ.get("ASR_SPECULATIVE", "1") not in ("0", "false", "no")
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-groq")

# Connection keep-warm: Groq's edge closes idle connections, and reopening one
# costs a TCP + TLS handshake on the user's next utterance. While a
# conversation is active (a transcription within ASR_KEEPALIVE_ACTIVE_SECS),
# a cheap models.list() every ASR_KEEPALIVE_SECS keeps the connection open.
# Idle for longer, it stops — no pings around the clock. 0 disables.
KEEPALIVE_SECS = float(os.environ.get("ASR_KEEPALIVE_SECS", "45"))
KEEPALIVE_ACTIVE_SECS = float(os.environ.get("ASR_KEEPALIVE_ACTIVE_SECS", "600"))
_last_activity = time.monotonic()

_encoder: OnnxVoiceEncoder | None = None
_encoder_failed = False
_warned_unenrolled = False
_warned_no_encoder = False


def _get_encoder() -> OnnxVoiceEncoder | None:
    """The ONNX voice encoder, built once — or None if it can't load.

    A failed load is not fatal: like an unenrolled voiceprint, the gate
    fails open (every voice is transcribed) and the reason lands in the log.
    """
    global _encoder, _encoder_failed
    if _encoder is None and not _encoder_failed:
        try:
            _encoder = OnnxVoiceEncoder()
        except Exception as e:
            _encoder_failed = True
            logger.error(f"voice encoder unavailable — speaker gate OFF ({e})")
    return _encoder


def _owner_name() -> str:
    """The user's name from data/profile.yaml, for labeling the voiceprint."""
    try:
        import yaml

        data = yaml.safe_load(PROFILE_YAML.read_text()) or {}
        user = data.get("user") or {}
        return user.get("full_name") or user.get("name") or "unknown"
    except Exception as e:
        logger.warning(f"profile.yaml unreadable ({e}) — labeling voiceprint 'unknown'")
        return "unknown"


_voiceprint_cache: tuple[int, dict | None] | None = None  # (mtime_ns, record)


def _load_voiceprint() -> dict | None:
    """The enrolled voiceprint, cached by file mtime.

    Every utterance used to re-read and re-parse the JSON before it could
    even reach Whisper; the mtime guard keeps re-enrollment instant while
    making the hot path a stat call.
    """
    global _voiceprint_cache
    try:
        mtime = VOICEPRINT_JSON.stat().st_mtime_ns
    except FileNotFoundError:
        _voiceprint_cache = (0, None)
        return None
    if _voiceprint_cache and _voiceprint_cache[0] == mtime:
        return _voiceprint_cache[1]
    try:
        data = json.loads(VOICEPRINT_JSON.read_text())
    except (OSError, ValueError):
        data = None
    record = data if data and data.get("embedding") else None
    _voiceprint_cache = (mtime, record)
    return record


def _decode_wav(audio_b64: str, fmt: str) -> tuple[np.ndarray, int]:
    """Decode Base64 audio to a mono float waveform + its sample rate."""
    audio = base64.b64decode(audio_b64)
    if fmt in ("raw", "pcm"):
        samples = np.frombuffer(audio, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0, 16000
    with wave.open(io.BytesIO(audio), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1)
    return frames.astype(np.float32) / 32768.0, rate


def _embed(audio_b64: str, fmt: str = "wav") -> np.ndarray:
    """L2-normalized speaker embedding of a speech segment.

    Raises if the audio has no usable speech (too short or all silence) or
    the encoder is unavailable — callers treat gate failures as rejects.
    """
    encoder = _get_encoder()
    if encoder is None:
        raise RuntimeError("voice encoder unavailable (models/voice_encoder.onnx)")
    wav, rate = _decode_wav(audio_b64, fmt)
    return encoder.embed_utterance(wav, source_sr=rate)


@mcp.tool()
def enroll(audio_b64: str, fmt: str = "wav") -> str:
    """Add a speech segment to the owner's voiceprint (2-3 different samples)."""
    embedding = _embed(audio_b64, fmt)
    data = _load_voiceprint() or {}
    samples = data.get("samples_list", []) + [embedding.tolist()]
    mean = np.asarray(samples).mean(axis=0)
    mean /= np.linalg.norm(mean)
    record = {
        "owner": _owner_name(),
        "sample_count": len(samples),
        "samples_list": samples,
        "embedding": mean.tolist(),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    VOICEPRINT_JSON.write_text(json.dumps(record))
    try:
        VOICEPRINT_JSON.chmod(0o600)  # owner-only: it IS the owner's voice key
    except OSError:
        pass
    return (
        f"Voiceprint updated: {record['owner']}, {len(samples)} sample(s). "
        "Speaker gating is active."
    )


def _whisper(audio: bytes, fmt: str) -> str:
    """One Groq Whisper request (the SDK retries transient failures)."""
    try:
        result = groq.audio.transcriptions.create(
            model=ASR_MODEL,
            file=(f"audio.{fmt}", io.BytesIO(audio)),
            language="en",
            temperature=0.0,
        )
    except RateLimitError:
        # 429 — SDK already exhausted its retries. Log loudly so the
        # free-tier audio-seconds budget is easy to spot in web.log.
        logger.warning("Groq transcription 429 (rate limit) after all retries")
        raise
    except (APITimeoutError, APIConnectionError) as e:
        logger.warning(f"Groq transcription network failure: {type(e).__name__}")
        raise
    except GroqError as e:
        status = getattr(e, "status_code", None)
        logger.error(f"Groq transcription failed (status={status}): {type(e).__name__}")
        raise
    return result.text


def _gate_accepts(voiceprint: dict, audio_b64: str, fmt: str) -> bool:
    """True when the segment matches the enrolled owner's voiceprint."""
    try:
        similarity = float(np.dot(voiceprint["embedding"], _embed(audio_b64, fmt)))
    except Exception as e:
        # Unembeddable audio (silence/noise, too short) — treat as a
        # reject rather than crashing the ASR call.
        logger.info(f"ASR gate: speech unembeddable ({e}) — rejected")
        return False
    logger.info(f"ASR gate: similarity {similarity:.2f} (threshold {MIN_SIMILARITY})")
    if similarity < MIN_SIMILARITY:
        logger.info(f"ASR gate: rejected — not {voiceprint['owner']}'s voice")
        return False
    return True


@mcp.tool()
def transcribe(audio_b64: str, fmt: str = "wav") -> str:
    """Transcribe Base64 audio using Groq Whisper — owner's voice only.

    Speech whose voiceprint does not match the enrolled owner returns the
    not-owner marker, so other speakers or the TV never become Lava's
    turns — and the UI can toast "Not your voice" instead of mistaking a
    rejected voice for real silence (owner spoke, Whisper heard nothing).
    """
    global _warned_unenrolled, _warned_no_encoder, _last_activity
    _last_activity = time.monotonic()
    voiceprint = _load_voiceprint()
    encoder = _get_encoder() if voiceprint is not None else None
    gated = voiceprint is not None and encoder is not None
    if voiceprint is None:
        if not _warned_unenrolled:
            logger.warning(
                "No voiceprint enrolled — speaker gating is OFF "
                "(run agent/enroll_voice.py to enable it)"
            )
            _warned_unenrolled = True
    elif encoder is None:
        if not _warned_no_encoder:
            logger.warning(
                "Voice encoder unavailable — speaker gating is OFF "
                "(every voice is transcribed; see the load error in the log)"
            )
            _warned_no_encoder = True

    audio = base64.b64decode(audio_b64)
    if fmt in ("raw", "pcm"):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio)
        audio = buffer.getvalue()
        fmt = "wav"

    if not gated:
        return _whisper(audio, fmt)
    if not SPECULATIVE:
        return _whisper(audio, fmt) if _gate_accepts(voiceprint, audio_b64, fmt) else _NOT_OWNER

    # Speculative: upload now, gate meanwhile; a reject discards the result
    # (the request cannot be recalled once sent — its errors are dropped too).
    pending = _POOL.submit(_whisper, audio, fmt)
    if not _gate_accepts(voiceprint, audio_b64, fmt):
        pending.add_done_callback(lambda f: f.exception())  # consume, don't log
        return _NOT_OWNER
    return pending.result()


def _keep_warm() -> None:
    """Open the Groq connection now, then keep it open while in use."""
    while True:
        try:
            groq.models.list()
        except Exception as e:
            logger.debug(f"Groq keep-warm ping failed: {e}")
        if KEEPALIVE_SECS <= 0:
            return
        time.sleep(KEEPALIVE_SECS)
        while time.monotonic() - _last_activity > KEEPALIVE_ACTIVE_SECS:
            time.sleep(KEEPALIVE_SECS)  # idle: stop pinging until speech returns


if __name__ == "__main__":
    # Daemon thread: the handshake happens at startup instead of on the
    # user's first utterance.
    threading.Thread(target=_keep_warm, name="groq-keep-warm", daemon=True).start()
    mcp.run(transport="stdio")
