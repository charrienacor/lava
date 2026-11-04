"""Shared helpers for the MCP servers (audio encoding, env loading, logging)."""

import base64
import io
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
LOGS_DIR = ROOT / "logs"

load_dotenv(ROOT / ".env")


def setup_logging(name: str) -> None:
    """Point loguru at two sinks: the console (INFO+) and logs/<name>.log (DEBUG+).

    Loguru's default stderr sink runs at DEBUG, so without this every debug
    line floods the terminal. The file stays at DEBUG as the durable copy
    (10 MB rotation, 3 days kept) that survives however the server is launched.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    logger.remove()  # drop loguru's default DEBUG-level stderr sink
    logger.add(sys.stderr, level="INFO")
    logger.add(
        LOGS_DIR / f"{name}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="3 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(data: str) -> bytes:
    return base64.b64decode(data)


def wav_to_pcm16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a mono/stereo 16-bit WAV into (int16 mono samples, sample_rate)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return pcm, sr


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def trim_silence(
    pcm: bytes,
    sample_rate: int,
    *,
    frame_ms: int = 40,
    threshold: float = 0.010,
    pad_ms: int = 120,
) -> bytes | None:
    """Cut leading/trailing near-silence from 16-bit mono PCM.

    Spans the VAD hands over (especially around music/TV) carry seconds of
    quiet tail; uploading them costs ASR time on the free tier and degrades
    Whisper's quality gate. Frame RMS drives the cut; ~-40 dBFS counts as
    silence and both ends keep `pad_ms` of margin so no phoneme is clipped.

    Returns the trimmed bytes, or None when there is no speech-level frame
    at all (callers treat that as silence and skip ASR entirely)."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    n = int(sample_rate * frame_ms / 1000)
    n_frames = len(samples) // n
    if n_frames < 2:
        return pcm
    x = samples[: n_frames * n].astype(np.float32) / 32768.0
    frames = x.reshape(n_frames, n)
    rms = np.sqrt((frames * frames).mean(axis=1))
    loud = np.nonzero(rms >= threshold)[0]
    if len(loud) == 0:
        return None
    pad_frames = int(pad_ms * sample_rate / 1000 / n)
    start = max(0, int(loud[0]) - pad_frames) * n
    end = min(n_frames, int(loud[-1]) + pad_frames + 1) * n
    if start <= 0 and end >= len(samples):
        return pcm  # nothing worth cutting
    return samples[start:end].tobytes()


def wav_duration(wav_bytes: bytes) -> tuple[float, int]:
    """(seconds, sample_rate) from the WAV header alone.

    Reads only the RIFF header (no frame data), so callers that just need
    the clip length - e.g. ASR logging - never pay for a full decode."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate(), wf.getframerate()
