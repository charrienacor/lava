"""Shared helpers for the MCP servers (audio encoding, env loading, logging)."""

import base64
import io
import os
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
    """Mirror loguru output to logs/<name>.log (10 MB rotation, 3 days kept).

    stderr stays the primary sink; the file is a durable copy that survives
    however the server is launched.
    """
    LOGS_DIR.mkdir(exist_ok=True)
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


def resample_int16(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resampler; good enough for speech going into ASR."""
    if src_sr == dst_sr or len(pcm) == 0:
        return pcm
    n_out = int(round(len(pcm) * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm.astype(np.float32)).astype(np.int16)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
