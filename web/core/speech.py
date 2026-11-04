"""Speech output: the TTS cache, sentence splitting, and the pipelined Speaker
that synthesizes one sentence ahead while the previous one streams."""

import asyncio
import base64
import os
import re
import struct
import time

from aiohttp import web
from loguru import logger

from web.core.config import _GREETING, CHUNK_BYTES
from web.core.mcp_client import tts_conn
from web.core.state import (
    SPEECH_CACHE,
    SPEECH_CACHE_MAX,
    SPEECH_CACHE_MAX_CHARS,
    _fd_record,
)


def _cache_speech(text: str, pcm: bytes, sr: int) -> None:
    if len(text) > SPEECH_CACHE_MAX_CHARS:
        return
    if len(SPEECH_CACHE) >= SPEECH_CACHE_MAX:
        SPEECH_CACHE.pop(next(iter(SPEECH_CACHE)))
    SPEECH_CACHE[text] = (pcm, sr)


async def synth_audio(text: str, stats: dict | None = None) -> tuple[bytes, int]:
    """Synthesize one sentence (or fetch it from the cache)."""
    cached = SPEECH_CACHE.get(text)
    if cached:
        return cached
    t0 = time.perf_counter()
    res = await tts_conn.call("synthesize", {"text": text})
    pcm = base64.b64decode(res["audio_pcm16_b64"])
    if stats is not None:
        stats["tts_ms"] = stats.get("tts_ms", 0) + (time.perf_counter() - t0) * 1000
    _cache_speech(text, pcm, res["sample_rate"])
    return pcm, res["sample_rate"]


async def stream_pcm(
    ws: web.WebSocketResponse,
    sess: dict | None,
    pcm: bytes,
    sr: int,
    stats: dict | None = None,
):
    """Stream one sentence's PCM to the browser as binary frames.

    Each frame is a uint16 LE sample rate followed by 16-bit LE mono PCM.
    bot_audio_until tracks where scheduled playback ends (monotonic clock) -
    the session uses it to arm echo-aware barge-in."""
    for i in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[i : i + CHUNK_BYTES]
        try:
            await ws.send_bytes(struct.pack("<H", sr) + chunk)
        except (ConnectionError, RuntimeError) as e:
            # Client vanished mid-stream; the session teardown handles it.
            logger.debug(f"[web] audio stream send failed: {e}")
        if sess is not None:
            secs = len(chunk) / 2 / sr
            head = max(time.monotonic(), sess.get("bot_audio_until", 0.0))
            sess["bot_audio_until"] = head + secs
    if stats is not None:
        stats["audio_secs"] = stats.get("audio_secs", 0) + len(pcm) / 2 / sr


async def speak_text(
    ws: web.WebSocketResponse, sess: dict | None, text: str, stats: dict | None = None
):
    """Synthesize one line and stream it. Used for the greeting, apologies and
    timer announcements - turns use the pipelined Speaker instead."""
    pcm, sr = await synth_audio(text, stats)
    await stream_pcm(ws, sess, pcm, sr, stats)


async def send_greeting(ws: web.WebSocketResponse, sess: dict):
    """Stream the pre-synthesized greeting. Runs as the session's turn task,
    so barge-in (or a completed utterance) cancels it and cuts the audio."""
    greet = _GREETING
    gstats: dict = {}
    await send(ws, {"type": "text", "t": greet})
    try:
        await speak_text(ws, sess, greet, gstats)
    except Exception as e:
        logger.warning(f"[web] greeting TTS failed: {e}")
    await send(
        ws,
        {
            "type": "done",
            "reply": greet,
            "stats": {
                "asr_ms": 0,
                "llm_ms": 0,
                "tts_ms": round(gstats.get("tts_ms", 0)),
                "total_ms": 0,
            },
        },
    )


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Text-delta batching window for the client's chat bubble (seconds).
TEXT_FLUSH_SECS = 0.06
# Before the first chunk has gone out, a comma break is also a split point, so
# first audio starts after the opening clause instead of the first sentence.
FIRST_CLAUSE_MIN_CHARS = int(os.getenv("TTS_FIRST_CHUNK_CHARS", "28"))
FIRST_CLAUSE_TRIGGER = 48
# Safety valve for run-on output with no punctuation at all.
HARD_SPLIT_CHARS = 200


def split_sentences(buf: str, allow_clause: bool = False):
    """Return (chunks, leftover) - speakable chunks plus the trailing fragment.

    Sentence-ending punctuation always splits. allow_clause (used only before
    the first chunk of a turn) additionally splits after the first comma past
    FIRST_CLAUSE_MIN_CHARS, which cuts time-to-first-audio roughly in half
    without chopping off short openers like "Sure,". A run-on longer than
    HARD_SPLIT_CHARS is split at its last space before that limit so Piper
    never waits forever."""
    parts = _SENT_END.split(buf)
    if len(parts) > 1:
        return parts[:-1], parts[-1]
    if allow_clause and len(buf) >= FIRST_CLAUSE_TRIGGER:
        idx = buf.find(", ", FIRST_CLAUSE_MIN_CHARS - 1)
        if idx != -1:
            return [buf[: idx + 1]], buf[idx + 2 :]
    if len(buf) >= HARD_SPLIT_CHARS:
        cut = buf.rfind(" ", HARD_SPLIT_CHARS // 2, HARD_SPLIT_CHARS)
        if cut > 0:
            return [buf[:cut]], buf[cut + 1 :]
    return [], buf


# Session turn


async def send(ws: web.WebSocketResponse, obj: dict):
    try:
        await ws.send_json(obj)
    except (ConnectionError, RuntimeError) as e:
        # Dead socket; the handler's disconnect path cleans up the session.
        logger.debug(f"[web] send failed (socket closed?): {e}")


class Speaker:
    """Pipelined speech for one turn.

    Sentences flow through two workers: a synth worker calls the TTS MCP one
    sentence ahead, and a stream worker writes the finished PCM to the
    browser in order. So Piper synthesizes sentence N+1 while sentence N is
    playing (on a Pi, where Piper's RTF approaches 1, this removes the
    inter-sentence gaps almost entirely), and the LLM keeps streaming while
    Piper works instead of stalling behind every synth call."""

    def __init__(self, ws: web.WebSocketResponse, sess: dict, stats: dict):
        self.ws, self.sess, self.stats = ws, sess, stats
        self._syn_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._out_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._tasks: list[asyncio.Task] = []
        self.closed = False
        self._ttfa_sent = False

    def start(self):
        self._tasks = [
            asyncio.create_task(self._synth_worker()),
            asyncio.create_task(self._stream_worker()),
        ]

    async def enqueue(self, text: str):
        await self._syn_q.put(text)

    async def finish(self):
        """Normal end of turn: stop accepting sentences and wait until all
        audio has been streamed to the browser. If a worker died mid-drain,
        raise a regular error so the turn's error path runs."""
        await self._syn_q.put(None)
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        self.closed = True
        for r in results:
            if r is not None:
                # Workers return None on a clean drain; a non-None result is
                # an exception object (typically a CancelledError) meaning a
                # worker died mid-drain. Surface it so the turn's error path
                # runs.
                raise RuntimeError("speaker drain aborted (worker died)")

    async def aclose(self):
        """Cancel both workers (barge-in or error)."""
        if self.closed:
            return
        self.closed = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _synth_worker(self):
        try:
            while True:
                text = await self._syn_q.get()
                if text is None:
                    break
                try:
                    pcm, sr = await synth_audio(text, self.stats)
                except Exception as e:
                    # One failed sentence should not kill the turn.
                    logger.warning(f"[web] tts failed, skipping sentence: {e}")
                    continue
                await self._out_q.put((pcm, sr))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[web] speaker synth worker died: {e}")
        await self._out_q.put(None)

    async def _stream_worker(self):
        try:
            while True:
                item = await self._out_q.get()
                if item is None:
                    return
                if not self._ttfa_sent:
                    self._ttfa_sent = True
                    t0 = self.stats.get("turn_t0")
                    if t0:
                        self.stats["ttfa_ms"] = (time.perf_counter() - t0) * 1000
                    # Full-Duplex-Bench v1.5: t_resp = barge cut -> first
                    # audio of the response turn. Consumed once; a turn that
                    # never speaks clears the anchor in run_turn's finally.
                    cut_at = self.sess.get("fd_cut_at")
                    if cut_at:
                        self.sess["fd_cut_at"] = 0.0
                        resp_ms = (time.monotonic() - cut_at) * 1000
                        self.stats["fd_resp_ms"] = round(resp_ms)
                        _fd_record("fd_resp_ms", resp_ms)
                await stream_pcm(self.ws, self.sess, item[0], item[1], self.stats)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[web] speaker stream worker died: {e}")
            # Wake the synth worker so finish() cannot hang on a full queue.
            self._tasks[0].cancel()
