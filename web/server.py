"""Lava web bridge: run the full-duplex voice agent in the browser.

Browser mic -> [ASR MCP :8001] -> Groq LLM + [tools MCP :8003] -> [TTS MCP :8002] -> browser

Same brain, same tools, same voice - just a browser tab instead of the Pi.

Wire protocol (binary audio, low overhead):
  client -> server  binary frames: raw 16-bit LE mono PCM at 16 kHz, sent as
                    512-sample (32 ms) frames by the mic worklet. JSON text
                    frames carry control only ({"type": "ready"}).
  server -> client  binary frames: uint16 LE sample rate + 16-bit LE mono PCM,
                    one TTS chunk per frame. JSON text frames carry control
                    (text, transcript, barge, tool, done, error, silence, bye).

Run:  python -m web.server       (port 8080 by default)
"""

import asyncio
import base64
import json
import os
import random
import re
import struct
import time
from pathlib import Path

import httpx
from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from loguru import logger
from mcp import Client
from openai import AsyncOpenAI

from mcp_servers.common import LOGS_DIR, pcm16_to_wav, setup_logging

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")
# Fallback chain, tried in order: Groq's daily quota is per model, so the
# next model still has a fresh quota when the primary is drained.
_chain = [m.strip() for m in os.getenv("GROQ_LLM_MODELS", "").split(",") if m.strip()]
if not _chain:
    _chain = [GROQ_LLM_MODEL]
if GROQ_LLM_MODEL not in _chain:
    _chain.insert(0, GROQ_LLM_MODEL)
LLM_CHAIN = _chain
TURN_DEADLINE_SECS = float(os.getenv("TURN_DEADLINE_SECS", "45"))
# Idle handling: after IDLE_PROMPT_SECS of silence the agent asks if you are
# still there (a different wording each time); IDLE_CHECK_SECS of total
# silence stops the mic - the next tap resumes the same conversation.
IDLE_CHECK_SECS = float(os.getenv("IDLE_CHECK_SECS", "45"))
IDLE_PROMPT_SECS = min(float(os.getenv("IDLE_PROMPT_SECS", "25")),
                       IDLE_CHECK_SECS - 5.0)
CITY = os.getenv("DEFAULT_CITY", "Quezon City")
TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Manila")
ASR_URL = os.getenv("ASR_MCP_URL", "http://127.0.0.1:8001/mcp")
TTS_URL = os.getenv("TTS_MCP_URL", "http://127.0.0.1:8002/mcp")
TOOLS_URL = os.getenv("TOOLS_MCP_URL", "http://127.0.0.1:8003/mcp")

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
# PCM bytes per websocket message: small enough that barge-in lands ~150 ms
# after the audio the user hears; smaller adds client scheduling overhead.
CHUNK_BYTES = 4800
HISTORY_TURNS = 7
# Per-tab history (CONVERSATIONS): tapping the orb after an idle close
# resumes the same conversation.
CONVERSATIONS: dict[str, list[dict]] = {}
CONVERSATION_TTL = 60 * 60 * 24

SYSTEM_PROMPT = (
    "You are Lava, a friendly full-duplex voice assistant living in {city} (timezone {tz}). "
    "You are spoken to by voice and your replies are read aloud, so keep answers to at most "
    "two short, natural sentences. Never use markdown, bullet points, emojis, or stage "
    "directions - only words that sound good out loud. Use the available tools whenever they "
    "can answer the question (time, weather, search, wikipedia, timers and reminders, saved "
    "notes, lights, system status); reminders and notes are the same store - the notes "
    "tools. Tool results are ground truth for dynamic facts, so quote them exactly. If the "
    "user doubts an answer, call the tool again and stand by what it returns - never invent "
    "a different or empty result just to agree with them. If a tool returns nothing, say so "
    "honestly. Treat ambiguous or emotional remarks like 'I miss it', 'that one' or 'are "
    "you sure' as conversation, not commands: before adding or deleting anything, confirm "
    "what they want, and if a request could match several notes or timers, ask which one "
    "instead of guessing. After a tool result, speak the answer briefly and warmly. "
    "If the user interrupts or changes the subject, follow the new subject. "
    "The mic is in a room with a TV and phones, so some turns are their speech "
    "picked up by accident: fragments of show dialogue, ads, narration, or "
    "anyone who is clearly not talking to you. Reply to those with the single "
    "word SKIP and nothing else - it is never spoken aloud. When unsure, treat "
    "a plausible direct question or request as real and answer it."
).format(city=CITY, tz=TZ)

STATE = {
    "sessions": 0,
    "tools": [],
    "model": LLM_CHAIN[0],
    "chain": LLM_CHAIN,
    "live_sessions": set(),
    "by_ws": {},
}


# Fixed line fired once the mic is live (after the permission dialog).
# Pre-synthesized at startup, so it plays with ~10 ms to first audio.
# Deliberately not LLM-generated. Plays as the turn task, so barge-in cuts it
# like any other speech.
_GREETING = "Hi there! I'm Lava. What can I do today?"
# Spoken (rotating) when the session has gone quiet, before the mic turns off.
_IDLE_PROMPTS = [
    "Hi, are you still there?",
    "Still with me?",
    "You still there?",
    "Hello? Are you there?",
]
ASR_APOLOGY = "Sorry, my ears glitched. Could you say that again?"
LLM_APOLOGY = "Sorry, my brain hiccuped for a moment. Could you say that again?"

# Text -> (pcm, sr) cache. Seeded with the greeting and apologies at startup;
# short repeated lines (acknowledgments, timer announcements) accumulate at
# runtime. FIFO eviction, capped.
SPEECH_CACHE: dict[str, tuple[bytes, int]] = {}
SPEECH_CACHE_MAX = 40
SPEECH_CACHE_MAX_CHARS = 80


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


async def stream_pcm(ws: web.WebSocketResponse, sess: dict | None,
                     pcm: bytes, sr: int, stats: dict | None = None):
    """Stream one sentence's PCM to the browser as binary frames.

    Each frame is a uint16 LE sample rate followed by 16-bit LE mono PCM.
    bot_audio_until tracks where scheduled playback ends (monotonic clock) -
    the session uses it to arm echo-aware barge-in."""
    for i in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[i:i + CHUNK_BYTES]
        try:
            await ws.send_bytes(struct.pack("<H", sr) + chunk)
        except Exception:
            pass
        if sess is not None:
            secs = len(chunk) / 2 / sr
            head = max(time.monotonic(), sess.get("bot_audio_until", 0.0))
            sess["bot_audio_until"] = head + secs
    if stats is not None:
        stats["audio_secs"] = stats.get("audio_secs", 0) + len(pcm) / 2 / sr


async def speak_text(ws: web.WebSocketResponse, sess: dict | None,
                     text: str, stats: dict | None = None):
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
    await send(ws, {"type": "done", "reply": greet, "stats": {
        "asr_ms": 0, "llm_ms": 0, "tts_ms": round(gstats.get("tts_ms", 0)),
        "total_ms": 0}})


def _prune_conversations() -> None:
    """Drop conversations older than CONVERSATION_TTL."""
    now = time.time()
    stale = [cid for cid, (_h, ts) in CONVERSATIONS.items()
             if now - ts > CONVERSATION_TTL]
    for cid in stale:
        CONVERSATIONS.pop(cid, None)


# MCP plumbing (persistent connections to the three MCP servers)

class MCPConn:
    """Persistent MCP client session held in one background task.

    A connection that drops after a successful start (backend crash, restart)
    is re-established every few seconds, so a revived MCP server is picked up
    without restarting the bridge."""

    RECONNECT_SECS = 3.0
    HEALTH_SECS = 10.0

    def __init__(self, url: str, name: str):
        self.url, self.name = url, name
        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._closing: asyncio.Event | None = None
        self._ready: asyncio.Future | None = None
        self._fail: BaseException | None = None

    async def start(self):
        if self._task:
            await self._ready
            return
        self._closing = asyncio.Event()
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._own())
        await self._ready

    async def _own(self):
        first = True
        try:
            while not self._closing.is_set():
                try:
                    async with Client(self.url, read_timeout_seconds=30) as client:
                        self._client = client
                        self._fail = None
                        if not self._ready.done():
                            self._ready.set_result(None)
                        await self._supervise(client)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    self._fail = e
                    if first and not self._ready.done():
                        # Never connected: surface it once (startup reports the
                        # error) instead of spinning forever.
                        self._ready.set_exception(
                            ConnectionError(f"Cannot reach {self.name} MCP server "
                                            f"at {self.url}: {e}"))
                        return
                    logger.error(f"[web] {self.name} MCP session ended: {e}; "
                                 f"reconnecting in {self.RECONNECT_SECS:.0f}s")
                    await asyncio.sleep(self.RECONNECT_SECS)
                first = False
        except asyncio.CancelledError:
            pass
        finally:
            self._client = None
            self._task = None

    async def _supervise(self, client):
        """Wait for close, probing the backend so a silent death is noticed
        (the MCP client only errors on the next call otherwise)."""
        closer = asyncio.create_task(self._closing.wait())
        try:
            while not self._closing.is_set():
                await asyncio.wait({closer}, timeout=self.HEALTH_SECS)
                if self._closing.is_set():
                    return
                # A POST to /mcp gets an immediate HTTP response from a live
                # server (even 4xx); a dead one refuses the connection.
                async with httpx.AsyncClient() as probe:
                    await probe.post(self.url, timeout=5)
        finally:
            closer.cancel()

    async def call(self, tool: str, args: dict):
        if not self._client:
            reason = f" (last failure: {self._fail})" if self._fail else ""
            raise RuntimeError(f"{self.name} MCP server not connected{reason}")
        result = await self._client.call_tool(tool, args)
        if result.is_error:
            raise RuntimeError(f"{self.name}.{tool} failed: {result.content}")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads(result.content[0].text)

    async def close(self):
        if self._closing is None:
            return
        self._closing.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    @property
    def connected(self) -> bool:
        return self._client is not None


asr_conn = MCPConn(ASR_URL, "asr")
tts_conn = MCPConn(TTS_URL, "tts")
tools_conn = MCPConn(TOOLS_URL, "tools")


async def load_tools():
    """Pull the tool schemas from the tools MCP server (retry until it is up)."""
    while True:
        try:
            await tools_conn.start()
            lt = await tools_conn._client.list_tools()
            STATE["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": t.input_schema or {"type": "object", "properties": {}},
                    },
                }
                for t in lt.tools
                if t.name != "pop_finished_timers"
            ]
            logger.info(f"[web] loaded {len(STATE['tools'])} tools from tools MCP")
            return
        except Exception as e:
            logger.warning(f"[web] tools MCP not ready ({e}); retrying in 3 s")
            await asyncio.sleep(3)


TOOL_TIMEOUT_SECS = float(os.getenv("TOOL_TIMEOUT_SECS", "12"))


async def exec_tool(name: str, args: dict) -> str:
    t0 = time.perf_counter()
    try:

        res = await asyncio.wait_for(tools_conn.call(name, args), TOOL_TIMEOUT_SECS)
        out = json.dumps(res) if not isinstance(res, str) else res
    except asyncio.TimeoutError:
        out = json.dumps({"error": f"{name} timed out after {TOOL_TIMEOUT_SECS:.0f}s"})
    except Exception as e:
        out = json.dumps({"error": str(e)})
    logger.info(f"[web] tool {name}({json.dumps(args)}) -> {out[:120]} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    return out


# LLM (Groq, streaming, with the tool loop)

llm = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    timeout=30.0,
)

# 0.8 (down from 1.0): a voice agent pays for every quirky generation with
# several seconds of TTS, so consistency beats sampling diversity.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
# Bounds a runaway generation; voice replies are two short sentences.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))

RETRYABLE = (429, 500, 502, 503, 524, 529)


def _is_throttled(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or any(f" {c} " in f" {msg} " for c in RETRYABLE)


async def _open_stream(start: int, msgs: list[dict]):
    """Open a streaming completion, walking the model chain.

        On a 429 the next model in LLM_CHAIN still has a fresh quota (Groq's
        daily quota is per model). Returns (stream, model_index_used)."""
    last_err: Exception | None = None
    for i in range(start, len(LLM_CHAIN)):
        model = LLM_CHAIN[i]
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None
        try:
            stream = await llm.chat.completions.create(
                model=model,
                messages=msgs,
                tools=STATE["tools"] or None,
                stream=True,
                temperature=LLM_TEMPERATURE,
                top_p=0.95,
                max_tokens=LLM_MAX_TOKENS,
                extra_body=extra,
            )
            if i > start:
                logger.info(f"[web] LLM fell back to {model}")
            return stream, i
        except Exception as e:
            last_err = e
            if not _is_throttled(e):
                raise
            if i + 1 < len(LLM_CHAIN):
                logger.warning(f"[web] {model} rate-limited; trying {LLM_CHAIN[i + 1]}")
                await asyncio.sleep(1.0)
            else:
                logger.error(f"[web] all LLM models rate-limited: {str(e)[:200]}")
    assert last_err is not None
    raise last_err


async def llm_stream(messages: list[dict]):
    """Yield ('text', str) and ('tool_call', dict) tuples.

        Runs the tool loop: execute any tool calls, feed results back, keep
        streaming until prose."""
    msgs = [dict(m) for m in messages]
    model_idx = 0
    while True:
        stream, model_idx = await _open_stream(model_idx, msgs)
        text_buf = ""
        pending: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_buf += delta.content
                yield "text", delta.content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    e = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        e["id"] = tc.id
                    if tc.function and tc.function.name:
                        e["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        e["args"] += tc.function.arguments
        if pending:

            for idx in sorted(pending):
                e = pending[idx]
                try:
                    args = json.loads(e["args"]) if e["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                yield "tool_call", {"name": e["name"], "arguments": args}
                result = await exec_tool(e["name"], args)
                msgs.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": e["id"], "type": "function",
                     "function": {"name": e["name"], "arguments": e["args"]}}
                ]})
                msgs.append({"role": "tool", "tool_call_id": e["id"], "content": result})
            continue
        return


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
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
            return [buf[: idx + 1]], buf[idx + 2:]
    if len(buf) >= HARD_SPLIT_CHARS:
        cut = buf.rfind(" ", HARD_SPLIT_CHARS // 2, HARD_SPLIT_CHARS)
        if cut > 0:
            return [buf[:cut]], buf[cut + 1:]
    return [], buf


# VAD (Silero, server-side; the browser just streams raw PCM)
# Silero, not an energy VAD: it scores each 32 ms window by how
# human-speech-like it is, so TV speech plays out while the user still
# triggers.

from pipecat.audio.utils import exp_smoothing
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


class VADState:
    """Silero VAD + loudness gate + span gate.

    1. Silero scores each 32 ms window by how speech-like it is (music and
       TV noise stay below CONFIDENCE even when loud).
    2. A BS.1770 loudness gate (MIN_VOLUME) rejects far-away speech: the
       user is close to the mic, the TV is not. The gate has a fast attack
       and a slow release, so speech onsets are not eaten by smoothing.
    3. A span gate drops spans shorter than MIN_SAMPLES or sparser than
       MIN_VOICED_RATIO.

    A span opens on the first gated-speech window and closes after
    STOP_SECS of non-speech (or the 30 s cap). Spans include PRE_ROLL of
    audio from before the first voiced window, so the first word survives
    the loudness gate's rise time.

    A barge hint (BARGE_MS of gated speech) fires once per span; the session
    decides whether it cuts the bot (only while a turn is live or bot audio is
    playing)."""
    CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    START_SECS = float(os.getenv("VAD_START_SECS", "0.2"))
    STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.2"))
    MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.7"))
    MIN_SAMPLES = int(float(os.getenv("VAD_MIN_VOICED_MS", "450")) * SAMPLE_RATE // 1000)

    BARGE_SAMPLES = int(float(os.getenv("VAD_BARGE_MS", "200")) * SAMPLE_RATE // 1000)
    MIN_VOICED_RATIO = float(os.getenv("VAD_MIN_VOICED_RATIO", "0.30"))
    PRE_ROLL_BYTES = int(float(os.getenv("VAD_PRE_ROLL_MS", "120")) * SAMPLE_RATE * 2 / 1000)
    # Loudness smoothing: symmetric, matching the dynamics VAD_MIN_VOLUME was
    # tuned against. Fast-attack/slow-release sounds better in theory but in a
    # room with a TV it glues background speech into giant spans that the
    # voiced-ratio gate then discards - along with the command inside them.
    ATTACK = float(os.getenv("VAD_ATTACK", "0.2"))
    RELEASE = float(os.getenv("VAD_RELEASE", "0.2"))
    # Spans that fail the voiced-ratio gate are usually the user's command
    # glued to TV dialogue. Transcribe them anyway (Whisper's quality gate and
    # the LLM's SKIP instruction are the filters designed for this); spans with
    # almost no voiced audio still get dropped. Set false to restore hard
    # discarding.
    PROCESS_NOISY_SPANS = os.getenv("VAD_PROCESS_NOISY_SPANS", "true").lower() in (
        "1", "true", "yes", "on")
    MAX_SAMPLES = SAMPLE_RATE * 30
    MAX_IDLE_BYTES = 2 * SAMPLE_RATE * 2
    STOP_FRAMES = int(STOP_SECS * SAMPLE_RATE / FRAME_SAMPLES)

    def __init__(self):
        self._analyzer = SileroVADAnalyzer(
            sample_rate=SAMPLE_RATE,
            params=VADParams(
                confidence=self.CONFIDENCE,
                start_secs=self.START_SECS,
                stop_secs=self.STOP_SECS,
                min_volume=self.MIN_VOLUME,
            ),
        )

        self._analyzer.set_sample_rate(SAMPLE_RATE)

        self._prev_volume = 0.0
        self._pending = b""
        self.buf = bytearray()
        self.first_voiced = None
        self.last_voiced = None
        self.utterance = None
        self.barge_hint = False
        self._barge_fired = False
        self._silent = 0
        self._voiced_frames = 0
        self._span_frames = 0

    def feed(self, pcm: bytes):
        self._pending += pcm

        if self.first_voiced is None and len(self.buf) > self.MAX_IDLE_BYTES:
            del self.buf[: len(self.buf) - self.MAX_IDLE_BYTES]
        while len(self._pending) >= FRAME_BYTES:
            frame = self._pending[:FRAME_BYTES]
            self._pending = self._pending[FRAME_BYTES:]
            self.buf.extend(frame)
            confidence = self._analyzer.voice_confidence(frame)
            self._analyzer._volume_tracker.update(frame, SAMPLE_RATE)
            volume = self._analyzer._volume_tracker.volume
            alpha = self.ATTACK if volume > self._prev_volume else self.RELEASE
            volume = exp_smoothing(volume, self._prev_volume, alpha)
            self._prev_volume = volume
            speech = (confidence >= self.CONFIDENCE
                      and volume >= self.MIN_VOLUME)
            g = (len(self.buf) - FRAME_BYTES) // 2
            if speech:
                if self.first_voiced is None:
                    self.first_voiced = g
                    self._voiced_frames = 0
                    self._span_frames = 0
                    self._barge_fired = False
                self.last_voiced = g + FRAME_SAMPLES
                self._silent = 0
                self._voiced_frames += 1
                self._span_frames += 1
            else:
                if self.first_voiced is not None:
                    self._span_frames += 1
                    self._silent += 1
            span = (self.last_voiced - self.first_voiced
                    if self.first_voiced is not None and self.last_voiced is not None else 0)


            if (self.first_voiced is not None and not self._barge_fired
                    and self._voiced_frames * FRAME_SAMPLES >= self.BARGE_SAMPLES):
                self._barge_fired = True
                self.barge_hint = True
            if (self.first_voiced is not None and span >= self.MIN_SAMPLES
                    and (self._silent >= self.STOP_FRAMES or span >= self.MAX_SAMPLES)):
                voiced_ratio = (self._voiced_frames / self._span_frames
                                if self._span_frames else 0.0)
                voiced_ms = self._voiced_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE
                start = max(0, self.first_voiced * 2 - self.PRE_ROLL_BYTES)
                span_bytes = bytes(self.buf[start:])
                self.buf.clear()
                self.first_voiced = self.last_voiced = None
                self._silent = 0
                self.barge_hint = False
                self._barge_fired = False
                self._voiced_frames = self._span_frames = 0
                if (voiced_ratio < self.MIN_VOICED_RATIO
                        and (not self.PROCESS_NOISY_SPANS
                             or voiced_ms < self.MIN_SAMPLES * 1000 / SAMPLE_RATE)):
                    logger.info(
                        f"[vad] discarded span: {span / SAMPLE_RATE:.1f}s, only "
                        f"{voiced_ratio:.0%} speech ({voiced_ms:.0f} ms voiced)")
                else:
                    if voiced_ratio < self.MIN_VOICED_RATIO:
                        logger.info(
                            f"[vad] noisy span: {span / SAMPLE_RATE:.1f}s, "
                            f"{voiced_ratio:.0%} speech - transcribing anyway")
                    self.utterance = span_bytes


# Session turn

async def send(ws: web.WebSocketResponse, obj: dict):
    try:
        await ws.send_json(obj)
    except Exception:
        pass


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
            if isinstance(r, asyncio.CancelledError):
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
                await stream_pcm(self.ws, self.sess, item[0], item[1], self.stats)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[web] speaker stream worker died: {e}")
            # Wake the synth worker so finish() cannot hang on a full queue.
            self._tasks[0].cancel()


def _is_skip(text: str) -> bool:
    """True if the LLM's output is just its 'not addressed to me' marker.

        Skipped text is never spoken and is dropped from history."""
    return text.strip().strip(".!?\"' ").upper() == "SKIP"


async def run_turn(ws: web.WebSocketResponse, sess: dict, user_text: str, stats: dict):
    """ASR -> LLM(+tools) -> TTS for one utterance. Bounded by a hard deadline."""
    deadline = time.monotonic() + TURN_DEADLINE_SECS
    history = sess["history"]
    history.append({"role": "user", "content": user_text})
    reply_parts: list[str] = []
    speaker = Speaker(ws, sess, stats)
    speaker.start()
    llm_t0 = time.perf_counter()
    try:
        buf = ""
        spoken_any = False
        async for kind, payload in llm_stream(history):
            if time.monotonic() > deadline:
                raise TimeoutError("turn deadline exceeded")
            if kind == "tool_call":
                await send(ws, {"type": "tool", "name": payload["name"],
                                "args": json.dumps(payload["arguments"])[:120]})
                continue
            buf += payload
            await send(ws, {"type": "text", "t": payload})
            while True:
                chunks, buf = split_sentences(buf, allow_clause=not spoken_any)
                if not chunks:
                    break
                for s in chunks:
                    s = s.strip()
                    if s and not _is_skip(s):
                        reply_parts.append(s)
                        await speaker.enqueue(s)
                        spoken_any = True
        if buf.strip() and not _is_skip(buf):
            reply_parts.append(buf.strip())
            await speaker.enqueue(buf.strip())
        stats["llm_ms"] = (time.perf_counter() - llm_t0) * 1000
        await speaker.finish()
        reply = " ".join(reply_parts).strip()
        if reply:
            history.append({"role": "assistant", "content": reply})
        else:

            if history and history[-1].get("role") == "user":
                history.pop()

        if len(history) > 1 + HISTORY_TURNS * 2:
            del history[1 : -(HISTORY_TURNS * 2)]
        await send(ws, {"type": "done", "reply": reply, "stats": {
            "asr_ms": round(stats.get("asr_ms", 0)),
            "llm_ms": round(stats.get("llm_ms", 0)),
            "tts_ms": round(stats.get("tts_ms", 0)),
            "ttfa_ms": round(stats.get("ttfa_ms", 0)),
            "total_ms": round(stats.get("asr_ms", 0) + stats.get("llm_ms", 0) + stats.get("tts_ms", 0)),
        }})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[web] turn failed: {e}")
        # Cut anything still queued before apologizing, so the apology never
        # interleaves with leftover sentence audio.
        await speaker.aclose()
        try:
            await speak_text(ws, sess, LLM_APOLOGY, stats)
        except Exception:
            pass
        await send(ws, {"type": "error", "message": str(e)[:200], "apologized": True})
    finally:
        await speaker.aclose()


async def process_utterance(ws: web.WebSocketResponse, sess: dict, pcm: bytes):
    """One complete utterance: transcribe, then run the LLM/TTS turn."""
    try:
        await asyncio.wait_for(sess["turn_lock"].acquire(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.info("[web] dropping utterance (previous turn still running)")
        return
    try:
        stats = {"turn_t0": time.perf_counter()}
        t0 = time.perf_counter()
        try:
            res = await asr_conn.call("transcribe", {
                "audio_wav_b64": base64.b64encode(pcm16_to_wav(pcm, SAMPLE_RATE)).decode("ascii"),
            })
        except Exception as e:

            logger.error(f"[web] ASR failed: {e}")
            try:
                await speak_text(ws, sess, ASR_APOLOGY, stats)
            except Exception:
                pass
            await send(ws, {"type": "error", "message": f"ASR failed: {str(e)[:200]}"})
            return
        stats["asr_ms"] = (time.perf_counter() - t0) * 1000
        text = res.get("text", "").strip()
        logger.info(f"[web] ASR ({stats['asr_ms']:.0f} ms): {text!r}")
        if not text:
            await send(ws, {"type": "silence"})
            return
        sess["last_voice"] = time.monotonic()
        await send(ws, {"type": "transcript", "text": text})
        await run_turn(ws, sess, text, stats)
    finally:
        sess["turn_lock"].release()


async def handle_barge(ws: web.WebSocketResponse, sess: dict, turn_live: bool):
    """User speech onset while the bot may be talking: cut playback."""
    turn = sess["turn_task"]
    if turn_live and turn and not turn.done():
        turn.cancel()
        try:

            await turn
        except (asyncio.CancelledError, Exception):
            pass
    sess["turn_task"] = None
    # Playback is being cut client-side; drop the schedule.
    sess["bot_audio_until"] = 0.0
    await send(ws, {"type": "barge"})
    if turn_live:
        logger.info("[web] barge-in: user talking over the bot")
    else:
        logger.debug("[web] barge-in onset: no active turn, "
                     "clearing client playback")


# HTTP + WebSocket handlers

async def h_status(request: web.Request):
    return web.json_response({
        "ok": True,
        "model": STATE["model"],
        "chain": STATE["chain"],
        "provider": "groq",
        "city": CITY,
        "tools": [t["function"]["name"] for t in STATE["tools"]],
        "sessions": STATE["sessions"],
    })


async def h_index(request: web.Request):
    return web.FileResponse(Path(__file__).parent / "index.html")


def _icon_handler(filename: str):
    """Serve the pre-rendered lava-orb icons (Safari ignores data-URI favicons)."""
    async def h(request: web.Request):
        return web.FileResponse(
            Path(__file__).parent / filename,
            headers={"Cache-Control": "no-cache"},
        )
    return h


async def h_ws(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=25)
    await ws.prepare(request)

    client_id = request.query.get("cid") or None
    STATE["sessions"] += 1
    STATE["live_sessions"].add(ws)

    stored = CONVERSATIONS.get(client_id) if client_id else None
    prior = stored[0] if stored else None
    sess = {
        "client_id": client_id,
        "history": list(prior) if prior else [{"role": "system", "content": SYSTEM_PROMPT}],
        "vad": VADState(),
        "turn_lock": asyncio.Lock(),
        "turn_task": None,
        "ready": False,
        "greeting_sent": False,
        "bot_audio_until": 0.0,
        "last_voice": time.monotonic(),
    }
    resumed = bool(prior)
    logger.info(f"[web] session started (total {STATE['sessions']}, "
                f"resumed={resumed})")
    idle_task = asyncio.create_task(idle_watch(ws, sess))
    STATE["by_ws"][ws] = sess


    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "ready":

                    sess["ready"] = True
                    sess["last_voice"] = time.monotonic()
                    logger.info("[web] client ready - mic live, greeting")
                    if not sess["greeting_sent"]:
                        sess["greeting_sent"] = True


                        sess["turn_task"] = asyncio.create_task(send_greeting(ws, sess))
                continue
            if msg.type != WSMsgType.BINARY:
                continue
            if not sess["ready"]:
                continue
            sess["vad"].feed(msg.data)

            if sess["vad"].barge_hint:
                sess["vad"].barge_hint = False
                turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
                # Cut the bot while a turn is live or its audio is (still)
                # scheduled to be playing.
                if turn_live or time.monotonic() < sess["bot_audio_until"]:
                    await handle_barge(ws, sess, turn_live)
                continue
            if sess["vad"].utterance is None:
                continue
            utterance = sess["vad"].utterance
            sess["vad"].utterance = None
            sess["last_voice"] = time.monotonic()
            if sess["turn_task"] and not sess["turn_task"].done():
                sess["turn_task"].cancel()
            sess["turn_task"] = asyncio.create_task(
                process_utterance(ws, sess, utterance))
    except Exception as e:
        logger.error(f"[web] ws error: {e}")
    finally:
        idle_task.cancel()
        STATE["live_sessions"].discard(ws)
        STATE["by_ws"].pop(ws, None)
        if sess["turn_task"] and not sess["turn_task"].done():
            sess["turn_task"].cancel()

        cid = sess.get("client_id")
        if cid:
            hist = sess["history"]
            if len(hist) > 1:
                CONVERSATIONS[cid] = (hist, time.time())
            else:
                CONVERSATIONS.pop(cid, None)
            _prune_conversations()
        STATE["sessions"] -= 1
        logger.info(f"[web] session ended ({STATE['sessions']} active)")
        if not ws.closed:
            await ws.close()
    return ws


async def idle_watch(ws: web.WebSocketResponse, sess: dict):
    """Check in on a quiet session, then close it if the silence continues.

    The browser streams raw PCM continuously - silence included - so presence
    means detected speech or an active turn, not incoming frames. After
    IDLE_PROMPT_SECS of quiet the agent asks if the user is still there
    (speaking over the check-in barges it like any other speech); if the
    silence reaches IDLE_CHECK_SECS the mic stops and the orb waits for a
    tap (history is restored by client id)."""
    prompted = False
    try:
        while not ws.closed:
            quiet = time.monotonic() - sess["last_voice"]
            turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
            if turn_live or quiet < IDLE_PROMPT_SECS:
                prompted = False
                await asyncio.sleep(1)
                continue
            if prompted:
                logger.info("[web] idle timeout - no answer to the check-in, "
                            "closing session")
                await send(ws, {"type": "bye"})
                return
            prompted = True
            prompt = random.choice(_IDLE_PROMPTS)
            logger.info(f"[web] idle {quiet:.0f}s - checking in")
            await send(ws, {"type": "text", "t": prompt})
            try:
                await speak_text(ws, sess, prompt, {})
            except Exception as e:
                logger.warning(f"[web] idle check-in TTS failed: {e}")
            await send(ws, {"type": "done", "reply": prompt, "stats": {}})
            await asyncio.sleep(max(1.0, IDLE_CHECK_SECS - IDLE_PROMPT_SECS))
    except asyncio.CancelledError:
        pass


async def announce_timers():
    """Speak finished timers (proactive speech). Browser-started timers would
        otherwise finish silently - the web bridge has no other listener."""
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(1.0)
        if not tools_conn.connected:
            continue
        try:
            done = (await tools_conn.call("pop_finished_timers", {})).get("finished", [])
        except Exception:
            continue
        for t in done:
            msg = f"Heads up, your {t['label']} timer is done."
            logger.info(f"[web] timer finished: {t['label']}")
            for ws in list(STATE["live_sessions"]):
                await send(ws, {"type": "text", "t": msg})
                try:
                    sess = STATE["by_ws"].get(ws)
                    await speak_text(ws, sess, msg, {})
                    await send(ws, {"type": "done", "reply": msg, "stats": {}})
                except Exception:
                    pass


async def _preseed_speech_cache():
    """Pre-synthesize the fixed lines so the greeting and error apologies
    play with no TTS wait at all."""
    for text in (_GREETING, ASR_APOLOGY, LLM_APOLOGY, *_IDLE_PROMPTS):
        try:
            pcm, sr = await synth_audio(text)
            SPEECH_CACHE[text] = (pcm, sr)
            logger.info(f"[web] pre-synthesized {len(pcm) / 2 / sr:.1f}s: {text!r}")
        except Exception as e:
            logger.warning(f"[web] pre-synth failed for {text!r}: {e}")


async def on_startup(app: web.Application):
    app["llm_task"] = asyncio.create_task(load_tools())
    app["timer_task"] = asyncio.create_task(announce_timers())
    for conn in (asr_conn, tts_conn):
        try:
            await conn.start()
            logger.info(f"[web] connected to {conn.name} MCP at {conn.url}")
        except Exception as e:
            logger.error(f"[web] cannot reach {conn.name} MCP at {conn.url}: {e}")
    # Only after the TTS connection is up - the preseed calls the synthesizer.
    app["preseed_task"] = asyncio.create_task(_preseed_speech_cache())


async def on_shutdown(app: web.Application):
    for conn in (asr_conn, tts_conn, tools_conn):
        await conn.close()


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", h_index)
    app.router.add_get("/api/status", h_status)
    app.router.add_get("/ws", h_ws)
    app.router.add_get("/favicon.png", _icon_handler("favicon.png"))
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    setup_logging("web")
    web.run_app(make_app(), host="127.0.0.1", port=WEB_PORT)
