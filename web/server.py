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
import contextlib
import json
import os
import random
import re
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from loguru import logger
from mcp import Client
from openai import AsyncOpenAI

from mcp_servers import spotify
from mcp_servers.common import LOGS_DIR, pcm16_to_wav, setup_logging, trim_silence

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
IDLE_CHECK_SECS = float(os.getenv("IDLE_CHECK_SECS", "70"))
IDLE_PROMPT_SECS = min(float(os.getenv("IDLE_PROMPT_SECS", "50")),
                       IDLE_CHECK_SECS - 5.0)
CITY = os.getenv("DEFAULT_CITY", "Quezon City")
TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Manila")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI",
                                 "http://127.0.0.1:8080/callback")
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
    "notes, lights, music playback on Spotify, system status); reminders and notes are the same store - the notes "
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
    "a plausible direct question or request as real and answer it. The same is "
    "true of singing: short song-title or artist fragments that arrive while "
    "music plays are usually the music, not the user."
).format(city=CITY, tz=TZ)

STATE = {
    "sessions": 0,
    "tools": [],
    "model": LLM_CHAIN[0],
    "chain": LLM_CHAIN,
    "live_sessions": set(),
    "by_ws": {},
    # Spotify playback mirror (drives music-aware mic gating). Kept fresh by
    # the 1 Hz infra loop via the tools MCP's cached music_state tool.
    "music_playing": False,
    "music_seen_at": 0.0,
    "music_mode_until": 0.0,
    # Full-duplex overlap accounting (Full-Duplex-Bench style): backchannels
    # recognized while the bot talked (kept talking) vs real barge-ins (cut).
    "backchannels": 0,
    "barge_ins": 0,
    # Full-Duplex-Bench v1.5: per-scenario behavior counts plus rolling
    # t_stop / t_resp samples (ms). See classify_overlap() and fd_summary().
    "fd_behaviors": {"interruption": 0, "backchannel": 0,
                     "talk_to_other": 0, "background": 0},
    "fd_stop_ms": [],
    "fd_resp_ms": [],
}


def _fd_record(key: str, ms: float) -> None:
    """Append one t_stop / t_resp sample (rolling window of 200)."""
    buf = STATE[key]
    buf.append(round(ms))
    if len(buf) > 200:
        del buf[: len(buf) - 200]


def fd_summary() -> dict:
    """Rolling Full-Duplex-Bench metrics for /api/status and the bench."""
    def stat(buf: list[int]) -> dict | None:
        if not buf:
            return None
        xs = sorted(buf)
        return {"n": len(xs),
                "p50": xs[len(xs) // 2],
                "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))]}
    return {"behaviors": dict(STATE["fd_behaviors"]),
            "t_stop_ms": stat(STATE["fd_stop_ms"]),
            "t_resp_ms": stat(STATE["fd_resp_ms"])}

# --- music mode --------------------------------------------------------------
# Spotify playing out loud (usually the web player tab, so the same laptop's
# speakers feed the mic) is the ghost in this agent: vocals survive the VAD,
# Whisper transcribes lyrics ("I declare a Benin song"), and innocent
# fragments have triggered plays. While music plays we:
#   1. tighten the VAD gates (MUSIC_MODE_* below),
#   2. handle control words (stop/pause/next/louder/volume 40/...) directly,
#      no LLM — sub-second and immune to hallucinated tool calls,
#   3. tell the LLM what the music is so lyric fragments get SKIPped.
MUSIC_MODE_GRACE = float(os.getenv("MUSIC_MODE_GRACE_SECS", "20"))
MUSIC_MODE_MIN_RATIO = float(os.getenv("MUSIC_MODE_MIN_RATIO", "0.45"))
MUSIC_MODE_VOLUME_DELTA = float(os.getenv("MUSIC_MODE_VOLUME_DELTA", "0.04"))
# How long the bridge keeps polling music_state after music stopped (Sees an
# external pause on your phone, e.g.) before dropping back to lazy checks.
MUSIC_POLL_IDLE_MAX = float(os.getenv("MUSIC_POLL_IDLE_MAX_SECS", "300"))
FAST_MUSIC_COMMANDS = os.getenv("FAST_MUSIC_COMMANDS", "true").lower() in (
    "1", "true", "yes", "on")

# Backchannels: a short acknowledgment ("uh-huh", "mm-hmm", "yeah") while the
# bot is talking is NOT an interruption - the bot should keep speaking. The
# barge-in cut is therefore deferred until the span transcribes; if the text is
# a pure backchannel the cut is dropped and the bot continues. A watchdog
# (BACKCHANNEL_CUT_MAX_SECS) still forces the cut if a real interruption keeps
# the span open past the cap, so responsiveness is bounded.
BACKCHANNEL_CUT_MAX_SECS = float(os.getenv("BACKCHANNEL_CUT_MAX_SECS", "1.0"))
BACKCHANNEL_MAX_WORDS = int(os.getenv("BACKCHANNEL_MAX_WORDS", "4"))
_BACKCHANNEL_TEXTS = {
    "uhhuh", "uhhuhhuh", "uh", "uhhuhuh", "uhuh", "uh huh", "uhhuh uh",
    "mhmm", "mmhmm", "mhm", "mm", "hmm", "hmmhmm", "mhmhmm", "mmm", "mmmm",
    "yeah", "yep", "yea", "yup", "yess", "ya", "yeah yeah", "yep yep",
    "right", "sure", "exactly", "got it", "gotcha", "okay", "ok", "k", "kk",
    "cool", "nice", "great", "good", "perfect", "absolutely", "indeed",
    "of course", "for sure", "yes", "yes sir", "yes maam", "affirmative",
    "that's right", "thats right", "i see", "i got it", "understood",
    "roger", "copy that", "sounds good", "makes sense", "no problem",
    "no worries", "sure thing", "my bad", "oh okay", "oh i see",
}

# --- Full-Duplex-Bench v1.5: TALKING_OTHER and BKG_SPEECH ---------------------
# The other two overlap scenarios are speech that is NOT addressed to the
# agent. The desired behavior there is the opposite of an interruption:
# HOLD THE FLOOR (keep talking, do not cut). We detect the cheap, high-
# precision case - a vocative opener ("hey John", "Maria, ...") or an explicit
# "not you" cue ("you guys", "talk to you") - and drop the cut. Far-field
# background speech is already handled acoustically: it stays under the
# VAD's loudness gate, and the music-mode lyric guard drops what survives.
FD_HOLD_FLOOR = os.getenv("FD_HOLD_FLOOR", "true").lower() in (
    "1", "true", "yes", "on")
_FD_VOCATIVES = {
    "hey", "hi", "hello", "yo", "eh", "hey hey", "hmm", "uh",
    "ok", "okay", "alright", "alrighty", "come", "listen", "wait",
    "so", "well", "um", "uhh", "erm",
}
_FD_NOT_YOU_PATTERNS = (
    re.compile(r"\b(you guys|you all|everyone|everybody|folks|guys|babe|hon|honey)\b"),
    re.compile(r"\b(talk|talking|speaking|saying|said|asked|asking) (to|with) you\b"),
    re.compile(r"\bnot (to|for) you\b"),
)
# Sentence-initial discourse markers that are often followed by a comma but
# are NOT an address ("Well, ...", "So, ..."). Excluded from the
# "Name, ..." vocative rule so they keep cutting as normal turns.
_FD_DISCOURSE_MARKERS = {
    "well", "so", "but", "and", "or", "yet", "now", "then", "here", "there",
    "actually", "basically", "literally", "honestly", "okay", "ok", "alright",
    "hey", "hi", "hello", "yo", "um", "uh", "erm", "come", "listen", "wait",
    "right", "yeah", "yep", "sure", "no", "yes", "look", "mean", "guess",
}
# The agent's own name(s): "Hey Lava, ..." addresses the agent, so it is an
# interruption, not talk-to-other.
_FD_AGENT_NAMES = {"lava"}
# Compact lookup (spaces, hyphens and apostrophes removed): "mm-hmm",
# "mmhmm" and "mm hmm" must all classify the same way.
_BACKCHANNEL_COMPACT = {re.sub(r"[^a-z]", "", x) for x in _BACKCHANNEL_TEXTS}

MUSIC_NOTE = (
    "CONTEXT UPDATE (not from the user): Spotify music is playing out loud in "
    "this room, and part of it reaches the microphone. Utterances that read "
    "like song lyrics, a track or artist name, or 'the next song' repetition "
    "with no command verb or clear question are almost certainly the music "
    "itself - reply SKIP to those. Accept music commands only when they start "
    "with a control verb (stop, pause, resume, play, skip, next, previous, "
    "restart, shuffle, repeat, volume, louder, quieter) or are a clear "
    "question about the music (what is playing, which devices). Never call "
    "music_play for lyric or song-name fragments."
)


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
    is re-established every couple of seconds, so a revived MCP server is
    picked up without restarting the bridge. In-flight calls that hit the
    drop - a dead "stop the music" used to be exactly that - are retried once
    on a fresh connection."""

    RECONNECT_SECS = 2.0
    HEALTH_SECS = 10.0

    # Error strings that mean the transport died (vs a tool returning an
    # error result, which must NOT be retried - that would skip two tracks).
    _DEAD_MARKERS = ("closed", "taskgroup", "broken", "eof", "disconnect",
                     "reset", "refused", "not connected", "corrupt", "handshake")

    def __init__(self, url: str, name: str):
        self.url, self.name = url, name
        host = urlsplit(url)
        self._host = host.hostname or "127.0.0.1"
        self._port = host.port or (443 if host.scheme == "https" else 80)
        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._closing: asyncio.Event | None = None
        self._ready: asyncio.Future | None = None
        self._fail: BaseException | None = None

    async def start(self):
        async with self._start_lock:
            await self._kick()

    async def _kick(self, force: bool = False):
        """(Re)establish the session. force=True tears down a live task first.

        A plain start() returns instantly when a supervisor task exists, which
        once let callers "succeed" into a self._client=None state and crash on
        the next attribute access; force=True is for the retry path in call()."""
        if not force and self._task and not self._task.done():
            await self._ready
            return
        if force:
            self._closing.set()               # let the supervise loop exit
            task = self._task
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, 1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    task.cancel()
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
        """Wait for close, probing the backend port so a silent death is
        noticed (the MCP client only errors on the next call otherwise).

        The probe is a raw TCP connect: the old empty POST /mcp made every
        server fabricate and terminate an MCP transport every 10 s ("Created
        new transport ... 400 Bad Request / Terminating session" churn)."""
        closer = asyncio.create_task(self._closing.wait())
        try:
            while not self._closing.is_set():
                await asyncio.wait({closer}, timeout=self.HEALTH_SECS)
                if self._closing.is_set():
                    return
                try:
                    reader, writer = await asyncio.open_connection(
                        self._host, self._port)
                except OSError as e:
                    raise ConnectionError(
                        f"{self._host}:{self._port} not listening: {e}") from e
                with contextlib.suppress(Exception):
                    writer.close()
                    await writer.wait_closed()
        finally:
            closer.cancel()

    async def _exchange(self, tool: str, args: dict):
        client = self._client
        result = await client.call_tool(tool, args)
        if result.is_error:
            raise RuntimeError(f"{self.name}.{tool} failed: {result.content}")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads(result.content[0].text)

    async def call(self, tool: str, args: dict):
        if not self._client:
            # Self-heal: the connection was never established (MCP was down at
            # bridge startup) or died. Try once now - raises if still down.
            try:
                await self.start()
            except Exception as e:
                reason = f" (last failure: {self._fail or e})"
                raise RuntimeError(f"{self.name} MCP server not connected{reason}") from e
        try:
            return await self._exchange(tool, args)
        except Exception as e:
            msg = str(e).lower()
            if not any(m in msg for m in self._DEAD_MARKERS):
                raise                          # a tool result error, not a drop
            logger.warning(f"[web] {self.name}.{tool} connection dropped "
                           f"({str(e)[:80]}) - reconnecting once")
            self._client = None
            try:
                async with self._start_lock:
                    await self._kick(force=True)
            except Exception as e2:
                raise RuntimeError(f"{self.name} MCP server not connected "
                                   f"(last failure: {e2 or self._fail})") from e
            if not self._client:
                raise RuntimeError(f"{self.name} MCP server not connected "
                                   f"(last failure: {self._fail})")
            return await self._exchange(tool, args)

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
                if t.name not in ("pop_finished_timers", "music_state")
            ]
            logger.info(f"[web] loaded {len(STATE['tools'])} tools from tools MCP")
            if FAST_MUSIC_COMMANDS:
                logger.info("[web] fast music commands enabled (music mode)")
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

# Post-ASR correction pass. Groq's Whisper endpoint rejects `initial_prompt`
# (400 unknown param), so the model can't be biased toward the user's
# vocabulary at the API level. Instead, a fast LLM call fixes obvious
# mis-hearings (proper nouns, homophones) using the recent conversation as
# context. It never changes meaning and falls back to the raw transcript on
# any error, so it can only help. Set ASR_CORRECT=false to disable.
ASR_CORRECT = os.getenv("ASR_CORRECT", "true").lower() in ("1", "true", "yes", "on")
# Skip the pass for very short turns ("hi", "next") where there is nothing to
# fix and a call would only add latency.
ASR_CORRECT_MIN_WORDS = int(os.getenv("ASR_CORRECT_MIN_WORDS", "3"))
# How many recent messages of context to hand the corrector.
ASR_CORRECT_CTX = int(os.getenv("ASR_CORRECT_CTX", "6"))
# Hard cap so a slow corrector can never stall a turn; on timeout the raw
# transcript is used.
ASR_CORRECT_TIMEOUT = float(os.getenv("ASR_CORRECT_TIMEOUT", "2.5"))
# But the turn never waits this long for the proofread regardless of the
# timeout above: the transcript is shown instantly and the LLM starts on the
# raw text after the grace expires, so a throttled corrector costs ~nothing
# instead of up to ASR_CORRECT_TIMEOUT of dead air. 0.5 s is the sweet spot:
# a healthy corrector answers in ~300-400 ms, and the common case in the
# logs was a full-grace timeout that bought nothing - every extra 100 ms
# here is dead air on the critical path of every turn.
ASR_CORRECT_GRACE = float(os.getenv("ASR_CORRECT_GRACE_SECS", "0.5"))
# Which model does the proofread. Defaults to the first NON-reasoning model in
# the chain: a simple correction task should not pay the reasoning tax (or
# burn the daily quota) of the heavy primary. Override to force one.
ASR_CORRECT_MODEL = os.getenv("ASR_CORRECT_MODEL", "")
# Whisper's own confidence signal (avg_logprob, lower = less sure) decides
# WHETHER to proofread at all. Proofreading every turn cost ~500 ms of dead
# air on the critical path and, per the logs, was a net loss (more timeouts
# than fixes) because a healthy transcript needs no repair. Run the pass only
# when Whisper itself is unsure (avg_logprob at or below this); confident
# transcripts skip it for free. Set to a very negative number to always
# proofread, or 0 to never.
ASR_PROOFREAD_LOGPROB_MAX = float(os.getenv("ASR_PROOFREAD_LOGPROB_MAX", "-0.7"))

RETRYABLE = (429, 500, 502, 503, 524, 529)


def _is_throttled(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or any(f" {c} " in f" {msg} " for c in RETRYABLE)


# A model that just refused a request (429) refuses the next one too -
# Groq's free-tier quota is per model. Retrying it every turn paid a full
# round trip + a 429 + a 1 s sleep before falling back, on EVERY turn.
# Putting the model on cooldown makes the fallback instant for the rest of
# the outage window; the last model is always tried even while cooling down
# (a 429 from it is better than no brain at all).
_model_cooldown: dict[str, float] = {}
LLM_COOLDOWN_SECS = float(os.getenv("LLM_COOLDOWN_SECS", "60"))


async def _open_stream(start: int, msgs: list[dict]):
    """Open a streaming completion, walking the model chain.

        On a 429 the next model in LLM_CHAIN still has a fresh quota (Groq's
        daily quota is per model) and the refused model goes on cooldown so
        the next turn skips it without paying for the refusal again.
        Returns (stream, model_index_used)."""
    last_err: Exception | None = None
    now = time.monotonic()
    for i in range(start, len(LLM_CHAIN)):
        model = LLM_CHAIN[i]
        if _model_cooldown.get(model, 0.0) > now and i + 1 < len(LLM_CHAIN):
            logger.debug(f"[web] {model} in cooldown - skipping")
            continue
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
                _model_cooldown[model] = time.monotonic() + LLM_COOLDOWN_SECS
                logger.warning(f"[web] {model} rate-limited "
                               f"(cooldown {LLM_COOLDOWN_SECS:.0f} s); trying "
                               f"{LLM_CHAIN[i + 1]}")
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
            # Parse every streamed call first, then run them concurrently:
            # models occasionally request two independent tools (time +
            # weather), and running them serially paid both round trips.
            calls: list[tuple[dict, dict]] = []
            for idx in sorted(pending):
                e = pending[idx]
                try:
                    args = json.loads(e["args"]) if e["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append((e, args))
            for e, args in calls:
                yield "tool_call", {"name": e["name"], "arguments": args}
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": e["id"], "type": "function",
                 "function": {"name": e["name"], "arguments": e["args"]}}
                for e, _ in calls]})
            results = await asyncio.gather(
                *(exec_tool(e["name"], args) for e, args in calls))
            for (e, _), result in zip(calls, results):
                msgs.append({"role": "tool", "tool_call_id": e["id"],
                             "content": result})
            continue
        return


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
    # Lower close bar for spans that opened on a barge hint (near-field, already
    # passed the loudness gate): a quick backchannel ("mm-hmm", ~300 ms) closes
    # and gets classified instead of waiting out the 450 ms far-field bar.
    BARGE_ORIGIN_MIN_SAMPLES = int(
        float(os.getenv("VAD_BARGE_ORIGIN_MIN_VOICED_MS", "150")) * SAMPLE_RATE // 1000)

    BARGE_SAMPLES = int(float(os.getenv("VAD_BARGE_MS", "200")) * SAMPLE_RATE // 1000)
    MIN_VOICED_RATIO = float(os.getenv("VAD_MIN_VOICED_RATIO", "0.30"))
    PRE_ROLL_BYTES = int(float(os.getenv("VAD_PRE_ROLL_MS", "120")) * SAMPLE_RATE * 2 / 1000)
    # Loudness smoothing: symmetric, matching the dynamics VAD_MIN_VOLUME was
    # tuned against. Fast-attack/slow-release sounds better in theory but in a
    # room with a TV it glues background speech into giant spans that the
    # voiced-ratio gate then discards - along with the command inside them.
    ATTACK = float(os.getenv("VAD_ATTACK", "0.2"))
    RELEASE = float(os.getenv("VAD_RELEASE", "0.2"))
    # Music mode: while Spotify is playing out loud the mic hears it too.
    # Raise the bar so song vocals don't become turns (handled by the
    # command fast path and the LLM music note instead).
    MUSIC_RATIO = max(0.45, MUSIC_MODE_MIN_RATIO)
    MUSIC_DELTA = MUSIC_MODE_VOLUME_DELTA
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

    @classmethod
    def music_mode(cls) -> bool:
        return time.monotonic() < STATE["music_mode_until"]

    @classmethod
    def ratio_floor(cls, music: bool) -> float:
        return cls.MUSIC_RATIO if music else cls.MIN_VOICED_RATIO

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
        self.last_ratio = 1.0    # voiced ratio of the last completed span
        self.barge_hint = False
        self.barge_origin = False  # True if the current span opened on a barge
        self._barge_fired = False
        self._silent = 0
        self._voiced_frames = 0
        self._span_frames = 0

    def feed(self, pcm: bytes):
        self._pending += pcm

        music = self.music_mode()
        conf_floor = self.CONFIDENCE + (self.MUSIC_DELTA if music else 0.0)
        vol_floor = self.MIN_VOLUME + (self.MUSIC_DELTA if music else 0.0)

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
            speech = (confidence >= conf_floor and volume >= vol_floor)
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
                # This span opened over the bot's own voice: it is near-field
                # and will be decided by the backchannel/interruption logic
                # downstream, so let it close on a short pause (a quick
                # "mm-hmm" classifies instead of hanging out the 450 ms
                # far-field bar).
                self.barge_origin = True
            # A span that opened on a barge hint is near-field (it already
            # passed the loudness gate) and is being decided by the
            # backchannel/interruption logic downstream - let it close on a
            # short pause so a quick "mm-hmm" is classified instead of hanging.
            # Normal spans keep the full MIN_SAMPLES bar (far-field TV noise).
            min_close = (self.BARGE_ORIGIN_MIN_SAMPLES if self.barge_origin
                         else self.MIN_SAMPLES)
            if (self.first_voiced is not None and span >= min_close
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
                self.barge_origin = False
                self._voiced_frames = self._span_frames = 0
                floor = self.ratio_floor(music)
                self.last_ratio = voiced_ratio
                if (voiced_ratio < floor
                        and (not self.PROCESS_NOISY_SPANS
                             or voiced_ms < self.MIN_SAMPLES * 1000 / SAMPLE_RATE)):
                    logger.info(
                        f"[vad] discarded span: {span / SAMPLE_RATE:.1f}s, only "
                        f"{voiced_ratio:.0%} speech ({voiced_ms:.0f} ms voiced)"
                        + (" [music gate]" if music else ""))
                else:
                    if voiced_ratio < floor:
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


def _is_skip(text: str) -> bool:
    """True if the LLM's output is just its 'not addressed to me' marker.

        Skipped text is never spoken and is dropped from history."""
    return text.strip().strip(".!?\"' ").upper() == "SKIP"


def is_backchannel(text: str) -> bool:
    """True if the transcript is a pure acknowledgment, not a real turn.

    Decides whether speech that overlapped the bot's own voice was a
    backchannel (keep talking) or an interruption (cut). Conservative on
    purpose: anything longer than BACKCHANNEL_MAX_WORDS, or not a known
    filler, is treated as a real turn so a genuine interruption is never
    swallowed. Runs on the raw (pre-proofread) transcript, so a Whisper
    mis-hearing of "uh-huh" that lands on a normal word still cuts - the
    safe direction. Matching is space-insensitive ("mm-hmm", "mmhmm" and
    "mm hmm" are the same phrase)."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t.split()) > BACKCHANNEL_MAX_WORDS:
        return False
    return t.replace(" ", "") in _BACKCHANNEL_COMPACT


def is_not_addressed(text: str) -> bool:
    """True if the speech reads as being for someone else (TALKING_OTHER).

    Full-Duplex-Bench v1.5: the desired behavior for talk-to-other and
    background speech is to HOLD THE FLOOR, not cut. We only claim it on
    high-precision cues so a real interruption is never swallowed:
      1. explicit not-you cue - "you guys", "everyone", "talk to you",
         "not to you";
      2. an address - a proper-noun-ish capitalized word (a name) either
         followed by a comma ("Maria, are you coming?") or preceded by a
         vocative ("hey John, ..."). Whisper capitalizes sentence-initial
         words, so a lone capitalized first word proves nothing - the comma
         or the vocative is what marks it as an address. Discourse markers
         ("Well,", "So,") are excluded so they keep cutting as normal turns.
    Runs on the RAW transcript (before the proofread pass may rewrite it)."""
    if not FD_HOLD_FLOOR:
        return False
    t = text.strip()
    if not t or len(t.split()) > 12:
        return False
    low = t.lower()
    for pat in _FD_NOT_YOU_PATTERNS:
        if pat.search(low):
            return True
    parts = re.split(r"[\s]+", t)
    # "hey John, ..." / "wait Maria -" : vocative + capitalized name.
    if len(parts) >= 2 and parts[0].lower().rstrip(".,!") in _FD_VOCATIVES:
        nxt = parts[1].strip(".,!?\"'()")
        if (nxt[:1].isupper() and nxt.isalpha() and len(nxt) >= 2
                and nxt.lower() not in _FD_AGENT_NAMES):
            return True
    # "Maria, ..." : capitalized name + comma (an address, not a sentence
    # start - Whisper keeps the comma right after the name).
    first = parts[0]
    if first.endswith(","):
        name = first[:-1].strip(".,!?\"'()")
        if (name[:1].isupper() and name.isalpha() and len(name) >= 2
                and name.lower() not in _FD_DISCOURSE_MARKERS
                and name.lower() not in _FD_AGENT_NAMES):
            return True
    return False


def classify_overlap(text: str) -> str:
    """Full-Duplex-Bench v1.5 behavior class for speech that overlapped the
    bot's own voice. Order matters: backchannel first (the cheapest, safest
    keep-talking case), then talk-to-other (hold floor), else interruption
    (cut and respond). Background speech rarely reaches this - the VAD
    loudness gate and the music-mode lyric guard drop it upstream."""
    if is_backchannel(text):
        return "backchannel"
    if is_not_addressed(text):
        return "talk_to_other"
    return "interruption"


def _recent_context(history: list[dict]) -> str:
    """Compact transcript of the last few turns, for the ASR corrector."""
    msgs = [m for m in history if m.get("role") in ("user", "assistant")][-ASR_CORRECT_CTX:]
    lines = []
    for m in msgs:
        role = "user" if m["role"] == "user" else "lava"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _corrector_models() -> list[str]:
    """Ordered models for the proofread: the configured corrector first, then
    the rest of the chain as fallback (deduplicated)."""
    primary = ASR_CORRECT_MODEL or next(
        (m for m in LLM_CHAIN if not m.startswith("openai/gpt-oss")), LLM_CHAIN[0])
    models = [primary] + [m for m in LLM_CHAIN if m != primary]
    return models


async def _correct_asr(text: str, history: list[dict], stats: dict,
                       timeout: float | None = None,
                       avg_logprob: float | None = None) -> str:
    """Fix obvious ASR mis-hearings using recent conversation context.

    Whisper (via Groq) can't take an `initial_prompt`, so proper nouns and
    unusual words get snapped to phonetically-similar common words ("Clara
    Benin" -> "Sarah Benin"). This fast LLM pass repairs those without
    changing what the user meant. It is conservative: if the transcript
    already looks right it is returned unchanged, and any error or timeout
    falls back to the original text, so the pass can only ever help.

    It runs only when Whisper's own confidence (avg_logprob) says the
    transcript is doubtful - proofreading a confident transcript is pure
    latency (see ASR_PROOFREAD_LOGPROB_MAX)."""
    if not ASR_CORRECT or not text:
        return text
    if len(text.split()) < ASR_CORRECT_MIN_WORDS:
        return text
    if (avg_logprob is not None
            and avg_logprob > ASR_PROOFREAD_LOGPROB_MAX):
        logger.debug(f"[asr-correct] skipped (confident: logprob "
                     f"{avg_logprob:.2f})")
        return text
    ctx = _recent_context(history)
    system = (
        "You proofread a voice transcript. The text is what a speech "
        "recognizer heard; it may contain mis-heard words, especially names, "
        "places, brands and spelled-out words. Using the recent conversation "
        "as context, fix only obvious recognition errors. Keep the user's "
        "exact meaning, tone and brevity. Never answer the user, never add or "
        "remove words, never fix grammar they actually used, never translate. "
        "Always output a non-empty corrected version of the transcript - if "
        "it already looks correct, output it exactly as-is. Output ONLY the "
        "corrected text: no quotes, no labels, no explanation."
    )
    user_msg = f"Recent conversation:\n{ctx or '(none)'}\n\nTranscript to fix:\n{text}"
    t0 = time.perf_counter()
    effective = timeout or ASR_CORRECT_TIMEOUT
    try:
        return await asyncio.wait_for(
            _correct_asr_inner(text, system, user_msg, stats, t0),
            timeout=effective,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[asr-correct] timed out after {effective:.1f}s; using raw")
        return text


async def _correct_asr_inner(text: str, system: str, user_msg: str,
                             stats: dict, t0: float) -> str:
    last_err: Exception | None = None
    for model in _corrector_models():
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None
        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=120,
                extra_body=extra,
            )
            fixed = (resp.choices[0].message.content or "").strip()
            stats["asr_correct_ms"] = (time.perf_counter() - t0) * 1000
            if not fixed:
                return text
            # Sanity: a recognition fix should not balloon the text.
            if len(fixed) > max(len(text) * 3, len(text) + 40):
                logger.info(f"[asr-correct] rejected (too long): {fixed!r}")
                return text
            if fixed != text:
                logger.info(f"[asr-correct] {text!r} -> {fixed!r}")
            return fixed
        except Exception as e:
            last_err = e
            if not _is_throttled(e):
                break
            logger.warning(f"[asr-correct] {model} rate-limited; trying next")
            await asyncio.sleep(0.5)
    logger.debug(f"[asr-correct] passed through unchanged ({last_err})")
    return text


# --- deterministic music commands (fast path) --------------------------------
# While music plays, control words ("stop", "next", "louder", "volume 40")
# are routed straight to their tool, bypassing the LLM: sub-second acks, and
# the lyrics-vs-command ambiguity never reaches a model that once PLAYED a
# phantom request. Matched strictly enough that ordinary lyrics never hit it.

_MUSIC_FILLERS = ("hey", "ok", "okay", "please", "could", "can", "would",
                  "will", "you", "just", "kindly", "lava", "the")
_MUSIC_WORDS = re.compile(
    r"\b(music|song|songs|spotify|track|tracks|album|playlist|shuffle|repeat|"
    r"rewind|skip|pause|resume|unpause|play|playing|volume|louder|quieter|"
    r"next|previous|queue|devices?)\b", re.I)
# Phrases that look like commands but belong to other tools/the LLM.
_MUSIC_NOT_OURS = re.compile(
    r"\b(timer|timers|remind(?:er)?|alarm|notes?|lights?|weather|time)\b", re.I)


def parse_music_command(text: str) -> dict | None:
    """Map a spoken music command -> {"tool", "args"} for the tools MCP.

    Returns None for anything that is not a clean music control. Never
    touches timers/notes/etc (see _MUSIC_NOT_OURS) - the LLM keeps those."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    while words and words[0] in _MUSIC_FILLERS:
        words.pop(0)
    while words and words[-1] in _MUSIC_FILLERS:
        words.pop()
    t = " ".join(words)
    if not t:
        return None
    # Restart first: its triggers contain "time"/"song" which NOT_OURS would
    # otherwise block ("play it one more time").
    if re.match(r"^(restart|rewind|start (this )?(song )?(over|again|from the beginning)"
                r"|beginning|play it (again|from the (top|start|beginning))|one more time)\b", t):
        return {"tool": "music_restart", "args": {}}
    if _MUSIC_NOT_OURS.search(t):
        return None
    if len(t) > 80:
        return None
    num = re.search(r"\b\d{1,3}\b", t)

    # now playing / devices --------------------------------------------
    if (re.match(r"^(what(?:'?s| is| s| are)?(?: (?:currently|right now))?"
                 r"(?: playing| song|track)"
                 r"|which (?:song|track)|now playing|tell me the (?:song|track)|"
                 r"this (?:song|track))\b", t)
            or re.match(r"^(what(?:'?s| is| s| are)? (?:this|that) "
                        r"(?:song|track))\b", t)):
        return {"tool": "music_now_playing", "args": {}}
    if re.search(r"\bdevices?\b", t):
        return {"tool": "music_devices", "args": {}}

    # stop / pause -------------------------------------------------------
    if re.match(r"^(stop|pause)\b", t):
        return {"tool": "music_pause", "args": {}}

    # shuffle / repeat ---------------------------------------------------
    if re.match(r"^shuffl(e|es|ing)?\b", t):
        state = ("on" if re.search(r"\bon\b", t)
                 else "off" if re.search(r"\boff\b", t) else "")
        return {"tool": "music_shuffle", "args": {"state": state}}
    if re.match(r"^(repeat|loop)\b", t):
        mode = ("off" if re.search(r"\b(off|stop|cancel|no)\b", t)
                else "track" if re.search(r"\b(song|track|this|current|one)\b", t)
                else "context" if re.search(r"\b(album|playlist|queue|context|list)\b", t)
                else "")
        return {"tool": "music_repeat", "args": {"mode": mode}}

    # previous -----------------------------------------------------------
    if re.match(r"^(previous|prev|last(?: song| one| track)?|go back|back)\b", t):
        return {"tool": "music_skip", "args": {"direction": "previous"}}

    # next ---------------------------------------------------------------
    if re.match(r"^(next|skip)\b", t):
        musicish = re.search(r"\b(song|track|music|spotify|tune|one|this|it)\b", t)
        if musicish or len(t.split()) == 1:   # bare "next"/"skip"; not "next Tuesday"
            return {"tool": "music_skip", "args": {"direction": "next"}}

    # resume / bare play ---------------------------------------------------
    if re.match(r"^(resume|unpause|continue|carry on|go ahead|go on|keep playing)\b", t) \
            or t in ("play", "music", "spotify"):
        return {"tool": "music_resume", "args": {}}

    # volume ---------------------------------------------------------------
    dir_up = re.search(r"\b(louder|loud|higher|boost|blare|blast|crank|up)\b", t) \
        and not re.match(r"^up\s*to\b", t)
    dir_down = re.search(r"\b(quieter|quiet|softer|soft|lower|down|dip)\b", t) \
        and not re.match(r"^down\s*to\b", t)
    if num or dir_up or dir_down or re.search(r"\b(max|maximum|full|mute|silent)\b", t) \
            or "volume" in t.split():
        if re.search(r"\bmute\b|\bsilent\b", t):
            level = "mute"
        elif re.search(r"\b(max|maximum|full)\b", t):
            level = "max"
        elif num is not None:
            n = num.group(0)
            if re.search(r"\b(?:up|down)\s+to\b", t):
                level = n            # "turn it up to 40" -> absolute 40
            else:
                level = (("up " if dir_up and not dir_down else
                          "down " if dir_down and not dir_up else "") + n)
        elif dir_up and not dir_down:
            level = "up"
        elif dir_down and not dir_up:
            level = "down"
        else:
            level = ""        # "what's the volume?" -> report current
        return {"tool": "music_volume", "args": {"level": level}}

    # queue ------------------------------------------------------------------
    m = re.match(r"^queue\s+(.+)$", t)
    if m and len(m.group(1)) >= 2:
        return {"tool": "music_queue", "args": {"query": m.group(1)}}
    m = re.match(r"^add\s+(.+?)\s+to\s+(?:the |my )?(?:play )?queue(?: (?:up )?next)?$", t)
    if m and len(m.group(1)) >= 2:
        return {"tool": "music_queue", "args": {"query": m.group(1)}}

    # play <query> -------------------------------------------------------------
    m = re.match(r"^(?:play|put on)\s+(.+)$", t)
    if m:
        q = m.group(1).strip()
        if q in ("it", "that", "this", "something", "some music", "the music"):
            return {"tool": "music_resume", "args": {}}
        if len(q) >= 2 and 1 <= len(q.split()) <= 10:
            kind = "playlist" if q.startswith("playlist ") \
                else "album" if q.startswith("album ") else "track"
            q = re.sub(r"^(playlist|album|artist) ", "", q).strip()
            q = re.sub(r"^(?:the )?(?:song|track|tune) ", "", q).strip() or q
            return {"tool": "music_play", "args": {"query": q, "kind": kind}}
    return None


def maybe_music_command(text: str) -> dict | None:
    """Gate for the fast path: music mode, or an explicit music word.

    Outside music mode a bare "next" must not hijack general talk ("next
    Tuesday") - only explicit music vocabulary unlocks the fast path."""
    if not FAST_MUSIC_COMMANDS:
        return None
    if _MUSIC_NOT_OURS.search(text):
        return None
    in_music = time.monotonic() < STATE["music_mode_until"]
    if not in_music and not _MUSIC_WORDS.search(text):
        return None
    return parse_music_command(text)


def _voice_of_track(line: str) -> str:
    """TTS-friendly track name: 'X — Y' -> 'X, by Y'."""
    parts = (line or "").split(" — ")
    if len(parts) >= 2:
        return f"{parts[0].strip()}, by {parts[-1].strip()}"
    return (line or "").strip()


def _web_player_note(name: str) -> str:
    return ", in the Spotify web player tab" if "web player" in (name or "").lower() else ""


def _music_error_speech(err: str) -> str:
    """Short spoken form of a tool error (details live in the logs)."""
    low = (err or "").lower()
    if "not linked" in low or "authorization expired" in low:
        return ("Spotify isn't linked to me. Open Lava's Spotify link page in "
                "this browser and approve once.")
    if "reach spotify" in low or "rate-limiting" in low or "network" in low:
        return "I can't reach Spotify right now. Give it a few seconds and try again."
    if "premium" in low:
        return "Spotify Premium is needed to control playback."
    if "no spotify device named" in low:
        return "I can't find a Spotify device with that name."
    if "several spotify devices" in low:
        return "Several Spotify devices are available. Which one should I use?"
    if "open spotify" in low or "no active spotify device" in low:
        return ("No Spotify device is active. Open Spotify somewhere, press "
                "play once, then ask me again.")
    if "nothing is playing" in low:
        return "Nothing is playing right now."
    cleaned = re.sub(r"https?://\S+", "the Lava link page", err or "")
    return "Hmm - " + cleaned[:140]


def _music_reply(tool: str, args: dict, res: dict) -> str:
    """Voice confirmation for the fast path, straight from the tool result."""
    err = res.get("error")
    if err:
        return _music_error_speech(err)
    webn = _web_player_note(res.get("device", ""))
    if tool == "music_pause":
        return "Nothing was playing." if res.get("paused") is False else "Paused."
    if tool == "music_resume":
        return "Resuming."
    if tool == "music_restart":
        return f"Starting {_voice_of_track(res.get('restarted', 'this song'))} over."
    if tool == "music_skip":
        tr = _voice_of_track(res.get("playing", "")) if res.get("playing") else ""
        return (f"Now on {tr}{webn}." if tr else
                "Back a track." if args.get("direction") == "previous" else "Skipped.")
    if tool == "music_play":
        tr = _voice_of_track(res.get("playing", ""))
        if not tr or tr.lower() == "started":
            return "Playback started."
        return f"Playing {tr}{webn}."
    if tool == "music_queue":
        return f"Queued {res.get('queued', 'that')}."
    if tool == "music_volume":
        pct = res.get("volume_pct")
        if pct is None:
            return "There's no volume to read right now."
        if not res.get("action"):
            return f"The volume is at {pct} percent."
        extra = " It was already there." if "already" in res.get("note", "") else ""
        return f"Volume {res['action']} to {pct} percent.{extra}"
    if tool == "music_shuffle":
        return res.get("note") or f"Shuffle is {res.get('shuffle', 'unknown')}."
    if tool == "music_repeat":
        return res.get("note") or f"Repeat mode is {res.get('repeat', 'unknown')}."
    if tool == "music_now_playing":
        if not res.get("playing"):
            return "Nothing is playing right now."
        name = _voice_of_track(res["playing"])
        return f"{name} is paused." if res.get("paused") else f"Now playing {name}{webn}."
    if tool == "music_devices":
        devs = res.get("devices") or []
        if not devs:
            return "No Spotify devices are online."
        return "Devices online: " + ", ".join(d.get("name", "?") for d in devs) + "."
    return "Done."


def _trim_history(history: list[dict]) -> None:
    if len(history) > 1 + HISTORY_TURNS * 2:
        del history[1 : -(HISTORY_TURNS * 2)]


async def run_music_command(ws: web.WebSocketResponse, sess: dict,
                            text: str, stats: dict, cmd: dict):
    """Execute a matched music command directly (no LLM) and ack by voice."""
    tool, args = cmd["tool"], cmd["args"]
    logger.info(f"[web] music fast path {tool}({json.dumps(args)}) <- {text!r}")
    await send(ws, {"type": "tool", "name": tool,
                    "args": json.dumps(args)[:120]})
    history = sess["history"]
    history.append({"role": "user", "content": text})
    result = await exec_tool(tool, args)
    try:
        res = json.loads(result)
    except ValueError:
        res = {}
    reply = _music_reply(tool, args, res)
    history.append({"role": "assistant", "content": reply})
    _trim_history(history)
    # Mirror into STATE so the VAD reacts the same tick as Spotify does.
    now = time.monotonic()
    if tool == "music_pause" and res.get("paused", True):
        STATE["music_playing"] = False
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
    elif tool == "music_resume" and res.get("resumed"):
        STATE["music_playing"] = True
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
        STATE["music_seen_at"] = now
    elif tool == "music_play" and "error" not in res and res.get("playing"):
        STATE["music_playing"] = True
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
        STATE["music_seen_at"] = now
    await send(ws, {"type": "text", "t": reply})
    try:
        await speak_text(ws, sess, reply, stats)
    except Exception as e:
        logger.warning(f"[web] music fast path TTS failed: {e}")
    await send(ws, {"type": "done", "reply": reply, "stats": {
        "asr_ms": round(stats.get("asr_ms", 0)),
        "llm_ms": 0,
        "tts_ms": round(stats.get("tts_ms", 0)),
        "ttfa_ms": round((time.perf_counter() - stats["turn_t0"]) * 1000),
        "total_ms": round(stats.get("asr_ms", 0) + stats.get("tts_ms", 0)),
    }})


async def run_turn(ws: web.WebSocketResponse, sess: dict, user_text: str, stats: dict,
                   music_note: bool = False):
    """ASR -> LLM(+tools) -> TTS for one utterance. Bounded by a hard deadline."""
    deadline = time.monotonic() + TURN_DEADLINE_SECS
    history = sess["history"]
    if music_note:
        # Transient context for THIS turn only (swept right after) so the LLM
        # knows the room noise it just heard may be the music itself.
        history.append({"role": "system", "content": MUSIC_NOTE})
    history.append({"role": "user", "content": user_text})
    reply_parts: list[str] = []
    speaker = Speaker(ws, sess, stats)
    speaker.start()
    llm_t0 = time.perf_counter()
    try:
        buf = ""
        spoken_any = False
        # LLM deltas are batched into one text frame per flush instead of one
        # JSON message per token: a token-per-frame stream made the browser
        # JSON-parse and reflow its chat bubble dozens of times per reply,
        # stealing main-thread time from audio scheduling while the bot talks.
        text_out: list[str] = []
        last_text_flush = time.perf_counter()

        async def flush_text():
            nonlocal last_text_flush
            if text_out:
                await send(ws, {"type": "text", "t": "".join(text_out)})
                text_out.clear()
                last_text_flush = time.perf_counter()

        async for kind, payload in llm_stream(history):
            if time.monotonic() > deadline:
                raise TimeoutError("turn deadline exceeded")
            if kind == "tool_call":
                await flush_text()
                await send(ws, {"type": "tool", "name": payload["name"],
                                "args": json.dumps(payload["arguments"])[:120]})
                continue
            buf += payload
            text_out.append(payload)
            while True:
                chunks, buf = split_sentences(buf, allow_clause=not spoken_any)
                if not chunks:
                    break
                for s in chunks:
                    s = s.strip()
                    if s and not _is_skip(s):
                        reply_parts.append(s)
                        # Show the sentence's text before its audio starts.
                        await flush_text()
                        await speaker.enqueue(s)
                        spoken_any = True
            if time.perf_counter() - last_text_flush >= TEXT_FLUSH_SECS:
                await flush_text()
        await flush_text()
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
        done_stats = {
            "asr_ms": round(stats.get("asr_ms", 0)),
            "llm_ms": round(stats.get("llm_ms", 0)),
            "tts_ms": round(stats.get("tts_ms", 0)),
            "ttfa_ms": round(stats.get("ttfa_ms", 0)),
            "total_ms": round(stats.get("asr_ms", 0) + stats.get("llm_ms", 0) + stats.get("tts_ms", 0)),
        }
        # Full-Duplex-Bench v1.5: surface this turn's overlap metrics (present
        # only when this turn was the response to a barge-in).
        if stats.get("fd_stop_ms") is not None:
            done_stats["fd_stop_ms"] = round(stats["fd_stop_ms"])
        if stats.get("fd_resp_ms") is not None:
            done_stats["fd_resp_ms"] = round(stats["fd_resp_ms"])
        await send(ws, {"type": "done", "reply": reply, "stats": done_stats})
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
        # A turn that never spoke (SKIP, error) leaves the t_resp anchor
        # unconsumed - clear it so it can't bleed into the next turn.
        sess["fd_cut_at"] = 0.0
        # The music note is per-turn context, never a kept memory: without the
        # sweep it would multiply across turns inside the trimmed window.
        for i in range(len(history) - 1, 0, -1):
            m = history[i]
            if m.get("role") == "system" and m.get("content") == MUSIC_NOTE:
                del history[i]


async def process_utterance(ws: web.WebSocketResponse, sess: dict, pcm: bytes):
    """One complete utterance: transcribe, then run the LLM/TTS turn.

    ASR and the overlap classification run BEFORE the turn lock: while a
    possible backchannel is being transcribed, the previous turn keeps
    talking (the lock is taken only once a real turn is committed)."""
    stats = {"turn_t0": time.perf_counter()}
    t0 = time.perf_counter()
    # Spans dug out around music/TV carry seconds of quiet tail; trimming
    # cuts upload size and helps Whisper's quality gate. trim_silence
    # returns None when there is nothing above the audio floor at all.
    trimmed = trim_silence(pcm, SAMPLE_RATE)
    if trimmed is None:
        logger.info("[web] utterance is all silence - skipped ASR")
        sess["barge_onset"] = 0.0
        await send(ws, {"type": "silence"})
        return
    if len(trimmed) < len(pcm):
        pcm = trimmed
    # Tell the client we are working NOW, before the (sometimes slow) ASR
    # round trip: without this the orb sat on "listening" for the whole
    # transcription and looked deaf. The browser ignores it while bot audio
    # is playing, so a backchannel over the bot never flickers the state.
    await send(ws, {"type": "processing"})
    try:
        res = await asr_conn.call("transcribe", {
            "audio_wav_b64": base64.b64encode(pcm16_to_wav(pcm, SAMPLE_RATE)).decode("ascii"),
        })
        if isinstance(res, dict) and res.get("error"):
            # The ASR tool reports backend failures in-band (it catches its
            # own exceptions so one bad upload can't kill the MCP session);
            # route them through the same failure path as a transport drop.
            raise RuntimeError(str(res["error"])[:200])
    except Exception as e:
        logger.error(f"[web] ASR failed: {e}")
        sess["barge_onset"] = 0.0
        if sess.pop("ducked", False):
            # The user spoke over the bot and their words are lost - cut the
            # bot like a confirmed interruption, so the apology isn't layered
            # over the very audio they were trying to stop.
            await handle_barge(ws, sess)
        try:
            await speak_text(ws, sess, ASR_APOLOGY, stats)
        except Exception:
            pass
        await send(ws, {"type": "error", "message": f"ASR failed: {str(e)[:200]}"})
        return
    stats["asr_ms"] = (time.perf_counter() - t0) * 1000
    text = res.get("text", "").strip()
    logger.info(f"[web] ASR ({stats['asr_ms']:.0f} ms): {text!r}")
    music_mode = VADState.music_mode()

    # Overlap classification (deferred barge-in, Full-Duplex-Bench v1.5):
    # this span started over the bot. Decide FIRST, on the raw transcript,
    # which of the four overlap scenarios this is - before the transcript
    # message (the browser cuts its playback on that event) and before the
    # proofread pass:
    #   backchannel    ("mm-hmm")        -> keep talking, no cut
    #   talk_to_other  ("hey John, ...") -> hold the floor, no cut
    #   interruption   (real new intent) -> cut now, respond to the new text
    # A real interruption is already bounded upstream by the
    # BACKCHANNEL_CUT_MAX_SECS watchdog, which cuts if the span stays open.
    # The onset is carried in fd_pending_onset: the ws loop disarms
    # barge_onset at span consumption (so the watchdog can't double-fire
    # while this task is classifying) and hands the timestamp over here.
    onset = sess.pop("fd_pending_onset", 0.0) or sess.get("barge_onset")
    if onset:
        sess["barge_onset"] = 0.0
        old = sess.get("prev_turn")
        sess["prev_turn"] = None
        turn_live = bool(old and not old.done())
        if turn_live or time.monotonic() < sess["bot_audio_until"]:
            cls = classify_overlap(text)
            if cls == "backchannel":
                STATE["backchannels"] += 1
                STATE["fd_behaviors"]["backchannel"] += 1
                logger.info(f"[web] overlap was a backchannel "
                            f"({time.monotonic() - onset:.2f} s) - bot keeps "
                            f"talking: {text!r}")
            elif cls == "talk_to_other":
                STATE["fd_behaviors"]["talk_to_other"] += 1
                logger.info(f"[web] overlap was talk-to-other "
                            f"({time.monotonic() - onset:.2f} s) - holding "
                            f"the floor: {text!r}")
                # Same as a backchannel: the still-speaking task goes back in
                # charge so any later real interruption can cut it normally.
                # No client message at all - the bot keeps talking and the
                # fragment leaves no trace in the UI.
                if turn_live:
                    sess["turn_task"] = old
                if sess.pop("ducked", False):
                    await send(ws, {"type": "unduck"})
                return
            else:
                STATE["barge_ins"] += 1
                STATE["fd_behaviors"]["interruption"] += 1
                logger.info(f"[web] barge-in confirmed "
                            f"({time.monotonic() - onset:.2f} s span): {text!r}")
                sess["fd_cut_onset"] = onset   # t_stop anchor (FDB v1.5)
                await handle_barge(ws, sess, old)
                # Surface this turn's t_stop (recorded by handle_barge) in the
                # response turn's done message.
                stats["fd_stop_ms"] = sess.pop("fd_last_stop_ms", None)

    if not text:
        if sess.pop("ducked", False):
            await send(ws, {"type": "unduck"})
        await send(ws, {"type": "silence"})
        return
    # Show the transcript the moment ASR lands - holding it for the proofread
    # made every turn feel a beat slower even when the fix was a no-op.
    sess["last_voice"] = time.monotonic()
    await send(ws, {"type": "transcript", "text": text})

    # Repair obvious mis-hearings (esp. proper nouns) using recent context.
    # Capped at ASR_CORRECT_GRACE: if the proofread hasn't answered by then,
    # the turn starts on the raw text instead of waiting out the full
    # ASR_CORRECT_TIMEOUT, so a throttled corrector costs ~nothing. Skipped
    # while music plays: those turns are commands (fast path, no repairing
    # needed) or lyric noise (dropped below) - the proofread would only add
    # its whole round-trip to the latency. Also skipped when Whisper reports
    # itself confident (see ASR_PROOFREAD_LOGPROB_MAX): a clean transcript
    # needs no repair and the pass would be pure dead air.
    if not music_mode:
        try:
            fixed = await _correct_asr(text, sess["history"], stats,
                                       timeout=ASR_CORRECT_GRACE,
                                       avg_logprob=res.get("avg_logprob"))
            if fixed and fixed != text:
                logger.info(f"[web] transcript fixed: {text!r} -> {fixed!r}")
                text = fixed
                await send(ws, {"type": "transcript_fix", "text": fixed})
        except Exception as ce:
            logger.warning(f"[web] ASR correction failed, using raw: {ce}")
    try:
        await asyncio.wait_for(sess["turn_lock"].acquire(), timeout=8.0)
    except asyncio.TimeoutError:
        logger.info("[web] dropping utterance (previous turn still running)")
        return
    try:
        # 1) Deterministic music commands: run directly, no LLM.
        cmd = maybe_music_command(text)
        if cmd:
            await run_music_command(ws, sess, text, stats, cmd)
            return
        # 2) Phantom-lyrics guard: while music plays, a sparsely-voiced span
        #    is almost always the song itself. The fast path above already
        #    earned real commands through; anything else here is noise.
        if music_mode and sess.get("last_ratio", 1.0) < VADState.ratio_floor(True):
            logger.info(f"[web] music mode: dropped noisy lyric-like turn "
                        f"({sess.get('last_ratio', 0):.0%} voiced): {text!r}")
            if sess.pop("ducked", False):
                await send(ws, {"type": "unduck"})
            await send(ws, {"type": "silence"})
            return
        await run_turn(ws, sess, text, stats, music_note=music_mode)
    finally:
        sess["turn_lock"].release()


async def handle_barge(ws: web.WebSocketResponse, sess: dict, task=None):
    """Confirmed real overlap: cut the bot now.

    `task` pins the turn task to cancel when the caller holds the
    authoritative reference (deferred classification) - defaulting to the
    session's current turn task keeps the classic watchdog path working."""
    turn = task if task is not None else sess["turn_task"]
    turn_live = bool(turn and not turn.done())
    if turn_live:
        turn.cancel()
        try:
            await turn
        except (asyncio.CancelledError, Exception):
            pass
    if task is None:
        sess["turn_task"] = None
    # Playback is being cut client-side; drop the schedule.
    sess["bot_audio_until"] = 0.0
    # The client restores its playback gain on the barge message itself.
    sess["ducked"] = False
    # Full-Duplex-Bench v1.5: t_stop = barge onset -> the moment the bot
    # actually stops (the "barge" message is what cuts the client's audio).
    onset = sess.get("fd_cut_onset")
    if onset:
        sess["fd_cut_onset"] = 0.0
        stop_ms = (time.monotonic() - onset) * 1000
        sess["fd_last_stop_ms"] = stop_ms
        _fd_record("fd_stop_ms", stop_ms)
    # t_resp anchor: the response turn's first audio (recorded in
    # Speaker._stream_worker) measures how fast the agent gets back to
    # speaking after yielding.
    sess["fd_cut_at"] = time.monotonic()
    await send(ws, {"type": "barge"})
    logger.debug(f"[web] barge message delivered (cid={sess.get('client_id')}, "
                 f"ws closed: {ws.closed})")
    if turn_live:
        logger.info("[web] barge-in: user talking over the bot")
    else:
        logger.debug("[web] barge-in: no live turn, clearing client playback")


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
        "music_playing": STATE["music_playing"],
        "backchannels": STATE["backchannels"],
        "barge_ins": STATE["barge_ins"],
        "fd_metrics": fd_summary(),
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


async def h_spotify_auth(request: web.Request):
    """One-time Spotify link: bounce the user to Spotify's consent page."""
    if not spotify.configured():
        return web.Response(text="Spotify is not configured: add "
                                 "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET "
                                 "to .env and restart.", status=500)
    return web.HTTPFound(spotify.authorize_url(SPOTIFY_REDIRECT_URI))


async def h_spotify_callback(request: web.Request):
    code = request.query.get("code")
    err = request.query.get("error")
    if err or not code:
        return web.Response(text=f"Spotify authorization failed: {err or 'no code'}",
                            status=400)
    ok = await spotify.exchange_code(code, SPOTIFY_REDIRECT_URI)
    if not ok:
        logger.error("[web] Spotify token exchange failed")
        return web.Response(text="Token exchange failed - check the bridge logs.",
                            status=502)
    logger.info("[web] Spotify linked - refresh token stored")
    return web.Response(text="<html><body style='font-family:sans-serif;background:#0a0706;"
                            "color:#f2e9e2;display:flex;justify-content:center;align-items:center;"
                            "height:100vh'><div><h2>Spotify linked &check;</h2>"
                            "<p>You can close this tab and ask Lava to play something.</p>"
                            "</div></body></html>", content_type="text/html")


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
        "last_ratio": 1.0,       # voiced ratio of the latest completed span
        # Deferred barge-in (Full-Duplex-Bench style): when speech starts over
        # the bot we record the onset but do NOT cut yet. The span is
        # transcribed once it closes; a pure backchannel ("mm-hmm") keeps the
        # bot talking, a real interruption cuts. See the ws loop below.
        "barge_onset": 0.0,      # monotonic time the barge hint fired (0 = none)
        "barge_bot_live": False, # was a turn live (or audio playing) at onset
        "last_barge": 0.0,       # monotonic time of the last actual cut
        "prev_turn": None,       # task that was speaking when this span started
        # Full-Duplex-Bench v1.5 anchors: fd_cut_onset = barge onset (t_stop
        # reference, consumed by handle_barge); fd_cut_at = cut instant
        # (t_resp reference, consumed by the response turn's first audio).
        "fd_cut_onset": 0.0,
        "fd_cut_at": 0.0,
        # True while a deferred-overlap onset has ducked the client's
        # playback (the duck is released by unduck/barge).
        "ducked": False,
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
                # Deferred barge-in (Full-Duplex-Bench style): a barge onset
                # over the bot is not yet a cut. Record it and let the span
                # close; process_utterance then classifies the transcript -
                # a pure backchannel ("mm-hmm") keeps the bot talking, real
                # speech cuts. The watchdog below bounds how long a span may
                # stay open before we cut anyway.
                if turn_live or time.monotonic() < sess["bot_audio_until"]:
                    if not sess["barge_onset"]:
                        sess["barge_onset"] = time.monotonic()
                        sess["barge_bot_live"] = turn_live
                        # Duck now, decide later: speech over the bot is
                        # heard the moment it starts, so drop the playback
                        # gain immediately; when the span's transcript
                        # classifies as a backchannel / talk-to-other the
                        # gain comes back ("unduck"), a real interruption
                        # hard-cuts ("barge"). The client also auto-restores
                        # if neither message arrives.
                        sess["ducked"] = True
                        await send(ws, {"type": "duck"})
                        logger.debug(f"[web] barge onset over the bot "
                                     f"(cid={sess.get('client_id')}) - ducked, "
                                     "cut deferred until the span classifies")
                continue
            # Watchdog: a span still open BACKCHANNEL_CUT_MAX_SECS after its
            # barge onset is real speech (words keep coming, the bot is being
            # talked over too long) - cut like a classic barge-in.
            if (sess["barge_onset"] and sess["vad"].utterance is None
                    and time.monotonic() - sess["barge_onset"]
                    > BACKCHANNEL_CUT_MAX_SECS):
                sess["fd_cut_onset"] = sess["barge_onset"]  # t_stop anchor
                sess["barge_onset"] = 0.0
                turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
                if turn_live or time.monotonic() < sess["bot_audio_until"]:
                    STATE["barge_ins"] += 1
                    STATE["fd_behaviors"]["interruption"] += 1
                    logger.info("[web] barge-in: span open past the "
                                f"{BACKCHANNEL_CUT_MAX_SECS:.1f} s watchdog")
                    await handle_barge(ws, sess)
                continue
            if sess["vad"].utterance is None:
                continue
            utterance = sess["vad"].utterance
            sess["vad"].utterance = None
            sess["last_ratio"] = sess["vad"].last_ratio
            sess["last_voice"] = time.monotonic()
            # In a deferred overlap the classification (inside
            # process_utterance) must decide between backchannel and cut -
            # canceling the live turn here would cut before that verdict.
            # Hand the speaking task over via prev_turn: sess["turn_task"]
            # is about to become this span's own process task.
            deferred = bool(sess["barge_onset"])
            if deferred:
                # Disarm the watchdog NOW (not after ASR inside the task):
                # the span is consumed, so the watchdog's "span still open"
                # premise is false. Leaving barge_onset set until the task
                # runs let the watchdog double-fire the cut one frame later
                # and cancel the very task classifying this span.
                sess["fd_pending_onset"] = sess["barge_onset"]
                sess["barge_onset"] = 0.0
            sess["prev_turn"] = sess["turn_task"]
            if sess["turn_task"] and not sess["turn_task"].done() and not deferred:
                sess["turn_task"].cancel()
            sess["turn_task"] = asyncio.create_task(
                process_utterance(ws, sess, utterance))
    except Exception as e:
        import traceback as _tb
        logger.error(f"[web] ws error: {e}\n{_tb.format_exc()}")
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
            if VADState.music_mode():
                # Music listening is presence: never check in (and never time
                # out) while the room is playing something.
                prompted = False
                await asyncio.sleep(2)
                continue
            quiet = time.monotonic() - sess["last_voice"]
            turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
            # Still counting as present while the bot's own audio is queued or
            # playing (a long reply can outlast the turn task by tens of
            # seconds) - otherwise the agent check-ins over its own voice.
            playing = time.monotonic() < sess["bot_audio_until"]
            if turn_live or playing or quiet < IDLE_PROMPT_SECS:
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


async def poll_music_state():
    """Keep STATE's Spotify playback mirror fresh (drives music-aware VAD).

    music_state on the tools MCP is a cached read (refreshed from Spotify at
    most every ~4 s server-side), so polling at 1 Hz is nearly free while a
    session is live. Outside music we still poll for a short window after a
    music tool ran, to catch an external pause on the phone; long-idle
    sessions cost nothing."""
    if not tools_conn.connected or not STATE["live_sessions"]:
        return
    now = time.monotonic()
    playing = STATE["music_playing"]
    recent = now - STATE["music_seen_at"] < MUSIC_POLL_IDLE_MAX
    if not playing and not recent:
        return
    try:
        res = await tools_conn.call("music_state", {})
    except Exception:
        return
    is_playing = bool(res.get("is_playing")) if isinstance(res, dict) else False
    if is_playing:
        STATE["music_seen_at"] = now
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
        for sess in STATE["by_ws"].values():
            sess["last_voice"] = max(sess["last_voice"], now)
    if is_playing != playing:
        logger.info(f"[web] spotify playback mirror: {'playing' if is_playing else 'stopped'}")
        if not is_playing:
            # Silence the idle check-in for the music-mode grace window after
            # the song ends, so 'still there?' never fires over the outro.
            for sess in STATE["by_ws"].values():
                sess["last_voice"] = now + MUSIC_MODE_GRACE
    STATE["music_playing"] = is_playing


async def announce_timers():
    """Speak finished timers (proactive speech). Browser-started timers would
        otherwise finish silently - the web bridge has no other listener."""
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(1.0)
        await poll_music_state()
        # Only drain finished timers when a session could actually announce
        # them: popping with nobody connected used to both burn an MCP round
        # trip every second and silently discard the announcement (the timer
        # was gone by the time the user came back).
        if not tools_conn.connected or not STATE["live_sessions"]:
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
    app.router.add_get("/spotify/auth", h_spotify_auth)
    app.router.add_get("/callback", h_spotify_callback)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    setup_logging("web")
    web.run_app(make_app(), host="127.0.0.1", port=WEB_PORT)
