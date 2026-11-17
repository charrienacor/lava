"""Lava web server — the voice UI and the pipeline on one origin.

Serves ui/ (the page and its static assets), the tool-policy API the MCP
tray renders from, and the /ws voice websocket the page speaks:

    browser → server   binary: raw 16 kHz mono 16-bit LE PCM (mic frames)
                       text:   {"type": "ready"}   mic is live — VAD/turns on
                               {"type": "ask", …}  a tray chip as a user turn
    server → browser   binary: [uint16 LE sample rate][16-bit LE mono PCM]
                       text:   small JSON events — transcript, text, done,
                               processing, duck, barge, silence, busy, mcp,
                               tool_start, tool_done, toast, error, bye

    browser → REST     POST /api/greet?cid=…  mic is live — fire the instant
                                             greeting from the warm cache

Startup is the slow part, by design: _warm_up (lifespan) connects the MCP
servers once — including the asr/tts subprocesses, where Piper's voice model
loads at import — and pre-synthesizes the greeting. The "ready" link is logged
only after that, so ./run.sh hands you the URL with everything already loaded. Sessions share the warm clients, and
the greeting is streamed from cache the moment the mic is live: a tap on the
orb greets instantly instead of paying the pipeline boot (the pipeline itself
still boots in the background and is ready before the user's first turn ends).

The pipeline itself is agent/pipeline.py's run_bot. The wire is split in
two (see LavaSerializer and LavaTap): the serializer is the codec for
audio and the page's commands, while the taps turn pipeline frames into
the page's events — pipecat's websocket output transport only serializes
audio, so reply text and turn events would never reach the wire without
them. One pipeline session runs per browser tab, keyed by the ?cid= the
page sends: a second tab with the same id is told "busy" instead of
double-talking the same conversation.

Run:  .venv/bin/python ui/server.py [--port 8000]
"""

import asyncio
import base64
import io
import json
import os
import re
import struct
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from loguru import logger
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_UI = Path(__file__).resolve().parent
_ROOT = _UI.parent
sys.path.insert(0, str(_ROOT / "agent"))

import pipeline  # noqa: E402  (agent/pipeline.py — the voice agent)
import policy_view  # noqa: E402  (agent/policy_view.py — tray payload)
import context as ctx  # noqa: E402  (agent/context.py — LAST_SUMMARY stamp)
from llm import fallback_model, on_fallback_model  # noqa: E402  (LLM resilience)
from llm import on_rate_limit  # noqa: E402  (Groq 429 visibility)
from text_clean import plain_speech  # noqa: E402  (markdown -> plain sentences)
from asr import UtteranceDroppedFrame, VoiceNotRecognizedFrame  # noqa: E402
from mcp_client import (  # noqa: E402
    HEALTH as MCP_HEALTH,
    build_mcp_clients,
    call_mcp_tool,
    connect_mcp_clients_report,
)

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EagerTranscriptionFrame,
    ErrorFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    TTSTextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import (  # noqa: E402
    FrameDirection,
    FrameProcessor,
)
from pipecat.runner.types import WebSocketRunnerArguments  # noqa: E402
from pipecat.serializers.base_serializer import FrameSerializer  # noqa: E402
from pipecat.transports.websocket.fastapi import (  # noqa: E402
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# ---- logging: clean console, full detail in logs/web.log ----------------
# The console keeps INFO+ in a short "time | level | message" format; the
# DEBUG firehose (MCP tool listings, frames, metrics) goes to the file
# only. This must run after the pipecat
# imports — reconfiguring the global loguru logger covers their output too.
_LOGS_DIR = _ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add(
    _LOGS_DIR / "web.log",
    level="DEBUG",
    rotation="5 MB",
    retention=5,
    enqueue=True,
)

# The page's mic rate (app.js SR_IN): raw 16-bit LE mono, 512-sample frames.
MIC_SAMPLE_RATE = 16000

# Hard session cap (pipecat's session_timeout is a MAX DURATION, not an idle
# timer — its monitor just sleeps and fires). At 300 s it cut conversations
# off mid-sentence and the page's reconnect started a fresh session. 1 h is
# a sane ceiling: on expiry the page stops the mic quietly and the next tap
# starts over (the LLM context is per session and does not carry over).
SESSION_TIMEOUT_SECS = 3600

# Greeting cooldown: every tap, page reload and session-timeout reconnect
# starts a NEW session, and greeting each one turned a few quick taps into
# "Hi, I'm Lava" x4. So a client id hears the greeting at most once per this
# window; after it lapses (the user came back later), it plays again.
GREETING_COOLDOWN_SECS = 600
_LAST_GREET: dict[str, float] = {}  # cid -> monotonic time of last greeting

# The opening line (pipeline.GREETING_TEXT) and the warm state every session
# shares. Filled once by _warm_up at startup; when it fails, sessions fall
# back to run_bot's per-session connects and the pipeline's spoken greeting.
GREETING_TEXT = pipeline.GREETING_TEXT
SHARED_MCP: dict | None = None                   # connected MCP clients (all)
SHARED_TOOLS = None                              # ToolsSchema, no gateway tools
GREETING_AUDIO: tuple[bytes, int] | None = None  # (16-bit mono PCM, rate)

# Meridiem display form: "12:18 am" / "5 pm" / "9 a.m." -> "A.M." / "P.M.",
# matching the style rule in the system prompt. The prompt does the writing;
# this backstop guarantees the bubble even when the model ignores it. The
# digit lookbehind anchors to times, so "I am here" is never touched.
_MERIDIEM_RE = re.compile(r"(?<=\d)\s*([ap])\.?m\.?", re.IGNORECASE)


def _meridiem_caps(text: str) -> str:
    return _MERIDIEM_RE.sub(lambda m: f" {m.group(1).upper()}.M.", text)


async def _warm_up() -> None:
    """Load everything ./run.sh promises before the UI link is printed.

    Connects the MCP servers once (spawning the asr/tts subprocesses — the
    Piper voice model loads at import there), and pre-synthesizes the greeting
    so a tap on the orb greets from cache
    instantly. Every later session reuses the clients and tool schema; a
    failure here only costs speed (per-session connects, TTS greeting), not
    the server itself.
    """
    global SHARED_MCP, SHARED_TOOLS, GREETING_AUDIO
    try:
        clients = build_mcp_clients()
        tools, failed = await connect_mcp_clients_report(
            clients,
            exclude_from_schema=pipeline.BRIDGE_SERVERS | pipeline.GATEWAY_SERVERS,
        )
        if failed:
            logger.warning(f"warmup: some MCP servers failed: {', '.join(failed)}")
        # Speak the greeting now: warms the tts path end-to-end and caches
        # the WAV for the instant-greeting path (/api/greet).
        if "tts" in clients:
            raw = await call_mcp_tool(
                clients["tts"], "speak", {"text": GREETING_TEXT}, server="tts"
            )
            payload = json.loads(raw)
            with wave.open(io.BytesIO(base64.b64decode(payload["audio_b64"]))) as w:
                GREETING_AUDIO = (w.readframes(w.getnframes()), w.getframerate())
        # Warm the remote search edge: the first tavily call on a fresh MCP
        # session measured ~4.5 s vs ~1 s for the identical call afterwards —
        # a cold start that would otherwise land on the user's first question.
        # One 1-result probe moves it to startup; no keepalive loop (each
        # probe costs an API credit).
        if "tavily" in clients:
            try:
                await call_mcp_tool(
                    clients["tavily"], "tavily_search",
                    {"query": "today", "max_results": 1, "search_depth": "basic",
                     "include_raw_content": False, "include_images": False},
                    server="tavily",
                )
            except Exception as e:
                logger.warning(f"warmup: tavily probe failed ({e}) — first search may be slow")
        SHARED_MCP, SHARED_TOOLS = clients, tools
    except Exception as e:
        logger.error(f"warmup failed — sessions will connect per tap: {e}")
        return
    logger.info(f"Lava web UI → http://localhost:{_port()} (warm: MCP, TTS, greeting)")


class Session:
    """One connected browser tab, coordinated with the pipeline.

    The pipeline (agent/pipeline.py run_bot) calls back into this object:

        start(worker, llm_context)  once the pipeline is live — spawns the
                                    greeting gate and the tray-turn pump
        upstream_tap / downstream_tap   the frame taps that emit this
                                    page's events (see LavaTap)

    The greeting must not preempt the browser's mic-permission click: the
    page asks for the instant greeting (POST /api/greet) only after the mic
    is live, and that is what releases the cached "Hi, I'm Lava" audio. The
    pipeline's own greeting path then only records the line in the context
    (pipeline.py on_client_ready checks `greeted`).
    """

    def __init__(self, cid: str, websocket: WebSocket):
        self.cid = cid
        self.ws = websocket
        self.ready = asyncio.Event()
        self.greeted = False  # instant greeting already sent (warm cache)
        self.asks: asyncio.Queue[str] = asyncio.Queue()
        self.upstream_tap = LavaTap(self, "user")
        self.downstream_tap = LavaTap(self, "bot")
        self._tasks: list[asyncio.Task] = []
        # Per-turn latency budget (see turn_first_audio): the taps stamp
        # milestones as frames flow through, and the first reply audio
        # publishes the turn's TTFA with its stage split.
        self._turn_t0: float | None = None        # first turn signal (VAD stop or transcript)
        self._turn_asr_ms: float | None = None    # transcript in hand since then

    async def send(self, payload: dict) -> None:
        """Best-effort JSON event; a closing page has nothing left to hear."""
        try:
            await self.ws.send_json(payload)
        except Exception:
            pass

    def start(self, worker, llm_context) -> None:
        """Pipeline is live: launch the greeting gate and the ask pump."""
        self._tasks = [
            asyncio.create_task(self._ready_task(worker)),
            asyncio.create_task(self._ask_task(worker, llm_context)),
        ]

    def start_greeting(self) -> None:
        """Fire the instant greeting (warm cache) — at most once per client
        per cooldown window (see GREETING_COOLDOWN_SECS).

        Called when the page's mic goes live (/api/greet). The audio rides
        straight out of the warm cache — no pipeline, no LLM — so the orb
        greets the instant the mic allows it. The pipeline is told via the
        `greeted` latch; its on_client_ready only adds the line to context.
        """
        if self.greeted or GREETING_AUDIO is None or self.recent_greeting():
            self.greeted = True  # claimed — nothing else may greet this session
            return
        self.greeted = True
        self.note_greeted()
        asyncio.create_task(self._send_greeting())

    def recent_greeting(self) -> bool:
        """This client id heard the greeting within the cooldown window."""
        last = _LAST_GREET.get(self.cid)
        return last is not None and (time.monotonic() - last) < GREETING_COOLDOWN_SECS

    def note_greeted(self) -> None:
        """Timestamp the greeting for the cooldown window (either path)."""
        _LAST_GREET[self.cid] = time.monotonic()

    async def _send_greeting(self) -> None:
        pcm, rate = GREETING_AUDIO
        try:
            # Text first (bubble fills), then audio (orb flips to speaking),
            # then done — the same order the pipeline emits for a turn.
            await self.send({"type": "text", "t": GREETING_TEXT})
            await self.ws.send_bytes(struct.pack("<H", rate) + pcm)
            await self.send({"type": "done", "reply": GREETING_TEXT})
        except Exception:
            pass  # the page left before the greeting could land

    # ---- per-turn latency budget (time-to-first-audio) --------------------
    # The article pattern: TTFA — user stops speaking → first reply audio —
    # is the number the user experiences, and it must be split per stage to
    # know where the time hides. The taps stamp the milestones; the turn
    # ends at the first reply audio with an INFO line.

    def turn_vad_stop(self) -> None:
        """User's speech just ended (VAD): the TTFA clock starts now.

        The transcript frame occasionally races in FIRST (ASR runs on a
        background task), so the earlier of the two signals anchors the
        clock and the existing asr split is kept, not cleared.
        """
        if self._turn_t0 is None:
            self._turn_t0 = time.monotonic()

    def turn_transcript(self) -> None:
        """Final transcript in hand: how long endpointing + ASR took."""
        if self._turn_t0 is None:
            self._turn_t0 = time.monotonic()
        self._turn_asr_ms = (time.monotonic() - self._turn_t0) * 1000

    def turn_discard(self) -> None:
        """Turn ended without a reply (noise/drop/error): rewind the clock."""
        self._turn_t0 = None
        self._turn_asr_ms = None

    def turn_first_audio(self) -> None:
        """First reply audio is out: publish this turn's TTFA budget."""
        if self._turn_t0 is None:
            return  # no pending user turn (greeting, barge spill, ...)
        ttfa_ms = (time.monotonic() - self._turn_t0) * 1000
        asr_ms = self._turn_asr_ms
        self.turn_discard()
        if asr_ms is not None:
            rest_ms = ttfa_ms - asr_ms
            logger.info(
                f"TTFA {ttfa_ms:.0f} ms — endpointing+ASR {asr_ms:.0f} ms, "
                f"LLM+TTS {rest_ms:.0f} ms"
            )
        else:
            logger.info(f"TTFA {ttfa_ms:.0f} ms — (no transcript this turn)")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks = []

    async def _ready_task(self, worker) -> None:
        await self.ready.wait()
        # Fires the pipeline's on_client_ready handler — the same greeting
        # path the dev-runner's RTVI client drives.
        await worker.rtvi.set_client_ready()

    async def _ask_task(self, worker, llm_context) -> None:
        """Tray chips: typed user turns that flow through the voice loop."""
        while True:
            try:
                text = await self.asks.get()
                llm_context.add_message({"role": "user", "content": text})
                await worker.queue_frames([LLMRunFrame()])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"ask turn failed: {e}")
                await self.send({"type": "error"})


def _action_of(arguments) -> str:
    """Display label for a gateway call: github(action='merge_pull_request')."""
    if isinstance(arguments, dict):
        action = arguments.get("action")
        if action:
            return str(action)
    return ""


# ---- server-initiated toasts ----------------------------------------------
# The page renders any {"type": "toast"} event in the same tray the tool
# notifications use. Producers: the Groq rate-limit watcher, the
# speaker-gate reject (a TV in the background rejects continuously), and
# the context-summarize watcher — hence the throttle.

_TOAST_THROTTLE_SECS = 20.0
_last_toast_at: dict[str, float] = {}


def broadcast_toast(key: str, title: str, sub: str, cat: str = "system") -> None:
    """Send a toast to every connected session, throttled per key."""
    now = time.monotonic()
    if now - _last_toast_at.get(key, 0.0) < _TOAST_THROTTLE_SECS:
        return
    _last_toast_at[key] = now
    payload = {"type": "toast", "key": key, "cat": cat, "title": title, "sub": sub}
    for session in list(SESSIONS.values()):
        asyncio.create_task(session.send(payload))


def _on_groq_rate_limit():
    """Groq 429 hit the wire: say so while the silent SDK retry runs."""
    logger.warning("Groq rate limit (429) — SDK is backing off and retrying")
    broadcast_toast(
        "ratelimit",
        "Rate limited",
        "Groq is throttling — replies may lag a few seconds",
    )


on_rate_limit(_on_groq_rate_limit)


def _on_llm_fallback():
    """The primary model failed after retries: the small model took over."""
    logger.warning("LLM: primary model unavailable — answering with the fallback")
    broadcast_toast(
        "fallback-model",
        "Backup model",
        f"Primary rate-limited — answering with {fallback_model()}",
    )


on_fallback_model(_on_llm_fallback)


class LavaTap(FrameProcessor):
    """Pipeline tap that turns frames into the page's JSON events.

    pipecat's websocket output transport serializes only audio and transport
    messages — every other frame that reaches it is dropped silently
    (write_transport_frame is a no-op there). So the page's chat and orb
    events ride out through these taps instead, which the pipeline hosts on
    the session's behalf. Frames pass through untouched; this only observes.

    Two instances, one per side:

      "user" — after the STT service, before the user aggregator: user-side
               frames (final transcripts, dropped utterances). The user
               aggregator consumes TranscriptionFrames and never pushes them
               downstream, so this tap must sit before it.

      "bot"  — after TTS, before the output transport: reply text, turn end,
               barge-in, tool calls, ASR/TTS errors. Reply text arrives as
               AggregatedTextFrame slots sequenced in order with the speech,
               so bubbles fill in while Lava is talking.
    """

    def __init__(self, session: "Session", side: str):
        super().__init__(name=f"LavaTap#{side}")
        self._session = session
        self._side = side
        self._reply: list[str] = []
        self._bot_speaking = False
        self._run_tools = False  # the current LLM run called a tool

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if self._side == "user":
                await self._observe_user(frame)
            else:
                await self._observe_bot(frame)
        except Exception as e:
            logger.warning(f"tap event failed: {e}")
        await self.push_frame(frame, direction)

    # ---- user side ------------------------------------------------------

    async def _observe_user(self, frame) -> None:
        if isinstance(frame, TranscriptionFrame):
            if isinstance(frame, (InterimTranscriptionFrame, EagerTranscriptionFrame)):
                return
            self._session.turn_transcript()
            await self._session.send({"type": "transcript", "text": frame.text})
        elif isinstance(frame, UtteranceDroppedFrame):
            # Speech segment with no transcript (noise or a non-owner):
            # drop the orb back to listening instead of hanging on thinking.
            self._session.turn_discard()  # no reply will come — rewind TTFA
            await self._session.send({"type": "silence"})
        elif isinstance(frame, VoiceNotRecognizedFrame):
            # The speaker gate REJECTED the segment (another voice or
            # unembeddable noise) — same turn discard as silence, plus a
            # throttled toast so the user knows WHY they were ignored.
            self._session.turn_discard()  # no reply will come — rewind TTFA
            broadcast_toast(
                "voice-rejected",
                "Not your voice",
                "Speech ignored — another speaker or background noise",
            )
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # The TTFA clock starts when the VAD says the user stopped —
            # everything after this (endpointing verdict, ASR, LLM, TTS)
            # is delay the user experiences as silence.
            self._session.turn_vad_stop()

    # ---- bot side -------------------------------------------------------

    async def _observe_bot(self, frame) -> None:
        if isinstance(frame, TTSTextFrame):
            # The spoken sentence, sequenced with the speech that carries it
            # (the AggregatedTextFrame slot riding alongside it is the
            # context's copy — emitting both would double every bubble).
            # Display form: plain sentences (text_clean) with A.M./P.M.
            self._reply.append(frame.text)
            clean = _meridiem_caps(plain_speech(frame.text))
            await self._session.send({"type": "text", "t": clean})
        elif isinstance(frame, LLMFullResponseEndFrame):
            reply = "".join(self._reply)
            self._reply.clear()
            # `tools` marks runs that called a tool (pipecat ends EVERY run
            # with this frame — tool-call runs included). The page must not
            # erase the user's transcript for a turn where a command actually
            # ran; only a truly silent, tool-less run is a skipped fragment.
            tools = self._run_tools
            self._run_tools = False
            await self._session.send({"type": "done", "reply": reply, "tools": tools})
        elif isinstance(frame, TTSStartedFrame):
            self._bot_speaking = True
            self._session.turn_first_audio()  # publish this turn's TTFA
        elif isinstance(frame, TTSStoppedFrame):
            self._bot_speaking = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            # Someone started talking over the bot: duck, don't cut — the
            # overlap may be a backchannel ("mm-hmm"). A real interruption
            # arrives next as InterruptionFrame and hard-cuts via "barge";
            # the page's 2.5 s watchdog un-ducks if the overlap passes.
            if self._bot_speaking:
                await self._session.send({"type": "duck"})
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Utterance handed to ASR — flip the orb to thinking now, not
            # when the transcript lands a beat later.
            await self._session.send({"type": "processing"})
        elif isinstance(frame, InterruptionFrame):
            if self._bot_speaking:
                self._bot_speaking = False
                await self._session.send({"type": "barge"})
        elif isinstance(frame, FunctionCallsStartedFrame):
            calls = list(frame.function_calls or [])
            if calls:
                self._run_tools = True
                call = calls[0]
                await self._session.send({
                    "type": "tool_start",
                    "key": call.tool_call_id,
                    "cat": "tool",
                    "title": call.function_name,
                    "sub": _action_of(call.arguments),
                })
        elif isinstance(frame, FunctionCallResultFrame):
            sub, error = "done", None
            payload = frame.result
            try:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict):
                    if payload.get("error"):
                        error = str(payload["error"])
                        sub = "failed"
                    elif payload.get("ok") is False:
                        error = "failed"
                        sub = "failed"
            except (ValueError, TypeError):
                pass
            if frame.error:
                error, sub = frame.error, "failed"
            await self._session.send({
                "type": "tool_done",
                "key": frame.tool_call_id,
                "cat": "tool",
                "title": frame.function_name,
                "sub": sub,
                **({"error": error} if error else {}),
            })
        elif isinstance(frame, ErrorFrame):
            self._session.turn_discard()  # errored turn — rewind TTFA
            await self._session.send({"type": "error"})


class LavaSerializer(FrameSerializer):
    """Audio codec + the page's two commands (see module docstring).

    Audio rides as binary frames with a uint16 LE sample-rate header; mic
    PCM arrives as raw binary. Everything else the page needs flows through
    the taps (LavaTap) — frames with no wire meaning serialize to None and
    the transport drops them.
    """

    def __init__(self, session: "Session"):
        super().__init__()
        self._session = session

    async def deserialize(self, data: str | bytes):
        if isinstance(data, bytes):
            # Mic: raw 16 kHz mono s16le — page batches 512-sample frames.
            return InputAudioRawFrame(
                audio=data, sample_rate=MIC_SAMPLE_RATE, num_channels=1
            )
        try:
            msg = json.loads(data)
        except (ValueError, TypeError):
            return None
        kind = msg.get("type")
        if kind == "ready":  # mic live — the greeting may start
            self._session.ready.set()
        elif kind == "ask":  # tray chip → a spoken-style user turn
            text = str(msg.get("text", "")).strip()
            if text:
                self._session.asks.put_nowait(text)
        return None

    async def serialize(self, frame):
        if isinstance(frame, OutputAudioRawFrame):
            # [uint16 LE sample rate][16-bit LE mono PCM] — the page reads
            # the rate per chunk, so the pipeline's 24 kHz output just works.
            return struct.pack("<H", frame.sample_rate) + frame.audio
        return None


# One live voice session per client id (the page's stable per-tab cid).
SESSIONS: dict[str, Session] = {}


def _policy_paths() -> list[Path]:
    paths = sorted((_ROOT / "agent" / "tools").glob("*.toml"))
    paths.append(_ROOT / "mcp" / "mcp_servers.json")
    return paths


async def _watch_policy() -> None:
    """Push a fresh tray snapshot when a TOML or servers.json changes.

    The page renders the tray from these pushes (its /api/allowlist fetch
    is only the initial fallback), so flipping `enabled` in a TOML shows
    up live — no restart, no reload.
    """
    last = [(p, p.stat().st_mtime_ns) for p in _policy_paths()]
    while True:
        await asyncio.sleep(2)
        try:
            now = [(p, p.stat().st_mtime_ns) for p in _policy_paths()]
            if now == last:
                continue
            last = now
            payload = {"type": "mcp", "servers": _tray_snapshot()["servers"]}
            for session in list(SESSIONS.values()):
                await session.send(payload)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"policy watch: {e}")


async def _watch_summary() -> None:
    """Toast when the context summarizer compresses the conversation.

    pipecat's summarizer stamps ctx.LAST_SUMMARY via its
    on_summary_applied event (context.py, same process and event loop);
    this watcher polls the stamp every 2 s — the _watch_policy cadence —
    and toasts once per new stamp. The 20 s toast throttle absorbs the
    summarize-after-every-turn case a long conversation can produce.
    """
    seen = None
    while True:
        await asyncio.sleep(2)
        stamp = ctx.LAST_SUMMARY
        if stamp is None or stamp == seen:
            continue
        seen = stamp
        _, before, after = stamp
        broadcast_toast(
            "summarize",
            "Context compressed",
            f"{before} → {after} messages — older chat folded into a summary",
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Everything heavy loads here, before uvicorn serves anything: the link
    # that ./run.sh prints only appears once the pipeline is warm.
    await _warm_up()
    watcher = asyncio.create_task(_watch_policy())
    summary_watcher = asyncio.create_task(_watch_summary())
    yield
    watcher.cancel()
    summary_watcher.cancel()
    if SHARED_MCP:
        for name, client in SHARED_MCP.items():
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"MCP '{name}' close failed: {e}")


app = FastAPI(title="Lava", docs_url=None, redoc_url=None, lifespan=_lifespan)


@app.middleware("http")
async def _static_revalidate(request, call_next):
    # Static assets revalidate on every load (ETag → cheap 304 when
    # unchanged), so UI edits show up on a normal reload — no more stale
    # app.css/app.js after editing files.
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index():
    return FileResponse(_UI / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(_UI / "favicon.ico")


app.mount("/static", StaticFiles(directory=_UI / "static"), name="static")


def _tray_snapshot() -> dict:
    """policy_view's TOML truth, augmented with live MCP reachability.

    MCP_HEALTH is updated at connect time and on every tool call (retries
    make the last attempt's outcome the truth); missing = never contacted.
    """
    snap = policy_view.snapshot()
    for s in snap["servers"]:
        s["connected"] = MCP_HEALTH.get(s["name"])
    return snap


@app.get("/api/allowlist")
async def allowlist():
    # Re-reads the TOMLs on every call (policy_view's contract).
    return _tray_snapshot()


@app.post("/api/greet")
async def greet(cid: str = ""):
    """Mic is live: fire the instant greeting from the warm cache (once)."""
    session = SESSIONS.get(cid)
    if session:
        session.start_greeting()
    return {"ok": True}


@app.websocket("/ws")
async def voice(websocket: WebSocket):
    await websocket.accept()
    cid = websocket.query_params.get("cid", "anon")
    if cid in SESSIONS:
        # Same tab id on a second device/socket: one conversation, one owner.
        await websocket.send_json({"type": "busy"})
        await websocket.close()
        return

    session = Session(cid, websocket)
    SESSIONS[cid] = session
    logger.info(f"Session {cid} connected")
    try:
        # The tray renders from this snapshot (with live health dots); later
        # edits arrive via watch.
        await session.send({"type": "mcp", "servers": _tray_snapshot()["servers"]})

        transport = FastAPIWebsocketTransport(
            websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                # Acoustic front-end: music/TV/room noise is suppressed
                # before the VAD and ASR ever see it.
                audio_in_filter=RNNoiseFilter(),
                serializer=LavaSerializer(session),
                session_timeout=SESSION_TIMEOUT_SECS,
            ),
        )

        @transport.event_handler("on_session_timeout")
        async def on_session_timeout(_transport, _client):
            # The page stops the mic quietly and resumes on the next tap.
            await session.send({"type": "bye"})

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await pipeline.run_bot(
            transport, runner_args, session=session,
            mcp_clients=SHARED_MCP, base_tools=SHARED_TOOLS,
        )
    except Exception as e:
        logger.error(f"Session {cid} crashed: {e}")
        await session.send({"type": "error"})
    finally:
        await session.stop()
        SESSIONS.pop(cid, None)
        logger.info(f"Session {cid} disconnected")


def _port() -> int:
    argv = sys.argv[1:]
    if "--port" in argv:
        return int(argv[argv.index("--port") + 1])
    return int(os.environ.get("LAVA_UI_PORT", "8000"))


if __name__ == "__main__":
    port = _port()
    # The URL is logged by _warm_up once everything is loaded — ./run.sh
    # hands over a link that is already fully warm.
    logger.info("Warming up Lava (MCP servers, TTS voice)…")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
