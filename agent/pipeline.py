"""Lava — full-duplex voice agent pipeline (Pipecat + MCP servers).

Architecture (machine exercise):

    voice in → RNNoise → Silero VAD → SmartTurn end-of-turn (bundled ONNX,
    semantic: is the utterance finished?) → ASR (MCP: Groq Whisper large-v3-turbo)
    → LLM (Groq GPT-OSS-120B + MCP tools) → TTS (MCP: Piper) → voice out

This file is the orchestrator/entry point; the pieces live in sibling modules:

    prompt.py      versioned system prompt
    mcp_client.py  MCPClient per server (connections + sessions)
    tools/*.toml   tool policy: per-server allowlist (`enabled` per tool)
    tool_gateway.py  progressive disclosure: heavy servers behind ONE generic
                   tool (GATEWAY_SERVERS) — compact TOML-generated index
    asr.py / tts.py   the voice-loop bridge services over OUR MCP servers
    vad.py / llm.py / context.py   pipeline building blocks

.env is loaded once, at the top of this file, before any agent function runs
(single source of env loading).

- Barge-in: speech detected → Pipecat cancels the TTS stream and the LLM run.

Run:

    .venv/bin/python ui/server.py
      → the voice web UI (ui/) and one pipeline session per browser tab,
        on one origin — the normal way to talk to Lava from a laptop.

    .venv/bin/python agent/pipeline.py --transport websocket
      → the pipecat dev runner on :7860 speaking Protobuf frames over
        ws://.../ws-client (the RPi local client's wire format).

    .venv/bin/python agent/pipeline.py --check-mcp
      → connect to every MCP server and list its tools, then exit.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from mcp_client import build_mcp_clients, connect_mcp_clients, connect_mcp_clients_report
from tool_gateway import gateway_schemas
import prompt
from asr import McpWhisperSTTService
from tts import McpPiperTTSService
from llm import create_llm, keep_warm
from context import create_context

from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# Audio rates: RPi mic/Silero VAD at 16 kHz in; transport output at 24 kHz
# (Piper's 22 kHz WAV is resampled by the TTS base class as it plays).
AUDIO_IN_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000

# Voice-loop bridge servers: their tools (transcribe/speak) drive ASR/TTS
# directly and must never enter the LLM tool schema (token cost, and the
# model must not transcribe or speak by itself).
BRIDGE_SERVERS = {"asr", "tts"}

# Heavyweight servers hidden behind ONE generic tool each (tool_gateway.py):
# Groq's free tier is 8,000 tokens/minute, and github's full definitions
# alone cost ~2,540. Every capability server sends a compact TOML-generated
# index instead; asr/tts stay bridge-excluded (BRIDGE_SERVERS). Remove a
# server from this set to expose its full tool definitions again.
GATEWAY_SERVERS = {"github", "tavily", "time", "spotify"}

# The opening line, spoken the moment the voice client is ready. One fixed
# sentence: it goes straight to TTS with no LLM round trip, and the UI server
# pre-synthesizes it at startup so the orb tap greets from cache instantly
# (ui/server.py _warm_up). Single source of truth for both paths.
GREETING_TEXT = "Hi I'm Lava, how can I help you today?"


def input_filter():
    """The mic-side noise filter: RNNoise, unless RNNOISE=0.

    RNNoise runs at 48 kHz, so every 16 kHz mic frame is resampled up and
    back down around it — a steady CPU cost on the Pi's four cores, which
    Silero, SmartTurn, the speaker gate and Piper all compete for. Browser
    tabs denoise themselves (getUserMedia noiseSuppression) and skip this
    by connecting with ?dn=0 (ui/server.py /ws handler — no env change
    needed); RNNOISE=0 stays the global kill switch. Keep RNNoise on for
    raw mics (the RPi local client, ui/pi_client.py).
    """
    if os.environ.get("RNNOISE", "1") in ("0", "false", "no"):
        return None
    return RNNoiseFilter()


async def run_bot(
    transport, runner_args: RunnerArguments, session=None,
    mcp_clients=None, base_tools=None, on_worker=None,
) -> None:
    """Run the voice bot for one client session.

    `session` is optional. The pipecat dev runner passes nothing; ui/server.py
    passes a host object that owns the browser wire protocol and receives the
    pipeline's client-facing hooks:

        session.start(worker, llm_context)
            Called once the pipeline is live. The session uses it to gate the
            greeting on its own client-readiness signal (e.g. the page's
            "mic is live" message) and to feed tray-chip turns in as user
            messages. When absent, the greeting follows RTVI's
            on_client_ready instead — the dev-runner client's protocol.

        session.upstream_tap / session.downstream_tap
            FrameProcessor taps the pipeline hosts (after STT, and after
            TTS) so the session can turn pipeline frames into page events;
            see ui/server.py LavaTap for why both positions exist.

    `mcp_clients` + `base_tools` are optional and come as a pair: the UI
    server pre-connects the MCP servers once at startup (_warm_up) and hands
    the shared clients and their tool schema to every session, so an orb tap
    pays none of the connect cost. When absent (dev runner), this function
    connects per session and closes the clients when the session ends.

    `on_worker(worker, runner)` is an optional sync callback invoked right
    after the worker joins the runner — fd-benchmark's adapter uses it to
    hold the runner so it can cancel the pipeline when fdbench closes the
    socket (client transports emit on_disconnected, not the server-side
    on_client_disconnected this function listens for).
    """
    owns_clients = mcp_clients is None or base_tools is None

    if owns_clients:
        # --- MCP: connect every server from mcp_servers.json ---------------
        mcp_clients = build_mcp_clients()
        base_tools = await connect_mcp_clients(
            mcp_clients, exclude_from_schema=BRIDGE_SERVERS | GATEWAY_SERVERS
        )

    # --- Services ----------------------------------------------------------
    # The system prompt rides on the service (system_instruction), not in
    # the context — see llm.py. system_prompt() (not the SYSTEM_PROMPT
    # snapshot) keeps the "Current date" line honest for long-running
    # servers; a wrong internal year is what sent tavily_search after
    # 2024 results for a 2026 question (see prompt.py).
    llm = create_llm(system_instruction=prompt.system_prompt())

    # --- Progressive disclosure: heavy servers behind one generic tool -----
    # A fresh ToolsSchema per session: the shared base schemas plus NEW
    # gateway closures — their confirm-gate state is deliberately per session
    # (tool_gateway.py gateway_schemas). Awaited so a backend that was down
    # at startup can still heal its captured argument schemas here.
    tools = ToolsSchema(
        standard_tools=[
            *base_tools.standard_tools,
            *(await gateway_schemas(llm, mcp_clients, GATEWAY_SERVERS)),
        ]
    )

    stt = McpWhisperSTTService(mcp=mcp_clients["asr"])
    tts = McpPiperTTSService(mcp=mcp_clients["tts"])

    # --- Context: system prompt + merged MCP tools + VAD turns -------------
    llm_context, user_aggregator, assistant_aggregator = create_context(tools)

    # --- Pipeline ----------------------------------------------------------
    pipeline_items = [
        transport.input(),      # mic audio (16 kHz mono)
        stt,                    # ASR MCP: Whisper large-v3-turbo, final on VAD end
    ]
    if session is not None:
        # UI tap (user side): transcripts and dropped utterances. The user
        # aggregator consumes TranscriptionFrames and never pushes them
        # downstream, so this tap must sit before it.
        pipeline_items.append(session.upstream_tap)
    pipeline_items += [
        user_aggregator,        # VAD turns + user context
        llm,                    # Groq GPT-OSS-120B with merged MCP tools
        tts,                    # TTS MCP: Piper, streams WAV chunks
    ]
    if session is not None:
        # UI tap (bot side): reply text, turn end, barge-in, tool events.
        # The websocket output transport serializes only audio — without
        # this tap the page would never see what Lava says.
        pipeline_items.append(session.downstream_tap)
    pipeline_items += [
        transport.output(),     # speaker
        assistant_aggregator,   # assistant context
    ]
    pipeline = Pipeline(pipeline_items)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    if on_worker is not None:
        on_worker(worker, runner)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Greeting: one fixed line, spoken straight through TTS — no LLM
        # round trip on the tap path. Two sources can greet a session (this
        # pipeline path, and the warm-cache stream from ui/server.py's
        # /api/greet), so `session.greeted` is a two-way latch: whichever
        # fires first claims the greeting, the other backs off — the warm
        # pipeline boots in ~200 ms, often BEFORE the page's mic is live
        # and its greet request lands, so without the claim both would run.
        if session is not None and session.greeted:
            # The cache already greeted: only record the line in context.
            llm_context.add_message({"role": "assistant", "content": GREETING_TEXT})
            return
        if session is not None and session.recent_greeting():
            # This client heard the greeting moments ago (previous tap /
            # reconnect): start listening silently instead of saying hello
            # again. ui/server.py GREETING_COOLDOWN_SECS owns the window.
            return
        if session is not None:
            session.greeted = True  # the pipeline is greeting — block the cache
            session.note_greeted()
        await worker.queue_frames([TTSSpeakFrame(text=GREETING_TEXT)])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if session is not None:
            session.start(worker, llm_context)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    # Handshake with Groq now, while the session boots and the greeting
    # plays — not on the user's first turn (llm.py keep_warm).
    warm_task = asyncio.create_task(keep_warm(llm))
    try:
        await runner.run()
    finally:
        warm_task.cancel()
        # Only tear down what this session owns. Shared clients (passed in by
        # ui/server.py's _warm_up) outlive the session — they close at server
        # shutdown; pipeline teardown already drops the tool sessions.
        if owns_clients:
            for name, client in mcp_clients.items():
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"MCP '{name}' close failed: {e}")


async def bot(runner_args: RunnerArguments) -> None:
    """Bot entry point for the Pipecat dev runner."""
    transport_params = {
        # Default browser client of the dev runner (SmallWebRTC).
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Acoustic front-end: suppress music/TV/room noise before VAD and
            # ASR (RNNoise, on-device; see requirements.txt).
            audio_in_filter=input_filter(),
        ),
        # Plain WebSocket audio streaming (custom clients / RPi local client).
        "websocket": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_filter=input_filter(),
            serializer=ProtobufFrameSerializer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


async def check_mcp() -> int:
    """Connect to every configured MCP server and list its tools, then exit.

    Returns 0 when every server connected, 1 when any failed — usable as a
    cron/systemd-timer health check.
    """
    clients = build_mcp_clients()
    if not clients:
        logger.error("No MCP servers configured (check .env)")
        return 1
    try:
        _, failed = await connect_mcp_clients_report(clients)
    finally:
        for name, client in clients.items():
            try:
                await client.close()
            except Exception:
                pass
    if failed:
        logger.error(f"MCP health check FAILED for: {', '.join(failed)}")
        return 1
    logger.info("MCP health check passed: all servers connected")
    return 0


def cli() -> None:
    """Entry point shared by `python agent/pipeline.py` and main.py."""
    if "--check-mcp" in sys.argv:
        sys.argv.remove("--check-mcp")
        raise SystemExit(asyncio.run(check_mcp()))
    from pipecat.runner.run import main

    main()


if __name__ == "__main__":
    cli()
