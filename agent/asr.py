"""ASR — Groq Whisper (asr MCP server; model set there) via VAD segments."""

import asyncio
import base64
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    ControlFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from mcp_client import MCPClient, call_mcp_tool


class UtteranceDroppedFrame(ControlFrame):
    """A VAD speech segment produced no transcript — noise or a non-owner.

    The web client turns this into its "silence" event: the orb drops back
    to listening instead of waiting forever on a transcript that will
    never come (the speaker gate drops non-owner speech before ASR text
    exists, so there is nothing else to hang the UI state on).
    """


NOT_OWNER = "not-owner"  # keep in sync with mcp/asr_server.py _NOT_OWNER


class VoiceNotRecognizedFrame(ControlFrame):
    """The speaker gate REJECTED the segment — another voice, or noise.

    Distinct from UtteranceDroppedFrame (owner spoke, Whisper heard
    nothing): the server turns this one into a throttled "Not your
    voice" toast, so the user knows the orb ignoring them was the gate
    working, not a malfunction. Silence stays quiet.
    """


class McpWhisperSTTService(SegmentedSTTService):
    """Groq Whisper via the asr MCP server (model chosen there).

    SegmentedSTTService buffers audio per VAD speech segment, resamples it to
    16 kHz, wraps it in a WAV container, and calls ``run_stt`` on a background
    task — "final on VAD end" from the design. Frames are emitted with
    ``finalized=True``: they ARE the final transcript for their segment, which
    lets SmartTurn's stop strategy trigger the turn the moment both its
    verdict and the transcript are in (no safety-net timeout wait).

    ttfs_p99_latency is a conservative Groq speech-end→final-transcript budget
    broadcast as the STT safety net; the finalized fast path never waits for
    it — it only bounds the hang case when a transcript is lost.
    """

    def __init__(self, *, mcp: MCPClient, **kwargs):
        super().__init__(
            ttfs_p99_latency=1.0,
            # All fields given: pipecat validates the settings at service
            # start and errors on NOT_GIVEN leftovers. The model string is
            # display only (synthesis happens in the asr MCP server, which
            # pins its own model); language is fixed to English there too.
            settings=STTSettings(model="whisper-large-v3-turbo", language="en"),
            **kwargs,
        )
        self._mcp = mcp

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        try:
            audio_b64 = base64.b64encode(audio).decode("ascii")
            text = (
                await call_mcp_tool(
                    self._mcp, "transcribe", {"audio_b64": audio_b64, "fmt": "wav"},
                    server="asr",
                )
            ).strip()
            if text == NOT_OWNER:  # speaker-gate reject — mcp/asr_server.py
                logger.info("ASR gate: voice not recognized — another speaker or noise")
                yield VoiceNotRecognizedFrame()
            elif text:
                logger.info(f"ASR (mcp whisper): {text!r}")
                yield TranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), finalized=True
                )
            else:
                logger.debug("ASR returned an empty transcript")
                yield UtteranceDroppedFrame()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"ASR MCP error: {e}")
            yield ErrorFrame(error=f"ASR error: {e}")
        finally:
            await self.stop_processing_metrics()
