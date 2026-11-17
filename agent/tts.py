"""TTS — Piper (en_US-hfc_female-medium) via OUR tts MCP server."""

import asyncio
import base64
import json
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from mcp_client import MCPClient, call_mcp_tool
from text_clean import plain_speech


class McpPiperTTSService(TTSService):
    """Piper via the tts MCP server.

    Calls the ``speak`` tool with the aggregated sentence, decodes the Base64
    WAV, and streams it to the transport (resampled by the base class).
    Pipecat cancels this generator on barge-in, which also cancels the in
    flight MCP call.
    """

    # ~100 ms of 22 kHz mono 16-bit audio per frame.
    _WAV_CHUNK_BYTES = 4410 * 2

    def __init__(self, *, mcp: MCPClient, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            # All fields given (None = unsupported): pipecat validates the
            # settings at service start and errors on NOT_GIVEN leftovers.
            # Piper has no runtime model/voice switching — the voice file
            # is loaded by the MCP server; language is not an input.
            settings=TTSSettings(
                model="piper",
                voice="en_US-hfc_female-medium",
                language=None,
            ),
            **kwargs,
        )
        self._mcp = mcp

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Speak plain sentences: the model's markdown (**bold**, bullets,
        # symbols) is unwrapped before Piper sees it (text_clean.py).
        text = plain_speech(text)
        await self.start_tts_usage_metrics(text)
        try:
            raw = await call_mcp_tool(self._mcp, "speak", {"text": text}, server="tts")
            wav = base64.b64decode(json.loads(raw)["audio_b64"])

            async def wav_chunks() -> AsyncGenerator[bytes, None]:
                for i in range(0, len(wav), self._WAV_CHUNK_BYTES):
                    yield wav[i : i + self._WAV_CHUNK_BYTES]

            async for frame in self._stream_audio_frames_from_iterator(
                wav_chunks(), strip_wav_header=True, context_id=context_id
            ):
                yield frame
        except asyncio.CancelledError:
            logger.debug("TTS cancelled (barge-in)")
            raise
        except Exception as e:
            logger.error(f"TTS MCP error: {e}")
            yield ErrorFrame(error=f"TTS error: {e}")
