"""TTS — Piper (en_US-hfc_female-medium) via OUR tts MCP server."""

import asyncio
import base64
import json
import os
import re
import time
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from mcp_client import MCPClient, call_mcp_tool
from text_clean import plain_speech

# --- text helpers -----------------------------------------------------------

_ODD_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2002\u2003\u200b"), " ")
_SPLIT = re.compile(r"(?<=[,;:.!?])\s+")


def normalize_for_piper(text: str) -> str:
    """Replace characters Piper's phonemizer handles badly."""
    text = text.translate(_ODD_SPACES)
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
    )
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(
        r"(?<=\d)\s?([AaPp])\.?[Mm]\b",
        lambda m: f" {m.group(1).upper()} M",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def split_for_speed(text: str, min_chars: int = 25) -> list[str]:
    """Break text at punctuation into chunks of at least min_chars."""
    chunks, cur = [], ""
    for part in _SPLIT.split(text):
        cur = f"{cur} {part}".strip()
        if len(cur) >= min_chars:
            chunks.append(cur)
            cur = ""
    if cur:
        if chunks:
            chunks[-1] += " " + cur
        else:
            chunks.append(cur)
    return chunks


# --- service ----------------------------------------------------------------

class McpPiperTTSService(TTSService):
    """Piper via the tts MCP server.

    Splits each sentence at punctuation and plays the synthesized chunks in
    order, synthesizing the next chunk while the previous one plays.

    No silent lead-in: run_tts runs once PER SENTENCE, so a lead-in delayed
    first speech by its full length (the client plays it before anything
    else) and put the same gap between every pair of sentences. The audio
    context does not need it either — TTSStoppedFrame only fires after
    stop_frame_timeout_s (3 s) of idle, far longer than a Piper chunk.
    """

    # 4410 samples * 2 bytes = 8820 bytes = ~200 ms of 22.05 kHz mono 16-bit.
    _WAV_CHUNK_BYTES = 4410 * 2
    _WAV_HEADER_BYTES = 44

    def __init__(self, *, mcp: MCPClient, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(
                model="piper",
                # Display only: the tts MCP server owns synthesis and reads
                # the same TTS_VOICE from .env (the voice can differ per
                # host — the Pi runs a lighter tier).
                voice=os.environ.get("TTS_VOICE") or "en_US-hfc_female-medium",
                language=None,
            ),
            **kwargs,
        )
        self._mcp = mcp

    async def _synth(self, text: str) -> bytes:
        raw = await call_mcp_tool(self._mcp, "speak", {"text": text}, server="tts")
        return base64.b64decode(json.loads(raw)["audio_b64"])

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        raw = text
        text = normalize_for_piper(plain_speech(text))
        if not text.strip():
            logger.warning(f"TTS: nothing speakable in {raw!r}")
            return

        await self.start_tts_usage_metrics(text)
        t_call = time.monotonic()
        chunks = split_for_speed(text)
        logger.info(f"TTS request: {len(text)} chars in {len(chunks)} chunk(s): {chunks}")

        async def wav_stream() -> AsyncGenerator[bytes, None]:
            nxt = asyncio.create_task(self._synth(chunks[0]))
            try:
                for i in range(len(chunks)):
                    wav = await nxt
                    if i == 0:
                        logger.info(
                            f"TTS first speech ready {time.monotonic() - t_call:.2f}s after request"
                        )
                    if i + 1 < len(chunks):  # prefetch next while this one plays
                        nxt = asyncio.create_task(self._synth(chunks[i + 1]))
                    # The first chunk keeps its header: the base class reads
                    # the sample rate from it and strips it. Later chunks are
                    # the same voice, so their headers are cut here.
                    data = wav if i == 0 else wav[self._WAV_HEADER_BYTES:]
                    first = self._WAV_CHUNK_BYTES + (self._WAV_HEADER_BYTES if i == 0 else 0)
                    yield data[:first]
                    for j in range(first, len(data), self._WAV_CHUNK_BYTES):
                        yield data[j : j + self._WAV_CHUNK_BYTES]
            finally:
                nxt.cancel()  # barge-in or error: drop pending synthesis

        try:
            async for frame in self._stream_audio_frames_from_iterator(
                wav_stream(), strip_wav_header=True, context_id=context_id
            ):
                yield frame
        except asyncio.CancelledError:
            logger.info(f"TTS cancelled (barge-in) after {time.monotonic() - t_call:.2f}s")
            raise
        except Exception as e:
            logger.error(f"TTS MCP error: {e}")
            yield ErrorFrame(error=f"TTS error: {e}")
