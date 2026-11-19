"""tts: chunked Piper replies stream as one gap-free PCM run at the voice's rate."""

import asyncio
import base64
import io
import json
import wave

import tts as tts_mod

RATE = 22050


def _wav(samples: int, value: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(value.to_bytes(2, "little", signed=True) * samples)
    return buf.getvalue()


def test_reply_audio_has_no_lead_in_and_no_stray_headers(monkeypatch):
    # Two chunks: each distinct, non-silent sample value marks its chunk.
    pcm = {1: _wav(10000, 1), 2: _wav(3000, 2)}
    calls = []

    async def fake_call(_client, name, args, server=None):
        calls.append(args["text"])
        return json.dumps({"audio_b64": base64.b64encode(pcm[len(calls)]).decode()})

    monkeypatch.setattr(tts_mod, "call_mcp_tool", fake_call)
    svc = tts_mod.McpPiperTTSService(mcp=None)

    async def no_metrics(*_a, **_k):
        pass

    monkeypatch.setattr(svc, "start_tts_usage_metrics", no_metrics)

    async def run():
        text = "Here is the first part of the reply, and this is the second part."
        return [f async for f in svc.run_tts(text, "ctx")]

    frames = asyncio.run(run())
    assert len(calls) == 2
    audio = b"".join(f.audio for f in frames)
    assert {f.sample_rate for f in frames} == {RATE}
    # Exactly the two chunks' samples, back to back: no silence before the
    # speech, no 44-byte header decoded as noise between chunks.
    expected = (1).to_bytes(2, "little") * 10000 + (2).to_bytes(2, "little") * 3000
    assert audio == expected
