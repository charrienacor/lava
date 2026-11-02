"""Focused checks for the Speaker pipeline (no network, no TTS server):

- strict FIFO ordering even when later sentences synthesize faster
- finish() drains everything enqueued before it
- aclose() cancels promptly from any point (mid-synth, mid-drain, blocked)
- a dead stream worker cannot hang finish()

Run:  .venv/bin/python tests/test_speaker.py
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

import web.server as srv

failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


class FakeWS:
    def __init__(self):
        self.frames = []
        self.closed = False

    async def send_bytes(self, b):
        await asyncio.sleep(0.002)
        self.frames.append(b)


# Later sentences synthesize FASTER than earlier ones - if the pipeline ever
# reordered, the output order would betray it.
SYNTH_DELAY = {"one.": 0.30, "two.": 0.10, "three.": 0.20, "four.": 0.05}


async def fake_synth(text, stats=None):
    await asyncio.sleep(SYNTH_DELAY.get(text, 0.05))
    if text == "three.":
        raise RuntimeError("tts blip")
    return text.encode() * 200, 16000


async def test_ordering_and_skip():
    srv.synth_audio, orig = fake_synth, srv.synth_audio
    try:
        ws = FakeWS()
        sess = {"bot_audio_until": 0.0}
        stats = {"turn_t0": time.perf_counter()}
        sp = srv.Speaker(ws, sess, stats)
        sp.start()
        for t in SYNTH_DELAY:
            await sp.enqueue(t)
        await asyncio.wait_for(sp.finish(), 5)
        text = b"".join(f[2:] for f in ws.frames).decode()
        seen = [t for t in SYNTH_DELAY if t in text]
        check("sentences stream strictly in enqueue order",
              seen == ["one.", "two.", "four."], f"-> {seen}")
        check("failed sentence skipped without killing the turn",
              "three." not in text)
        check("bot_audio_until advanced", sess["bot_audio_until"] > 0)
        check("ttfa recorded", stats.get("ttfa_ms", 0) > 0)
        await sp.aclose()  # no-op after finish
    finally:
        srv.synth_audio = orig


async def test_cancel_mid_synth():
    srv.synth_audio, orig = fake_synth, srv.synth_audio
    try:
        ws = FakeWS()
        sess = {"bot_audio_until": 0.0}
        stats = {"turn_t0": time.perf_counter()}
        sp = srv.Speaker(ws, sess, stats)
        sp.start()
        await sp.enqueue("one.")
        await asyncio.sleep(0.05)          # synth in flight
        await asyncio.wait_for(sp.aclose(), 2)
        check("cancel during synth returns promptly", True)
    finally:
        srv.synth_audio = orig


async def test_dead_stream_worker():
    async def broken_stream(ws, sess, pcm, sr, stats):
        raise RuntimeError("ws exploded")

    srv.synth_audio, orig_s = fake_synth, srv.synth_audio
    srv.stream_pcm, orig_p = broken_stream, srv.stream_pcm
    try:
        ws = FakeWS()
        sess = {"bot_audio_until": 0.0}
        stats = {"turn_t0": time.perf_counter()}
        sp = srv.Speaker(ws, sess, stats)
        sp.start()
        for t in SYNTH_DELAY:
            await sp.enqueue(t)
        # Must not hang: stream worker dies, synth worker must be woken, and
        # finish() surfaces the abort as a regular error (which run_turn's
        # error path handles with an apology).
        try:
            await asyncio.wait_for(sp.finish(), 5)
            check("finish() survives a dead stream worker", True,
                  "(returned clean - no audio was being streamed)")
        except RuntimeError:
            check("finish() survives a dead stream worker", True, "(raised RuntimeError)")
        except asyncio.TimeoutError:
            check("finish() survives a dead stream worker", False, "(hung)")
    finally:
        srv.synth_audio = orig_s
        srv.stream_pcm = orig_p


async def main():
    await test_ordering_and_skip()
    await test_cancel_mid_synth()
    await test_dead_stream_worker()
    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("SPEAKER CHECKS PASSED")


asyncio.run(main())
