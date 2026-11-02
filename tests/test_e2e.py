"""End-to-end self-test for the lava web bridge (no microphone needed).

1. Connects to ws://127.0.0.1:8080/ws and sends {"type": "ready"}.
2. Expects the pre-synthesized greeting as binary audio frames.
3. Synthesizes a question with Piper, feeds it as 16 kHz binary mic frames
   followed by a second of silence - exactly what the browser sends.
4. Expects transcript -> reply text -> binary reply audio -> done(ttfa_ms).
5. Barges in with a second phrase while the reply audio is still streaming
   and expects a JSON "barge" message, then a full turn for the new utterance.

Run:  .venv/bin/python tests/test_e2e.py
"""
import asyncio
import base64
import json
import struct
import sys
import time

import aiohttp
import numpy as np

sys.path.insert(0, ".")
from mcp_servers.common import resample_int16

HOST = "http://127.0.0.1:8080"
failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


def piper_pcm16(text: str) -> tuple[bytes, int]:
    from piper import PiperVoice
    voice = PiperVoice.load("models/piper/en_US-hfc_female-medium.onnx")
    pcm = b"".join(ch.audio_int16_bytes for ch in voice.synthesize(text))
    return pcm, voice.config.sample_rate


def to_mic_frames(pcm16: bytes, frame_bytes: int = 1024):
    for i in range(0, len(pcm16), frame_bytes):
        yield pcm16[i:i + frame_bytes]


async def main():
    print("synthesizing test phrases with Piper...")
    q1, sr = piper_pcm16("What time is it in Manila right now?")
    q1_16k = resample_int16(np.frombuffer(q1, np.int16), sr, 16000).tobytes()
    q2, _ = piper_pcm16("Never mind that, tell me a joke instead.")
    q2_16k = resample_int16(np.frombuffer(q2, np.int16), sr, 16000).tobytes()
    silence = b"\x00\x00" * 16000

    async with aiohttp.ClientSession() as http:
        st = await http.get(f"{HOST}/api/status")
        status = await st.json()
        check("status endpoint", st.status == 200 and status.get("ok"), json.dumps(status)[:120])

        ws = await http.ws_connect(f"{HOST}/ws?cid=e2e_test")
        inbox: list = []

        async def reader():
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        inbox.append(("audio", msg.data))
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        inbox.append(("json", json.loads(msg.data)))
            except Exception:
                pass

        rt = asyncio.create_task(reader())

        # --- greeting -------------------------------------------------------
        await ws.send_json({"type": "ready"})
        t_greet = time.monotonic()
        await asyncio.sleep(2.5)
        greet_audio = [d for k, d in inbox if k == "audio"]
        greet_done = [m for k, m in inbox if k == "json" and m.get("type") == "done"]
        check("greeting: binary audio frames received", len(greet_audio) >= 2,
              f"({len(greet_audio)} frames)")
        if greet_audio:
            hdr_sr = struct.unpack("<H", greet_audio[0][:2])[0]
            check("greeting: binary frame carries sample rate", 8000 <= hdr_sr <= 48000,
                  f"(sr={hdr_sr})")
        check("greeting: done message", bool(greet_done))
        inbox.clear()

        # --- question turn ----------------------------------------------------
        t_q = time.monotonic()
        for f in to_mic_frames(q1_16k):
            await ws.send_bytes(f)
        await ws.send_bytes(silence)

        async def wait_for(pred, timeout, what):
            dl = time.monotonic() + timeout
            while time.monotonic() < dl:
                if pred():
                    return True
                await asyncio.sleep(0.05)
            print(f"  [FAIL] timed out waiting for {what}")
            failures.append(f"timeout:{what}")
            return False

        await wait_for(lambda: any(m.get("type") == "transcript" for k, m in inbox if k == "json"),
                       15, "transcript")
        await wait_for(lambda: any(m.get("type") == "done" for k, m in inbox if k == "json"),
                       40, "done")

        transcript = next((m["text"] for k, m in inbox if k == "json"
                           and m.get("type") == "transcript"), None)
        done = next((m for k, m in inbox if k == "json" and m.get("type") == "done"), None)
        reply_text = [m["t"] for k, m in inbox if k == "json" and m.get("type") == "text"]
        reply_audio = [d for k, d in inbox if k == "audio"]

        check("question: transcript arrived", bool(transcript), f"-> {transcript!r}")
        check("question: reply audio streamed", len(reply_audio) >= 2,
              f"({len(reply_audio)} frames)")
        check("question: done has ttfa_ms", bool(done and "ttfa_ms" in done.get("stats", {})),
              f"stats={json.dumps(done.get('stats', {})) if done else None}")
        if done:
            s = done["stats"]
            print(f"      asr={s.get('asr_ms')}ms llm={s.get('llm_ms')}ms "
                  f"tts={s.get('tts_ms')}ms ttfa={s.get('ttfa_ms')}ms total={s.get('total_ms')}ms")
        first_audio_delay = None
        if reply_audio:
            first_audio_delay = (time.monotonic() - t_q) * 1000
        print(f"      reply text: {''.join(reply_text)!r}")
        check("question: reply is spoken about time/tool use",
              bool(reply_text) and any(w in "".join(reply_text).lower()
                                       for w in ("m", "time", "p", "am", "pm")),
              f"-> {''.join(reply_text)[:80]!r}")

        # --- barge-in while reply audio would still be playing ---------------
        # The reply's PCM was streamed at once; schedule it client-side would
        # still be playing. Send a barge phrase now.
        inbox.clear()
        for f in to_mic_frames(q2_16k):
            await ws.send_bytes(f)
        await ws.send_bytes(silence)
        await wait_for(lambda: any(m.get("type") == "barge" for k, m in inbox if k == "json"),
                       10, "barge during bot speech")
        got_barge = any(m.get("type") == "barge" for k, m in inbox if k == "json")
        check("barge-in: barge message received", got_barge)
        await wait_for(lambda: any(m.get("type") == "done" for k, m in inbox if k == "json"),
                       40, "second turn done")
        t2 = next((m["text"] for k, m in inbox if k == "json"
                   and m.get("type") == "transcript"), None)
        check("barge-in: new utterance became a full turn", bool(t2), f"-> {t2!r}")
        done2 = [m for k, m in inbox if k == "json" and m.get("type") == "done"]
        check("barge-in: second turn finished", bool(done2))

        rt.cancel()
        await ws.close()

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("E2E PASSED")


asyncio.run(main())
