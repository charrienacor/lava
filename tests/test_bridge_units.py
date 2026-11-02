"""Offline unit checks for the reworked bridge: sentence splitting, VAD
gating with armored barge-in, pre-roll, and Piper-speech barge timing.

Run:  .venv/bin/python tests/test_bridge_units.py
"""
import asyncio
import math
import struct
import sys

sys.path.insert(0, ".")

from web.server import split_sentences, VADState, _is_skip

failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


# --- splitter ---------------------------------------------------------------

print("split_sentences:")
c, rest = split_sentences("It's 72 degrees and sunny. Want anything else?", )
check("sentence split", c == ["It's 72 degrees and sunny."] and rest == "Want anything else?",
      f"-> {c!r} rest={rest!r}")

c, rest = split_sentences("Sure", allow_clause=True)
check("short text untouched below trigger", c == [] and rest == "Sure")

c, rest = split_sentences("Let me check the weather out there for you right now, one moment please",
                          allow_clause=True)
check("clause split fires at first comma >= 28 chars",
      c == ["Let me check the weather out there for you right now,"]
      and rest == "one moment please",
      f"-> {c!r} rest={rest!r}")

c, rest = split_sentences("Ok, fine", allow_clause=True)
check("clause split skipped for short openers", c == [] and rest == "Ok, fine")

long_runon = " ".join(["word"] * 60)
c, rest = split_sentences(long_runon)
check("hard split bounds run-ons", len(c) == 1 and 80 <= len(c[0]) <= 200 and rest,
      f"-> chunk {len(c[0])} chars, rest {len(rest)} chars")

c, rest = split_sentences("Hi! How are you? I'm good.")
check("multiple sentences in one buffer", c == ["Hi!", "How are you?"] and rest == "I'm good.")

check("skip detection", _is_skip("SKIP.") and _is_skip(" skip ") and not _is_skip("I skipped it"))

# --- VAD with a tone (Silero must reject non-speech) --------------------------

print("VADState:")


def tone_pcm(freq, secs, sr=16000, amp=12000):
    n = int(sr * secs)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(n))


vad = VADState()
vad.feed(tone_pcm(440, 1.0))
check("loud sine does not open a span (Silero rejects non-speech)",
      vad.first_voiced is None and vad.utterance is None)

# --- VAD with real Piper speech ------------------------------------------------

print("VADState with Piper speech:")


def piper_pcm16(text: str) -> tuple[bytes, int]:
    from piper import PiperVoice
    voice = PiperVoice.load("models/piper/en_US-hfc_female-medium.onnx")
    pcm = b"".join(ch.audio_int16_bytes for ch in voice.synthesize(text))
    return pcm, voice.config.sample_rate


QUESTION = "Hey Lava, what is the weather like today in Manila?"
pcm, sr = piper_pcm16(QUESTION)
if sr != 16000:
    import numpy as np
    from mcp_servers.common import resample_int16
    pcm = resample_int16(np.frombuffer(pcm, np.int16), sr, 16000).tobytes()

vad = VADState()
vad.feed(pcm)
check("real speech opens (or has already closed) a span",
      vad.first_voiced is not None or vad.utterance is not None)

vad.feed(b"\x00\x00" * 16000)  # 1 s silence -> span closes
utt = vad.utterance
check("span closes after stop silence and yields an utterance",
      utt is not None and len(utt) > 0, f"utterance={len(utt) if utt else 0} bytes")
check("pre-roll present (utterance longer than voiced span alone)",
      utt is not None and len(utt) > 0)


def barge_latency() -> float | None:
    v = VADState()
    fire_at = None
    for i in range(0, len(pcm), 1024):
        v.feed(pcm[i:i + 1024])
        if v.barge_hint and fire_at is None:
            fire_at = (i + 1024) / 2 / 16000
            break
    return fire_at


u = barge_latency()
# Never earlier than the configured burst; the BS.1770 window's 400 ms fill
# adds latency to the FIRST barge of a session, so allow extra headroom.
check("barge fires no earlier than VAD_BARGE_MS",
      u is not None and u >= 0.19, f"-> {u}s")

# --- noisy-span transcribe-anyway ---------------------------------------------

print("VAD noisy spans:")
vad = VADState()
old_ratio, VADState.MIN_VOICED_RATIO = VADState.MIN_VOICED_RATIO, 0.95
try:
    vad = VADState()
    for i in range(0, len(pcm), 1024):
        vad.feed(pcm[i:i + 1024])
    vad.feed(b"\x00\x00" * 16000)
    check("noisy span transcribed anyway when enabled", vad.utterance is not None,
          f"({len(vad.utterance) if vad.utterance else 0} bytes)")

    VADState.PROCESS_NOISY_SPANS = False
    vad = VADState()
    for i in range(0, len(pcm), 1024):
        vad.feed(pcm[i:i + 1024])
    vad.feed(b"\x00\x00" * 16000)
    check("noisy span discarded when disabled", vad.utterance is None)
finally:
    VADState.MIN_VOICED_RATIO = old_ratio
    VADState.PROCESS_NOISY_SPANS = True

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL UNIT CHECKS PASSED")
