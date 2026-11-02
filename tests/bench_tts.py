"""TTS backend benchmark: piper-medium vs piper-low vs groq (Orpheus).

Synthesizes a few representative agent lines with each engine, reports
latency / audio seconds / RTF, and drops one WAV per config into
/tmp/lava_tts_bench/ so you can compare quality by ear.

Notes:
  - en_US-amy-low is downloaded on first use (~20 MB).
  - groq uses your existing GROQ_API_KEY and burns a few TTS quota units.
  - "latency" here is the full synthesis call - what the bridge waits for
    before the first byte of a sentence can be streamed (TTFA proxy).

Run:  .venv/bin/python tests/bench_tts.py [--runs 2]
"""
import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

LINES = [
    ("greeting", "Hi there! I'm Lava. What can I do today?"),
    ("ack", "Sure, let me check the weather for you."),
    ("time", "It's 12:25 AM on Tuesday, November 3, 2026 in Manila."),
    ("long", "Sure, I can help with that. The weather in Manila today is mostly cloudy "
             "with a chance of light rain in the afternoon, and temperatures should reach "
             "about 31 degrees, so you might want an umbrella."),
]

CONFIGS = [
    ("piper-medium", {"TTS_BACKEND": "piper", "PIPER_VOICE": "en_US-hfc_female-medium"}),
    ("piper-low", {"TTS_BACKEND": "piper", "PIPER_VOICE": "en_US-amy-low"}),
    ("groq-hannah", {"TTS_BACKEND": "groq", "GROQ_TTS_VOICE": "hannah"}),
]

OUT_DIR = Path("/tmp/lava_tts_bench")


def load_engine(spec: dict):
    for k, v in spec.items():
        os.environ[k] = v
    import mcp_servers.tts_server as tts
    tts = importlib.reload(tts)
    return tts.BACKENDS[spec["TTS_BACKEND"]]()


async def bench(runs: int):
    OUT_DIR.mkdir(exist_ok=True)
    summary = {}
    for label, spec in CONFIGS:
        print(f"\n=== {label} ===")
        try:
            eng = load_engine(spec)
            await eng.synthesize("Ready.", None)  # warmup (also downloads voice)
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")
            summary[label] = None
            continue
        rows = []
        for name, text in LINES:
            times = []
            for i in range(runs):
                t0 = time.perf_counter()
                try:
                    pcm, sr = await eng.synthesize(text, None)
                except Exception as e:
                    msg = str(e)
                    if "model_terms_required" in msg:
                        print(f"  [ERROR] {name}: Groq needs a one-time terms "
                              f"acceptance - open https://console.groq.com/playground"
                              f"?model=canopylabs%2Forpheus-v1-english then re-run")
                    else:
                        print(f"  [ERROR] {name}: {msg}")
                    times = None
                    break
                times.append(time.perf_counter() - t0)
            if not times:
                summary[label] = None
                break
            secs = len(pcm) / 2 / sr
            lat = min(times)  # steady-state, not cold-cache
            rtf = lat / secs if secs else 0.0
            rows.append((name, lat * 1000, secs, rtf, len(times)))
            print(f"  {name:<9} {lat*1000:6.0f} ms  -> {secs:5.2f}s audio "
                  f"(RTF {rtf:5.2f}, {len(pcm)/1024:5.0f} KiB @ {sr} Hz)")
            if name == "greeting":
                from mcp_servers.common import pcm16_to_wav
                (OUT_DIR / f"{label}-greeting.wav").write_bytes(pcm16_to_wav(pcm, sr))
        if rows:
            mean_lat = sum(r[1] for r in rows) / len(rows)
            mean_rtf = sum(r[3] for r in rows) / len(rows)
            summary[label] = (mean_lat, mean_rtf, sr)
            print(f"  mean: {mean_lat:.0f} ms per line, RTF {mean_rtf:.2f}")
            print(f"  sample wav: {OUT_DIR / (label + '-greeting.wav')}")

    print("\n=== summary (lower is better) ===")
    print(f"{'config':<14}{'mean latency':>14}{'mean RTF':>10}   sr")
    for label, s in summary.items():
        if s is None:
            print(f"{label:<14}{'FAILED':>14}")
        else:
            print(f"{label:<14}{s[0]:>11.0f} ms{s[1]:>9.2f}   {s[2]} Hz")


if __name__ == "__main__":
    runs = 2
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    asyncio.run(bench(runs))
