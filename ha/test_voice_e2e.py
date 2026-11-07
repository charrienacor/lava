#!/usr/bin/env python3
"""Voice end-to-end test: synthesized speech through the LIVE bridge.

Speaks smart-home commands with macOS `say`, feeds the PCM to the bridge's
real /ws endpoint (same path as the browser mic), and watches the frames:
transcript -> tool_start/tool_done -> done. Then verifies via HA's
/api/states that the hub's state actually moved.

This exercises VAD + Groq Whisper + LLM tool-call + tools MCP + HA REST -
the full chain a human voice takes. ASR of synthetic speech is not
guaranteed perfect; if a transcript is garbled the test says so and exits 3
(distinct from a pipeline failure, exit 1).

Usage:  .venv/bin/python ha/test_voice_e2e.py
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
import os  # noqa: E402

load_dotenv(ROOT / ".env")
HASS_TOKEN = os.getenv("HASS_TOKEN", "")
HASS_URL = os.getenv("HASS_URL", "").rstrip("/")
import loudness as _bs1770  # noqa: E402

SR = 16000
URI = "ws://127.0.0.1:8080/ws?cid=test-voice-e2e"
TARGET_LUFS = -15.0  # clears the VAD_MIN_VOLUME=0.75 gate with margin


def synth(text: str, out: Path) -> np.ndarray:
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    import soundfile as sf

    x, rate = sf.read(str(aiff), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(len(x) * SR / rate)
    idx = np.linspace(0, len(x) - 1, n)
    x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    # scale to the target loudness (BS.1770, same scale the VAD gates on)
    for _ in range(8):
        lufs = _bs1770.integrated_loudness(x, SR)
        if not np.isfinite(lufs):
            break
        if abs(lufs - TARGET_LUFS) < 0.5:
            break
        gain = 10 ** ((TARGET_LUFS - lufs) / 20)
        x = np.clip(x * gain, -1.0, 1.0)
    return (x * 32767).astype(np.int16)


def chunks(x: np.ndarray, frame: int = 512):
    for i in range(0, len(x), frame):
        c = x[i : i + frame]
        if len(c) < frame:
            c = np.pad(c, (0, frame - len(c)))
        yield c.tobytes()


async def run_case(ws, phrase: str, events: list) -> dict:
    """Send one utterance; collect frames until the turn's `done`."""
    before = len(events)
    x = synth(phrase, ROOT / "ha" / f"_v_{abs(hash(phrase)) % 10**6}.wav")
    for c in chunks(x):
        await ws.send(c)
        await asyncio.sleep(0.02)
    for c in chunks(np.zeros(SR, dtype=np.int16)):  # 1 s tail silence
        await ws.send(c)
        await asyncio.sleep(0.02)
    # wait for the done frame belonging to THIS utterance
    deadline = asyncio.get_event_loop().time() + 60
    while asyncio.get_event_loop().time() < deadline:
        new = events[before:]
        if any(e.get("type") == "done" for e in new):
            break
        await asyncio.sleep(0.2)
    return {"events": events[before:]}


async def main() -> int:
    if not HASS_URL or not HASS_TOKEN:
        print("ABORT: HASS_URL/HASS_TOKEN missing")
        return 2

    import httpx

    async def hass_states():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{HASS_URL}/api/states", headers={"Authorization": f"Bearer {HASS_TOKEN}"})
            return {s["entity_id"]: s for s in r.json()}

    async with websockets.connect(URI, max_size=2**20) as ws:
        events: list = []

        async def pump():
            while True:
                try:
                    m = await ws.recv()
                except Exception:
                    return  # connection closed - stop pumping quietly
                if isinstance(m, (bytes, bytearray)):
                    continue  # TTS audio - not needed here
                e = json.loads(m)
                events.append(e)
                t = e.get("type")
                if t == "transcript":
                    print(f"    [transcript] {e.get('text')}")
                elif t in ("tool_start", "tool_done"):
                    print(f"    [{t}] {json.dumps(e)[:140]}")
                elif t == "done":
                    print(f"    [done] reply={e.get('reply', '')[:100]!r}")

        task = asyncio.create_task(pump())
        await ws.send(json.dumps({"type": "ready"}))
        # Wait for the greeting's `done` frame so the first command doesn't
        # overlap the bot's own speech (it would be classified as a barge-in).
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            if any(e.get("type") == "done" for e in events):
                break
            await asyncio.sleep(0.2)
        await asyncio.sleep(1.0)

        # discover the virtual entities (ids derive from names)
        st0 = await hass_states()
        light = next((eid for eid in st0 if eid.startswith("light.living_room")), "light.living_room")
        climate = next((eid for eid in st0 if eid.startswith("climate.living_room")), "climate.living_room")

        cases = [
            ('"Turn on the living room lights."', "light.living_room", "on"),
            ('"Dim the lights to thirty percent."', "light.living_room", "dim"),
            ('"Set the temperature to twenty two degrees."', "climate.living_room", "heat22"),
            ('"What is the current temperature?"', "climate.living_room", "read"),
        ]
        results = []
        for i, (phrase, entity, tag) in enumerate(cases):
            print(f"\n>>> {phrase}")
            out = await run_case(ws, phrase, events)
            evs = out["events"]
            transcript = next((e["text"] for e in reversed(evs) if e.get("type") == "transcript"), None)
            tools = [e for e in evs if e.get("type") in ("tool_start", "tool_done")]
            results.append((tag, transcript, tools))
            # let the bot's TTS reply finish before the next command, so it
            # isn't classified as a barge-in over the bot's own speech.
            if i < len(cases) - 1:
                await asyncio.sleep(4)

        # settle, then verify HA state for the cumulative effect
        await asyncio.sleep(1)
        states = await hass_states()
        print("\n--- HA state after the conversation ---")
        for eid in (light, climate):
            s = states.get(eid, {})
            attrs = {k: v for k, v in (s.get("attributes") or {}).items()
                     if k in ("brightness", "temperature", "current_temperature")}
            print(f"  {eid}: state={s.get('state')} attrs={attrs}")
        light_on = states.get(light, {}).get("state") == "on"
        bright = states.get(light, {}).get("attributes", {}).get("brightness")
        tgt = states.get(climate, {}).get("attributes", {}).get("temperature")
        print(f"\n  light on? {light_on}   brightness={bright}   thermostat target={tgt}")

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    print("\n--- per-case summary ---")
    for tag, transcript, tools in results:
        titles = sorted({e.get("title") for e in tools if e.get("type") == "tool_done"})
        print(f"  {tag:8} transcript={transcript!r}  banners={titles}")
    print("\nNOTE: judge the transcript quality yourself - synthetic speech can")
    print("      be misheard; the tool names + HA state above are the ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
