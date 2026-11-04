"""Standardized latency benchmark for the Lava voice agent.

Measures the full interaction loop against a running stack. Designed for
classroom use — every student measures the exact same thing with zero
per-person setup:

  - Stimulus audio loads from tests/assets/*.wav (identical for everyone;
    regenerate with tests/generate_assets.py if the folder is missing).
  - Every run opens a fresh session with a unique id, so results never
    depend on conversation state.
  - The only requirement is a running stack (./run.sh) and a .env with a
    Groq key. Results are tagged with --label for submissions.

Metric definitions (all in milliseconds):

  client_ttfa    last mic-speech byte sent -> first reply-audio byte received.
                 What a user perceives: includes the VAD's 0.2 s end-of-speech
                 detection, ASR, the LLM's first clause, the first TTS chunk,
                 and transport.
  client_total   last mic-speech byte sent -> last reply-audio byte received
                 (whole reply streamed).
  server_ttfa    the bridge's own measurement (turn start, i.e. including ASR,
                 -> first audio streamed). Excludes the VAD end-of-speech
                 window, so client_ttfa ≈ server_ttfa + ~200-300 ms.
  asr/llm/tts    the bridge's component breakdown for the turn.
  barge_cut      first byte of interrupting speech sent -> "barge" message
                 received (the moment the bot's audio is cut). t_resp is
                 measured separately: cut -> first response audio.

Scenarios:

  no_tool       "Tell me a fun fact about the ocean."   (pure LLM reply)
  tool_time     "What time is it in Manila right now?" (get_time)
  tool_weather  "What's the weather like in Quezon City today?" (get_weather)
  barge_cut     interrupt the greeting and measure the cut (t_stop) and the
                response (t_resp, Full-Duplex-Bench v1.5: cut -> first audio
                of the reply).

Usage:
  .venv/bin/python tests/bench_latency.py --runs 3 --label alice
"""
import argparse
import asyncio
import json
import platform
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import aiohttp

ASSETS = Path(__file__).resolve().parent / "assets"
RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = ["no_tool", "tool_time", "tool_weather"]
FRAME_BYTES = 1024          # 512 samples = 32 ms of audio at 16 kHz
PACE = 0.028                # slightly faster than real time, like the browser


def pct(values, p):
    xs = sorted(values)
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def load_assets():
    """Load the stimulus WAVs (stdlib only). Missing files are regenerated
    when piper is available."""
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    sr = manifest["sample_rate"]
    wavs = {}
    for name, text in manifest["phrases"].items():
        path = ASSETS / f"{name}.wav"
        if not path.exists():
            _regenerate()
        with wave.open(str(path), "rb") as wf:
            if (wf.getframerate() != sr or wf.getnchannels() != 1
                    or wf.getsampwidth() != 2):
                raise SystemExit(f"{path}: expected {sr} Hz mono 16-bit WAV")
            wavs[name] = {"pcm": wf.readframes(wf.getnframes()),
                          "text": text, "sr": sr}
    return wavs


def _regenerate():
    import subprocess
    import sys
    print("  asset WAVs missing - regenerating with Piper ...")
    r = subprocess.run([sys.executable, str(Path(__file__).parent /
                                            "generate_assets.py")])
    if r.returncode != 0:
        raise SystemExit("Could not generate assets. Run "
                         "tests/generate_assets.py once (needs piper) or "
                         "copy tests/assets/ from the repository.")


class Turn:
    """One measured interaction."""

    def __init__(self, name):
        self.name = name
        self.t_speech_end = None      # last speech byte sent
        self.t_first_audio = None     # first reply audio byte received
        self.t_last_audio = None      # last reply audio byte received
        self.t_done = None
        self.t_resp_audio = None    # first response audio after a barge cut
        self.transcript = None
        self.stats = {}
        self.tools = []
        self.error = None

    @property
    def client_ttfa(self):
        if self.t_first_audio and self.t_speech_end:
            return (self.t_first_audio - self.t_speech_end) * 1000

    @property
    def client_total(self):
        if self.t_last_audio and self.t_speech_end:
            return (self.t_last_audio - self.t_speech_end) * 1000

    @property
    def t_resp(self):
        """Full-Duplex-Bench v1.5: barge cut -> first response audio (ms)."""
        if self.t_first_audio and self.t_resp_audio:
            return (self.t_resp_audio - self.t_first_audio) * 1000

    def record(self, ok: bool) -> dict:
        return dict(self.__dict__,
                    client_ttfa=self.client_ttfa,
                    client_total=self.client_total,
                    t_resp=self.t_resp, ok=ok)


class Session:
    """One WebSocket session with browser-identical mic streaming."""

    def __init__(self, http, port):
        self.http, self.port = http, port
        self.ws = None
        self.inbox = []               # (monotonic_time, kind, payload)
        self.cid = uuid.uuid4().hex[:8]

    async def open(self, keep_greeting: bool = False):
        self.ws = await self.http.ws_connect(
            f"http://127.0.0.1:{self.port}/ws?cid={self.cid}")
        self._reader = asyncio.create_task(self._read())
        await self.ws.send_json({"type": "ready"})
        t, m = await self.wait_done(20)          # greeting (pre-synthesized)
        if t is None:
            raise RuntimeError("no greeting from the bridge - is the stack "
                               "running? (./run.sh)")
        if not keep_greeting:
            self.inbox.clear()

    async def _read(self):
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    self.inbox.append((time.monotonic(), "audio", msg.data))
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    self.inbox.append((time.monotonic(), "json",
                                       json.loads(msg.data)))
        except Exception as e:
            # A dead reader makes every scenario "miss" silently - surface it.
            print(f"  [reader {self.cid}] died: {e!r}", flush=True)

    async def speak(self, pcm: bytes, pad_frames: int = 10) -> float:
        """Stream a phrase as mic input at real-time pace, then the stop pad.
        Returns the monotonic time the last speech byte was sent."""
        for j in range(0, len(pcm), FRAME_BYTES):
            await self.ws.send_bytes(pcm[j:j + FRAME_BYTES])
            await asyncio.sleep(PACE)
        t_speech_end = time.monotonic()
        for _ in range(pad_frames):              # 320 ms stop pad
            await self.ws.send_bytes(b"\x00\x00" * FRAME_BYTES)
            await asyncio.sleep(PACE)
        return t_speech_end

    async def wait_done(self, timeout: float = 90):
        dl = time.monotonic() + timeout
        while time.monotonic() < dl:
            for t, kind, m in self.inbox:
                if kind == "json" and m.get("type") == "done":
                    return t, m
            await asyncio.sleep(0.02)
        return None, None

    def collect(self, turn: Turn, since: float = 0.0) -> bool:
        """Pull this turn's events (those after `since`) out of the inbox,
        up to and including its done/error."""
        done_at = None
        remaining = []
        for t, kind, m in self.inbox:
            if t < since or done_at is not None:
                if done_at is None:
                    remaining.append((t, kind, m))
                continue
            if kind == "audio":
                if turn.t_first_audio is None:
                    turn.t_first_audio = t
                turn.t_last_audio = t
            elif kind == "json":
                mt = m.get("type")
                if mt == "transcript":
                    turn.transcript = m.get("text")
                elif mt == "tool":
                    turn.tools.append(m.get("name"))
                elif mt == "done":
                    turn.stats = m.get("stats", {})
                    turn.t_done = t
                    done_at = t
                elif mt == "error":
                    turn.error = str(m.get("message"))[:120]
                    done_at = t
        self.inbox = remaining
        return done_at is not None

    async def close(self):
        try:
            self._reader.cancel()
            await self.ws.close()
        except Exception:
            pass


async def run_benchmark(runs: int, port: int, label: str):
    wavs = load_assets()
    print("stimuli (identical for every student):")
    for name in SCENARIOS + ["barge"]:
        print(f"  {name:<14} {wavs[name]['text']!r}")

    results = {"label": label,
               "generated": datetime.now().isoformat(timespec="seconds"),
               "host": platform.node(),
               "python": platform.python_version(),
               "runs": runs,
               "stimuli": {k: v["text"] for k, v in wavs.items()},
               "scenarios": {}}

    async with aiohttp.ClientSession() as http:
        # --- turn scenarios: fresh session per run -------------------------
        for name in SCENARIOS:
            runs_out = []
            for i in range(runs):
                sess = Session(http, port)
                await sess.open()
                turn = Turn(name)
                turn.t_speech_end = await sess.speak(wavs[name]["pcm"])
                t_done, _ = await sess.wait_done()
                ok = sess.collect(turn, since=turn.t_speech_end - 0.1)
                turn.error = turn.error or (None if ok else "timeout")
                runs_out.append(turn.record(bool(ok)))
                print(f"  [{name} #{i + 1}] ttfa={turn.client_ttfa and round(turn.client_ttfa)}ms "
                      f"total={turn.client_total and round(turn.client_total)}ms "
                      f"tools={turn.tools or '-'} asr={round(turn.stats.get('asr_ms', 0))} "
                      f"llm={round(turn.stats.get('llm_ms', 0))} :: {(turn.transcript or '')[:44]!r}"
                      f"{' !! ' + str(turn.error) if turn.error else ''}")
                await sess.close()
            results["scenarios"][name] = runs_out

        # --- barge-in cut (barges the pre-synthesized greeting) ------------
        runs_out = []
        for i in range(runs):
            sess = Session(http, port)
            await sess.open(keep_greeting=True)
            # wait until the greeting is actually streaming
            first_audio = None
            dl = time.monotonic() + 5
            while time.monotonic() < dl and first_audio is None:
                for t, kind, m in sess.inbox:
                    if kind == "audio":
                        first_audio = t
                await asyncio.sleep(0.01)
            if first_audio is None:
                print("  [barge_cut] greeting never streamed; skipping run")
                await sess.close()
                continue
            await asyncio.sleep(0.1)

            turn = Turn("barge_cut")
            t_start = time.monotonic()
            cut_at = None
            pcm = wavs["barge"]["pcm"]
            # Stream the interrupting phrase, then keep the mic alive with
            # silence frames like a real browser does - the deferred
            # barge-in (backchannel filtering) cuts a bit after speech
            # STARTS, so the socket has to keep feeding the VAD.
            speech_frames = len(pcm) // FRAME_BYTES
            tail_frames = int(1.5 / PACE)          # ~1.5 s of silence tail
            for j in range(speech_frames + tail_frames):
                if j < speech_frames:
                    await sess.ws.send_bytes(pcm[j * FRAME_BYTES:(j + 1) * FRAME_BYTES])
                else:
                    await sess.ws.send_bytes(b"\x00\x00" * FRAME_BYTES)
                await asyncio.sleep(PACE)
                for t, kind, m in sess.inbox:
                    if (kind == "json" and m.get("type") == "barge"
                            and t >= t_start):
                        cut_at = t
                # Do NOT break on the cut: the VAD still needs the trailing
                # silence to CLOSE the span, which is what transcribes it and
                # fires the response turn (t_resp). Breaking here would starve
                # the VAD and the reply would never come.
            turn.t_speech_end = t_start
            if cut_at:
                turn.t_first_audio = cut_at
                # Full-Duplex-Bench v1.5 t_resp: cut -> first audio of the
                # response turn (the agent getting back to speaking).
                resp_at = None
                dl = time.monotonic() + 15
                while time.monotonic() < dl and resp_at is None:
                    for t, kind, m in sess.inbox:
                        if kind == "audio" and t >= cut_at:
                            resp_at = t
                    await asyncio.sleep(0.01)
                turn.t_resp_audio = resp_at
            turn.error = None if cut_at else "no barge"
            runs_out.append(turn.record(bool(cut_at)))
            t_resp = (turn.t_resp_audio - cut_at) * 1000 if cut_at and turn.t_resp_audio else None
            print(f"  [barge_cut #{i + 1}] t_stop={turn.client_ttfa and round(turn.client_ttfa)}ms "
                  f"t_resp={t_resp and round(t_resp)}ms"
                  f"{' !! ' + str(turn.error) if turn.error else ''}")
            await sess.close()
        results["scenarios"]["barge_cut"] = runs_out

    # --- summary -----------------------------------------------------------
    print("\n=== summary (ms; median / p95) ===")
    print(f"{'scenario':<14}{'n':>3}{'client_ttfa':>20}{'client_total':>20}"
          f"{'t_resp':>14}{'asr':>14}{'llm':>14}{'tts':>12}{'ttfa_srv':>14}")
    summary = {}
    for name, runs_out in results["scenarios"].items():
        oks = [r for r in runs_out if r.get("ok") and r.get("client_ttfa")]
        row = {"n": len(oks)}
        summary[name] = row
        if not oks:
            print(f"{name:<14}{0:>3}  (all runs failed)")
            continue
        stats_key = {"asr": "asr_ms", "llm": "llm_ms",
                     "tts": "tts_ms", "ttfa_srv": "ttfa_ms"}
        for key in ("client_ttfa", "client_total", "t_resp", "asr", "llm",
                    "tts", "ttfa_srv"):
            if key in stats_key:
                vals = [r["stats"][stats_key[key]] for r in oks
                        if r.get("stats", {}).get(stats_key[key]) is not None]
            else:
                vals = [r[key] for r in oks if r.get(key) is not None]
            if not vals:
                continue
            row[key] = {"p50": round(pct(vals, 50)), "p95": round(pct(vals, 95)),
                        "min": round(min(vals)), "max": round(max(vals))}
        line = f"{name:<14}{len(oks):>3}"
        for k in ("client_ttfa", "client_total", "t_resp", "asr", "llm",
                  "tts", "ttfa_srv"):
            r_ = row.get(k, {})
            p50, p95 = r_.get("p50", "-"), r_.get("p95", "-")
            if k in ("client_ttfa", "client_total"):
                line += f"{p50:>11}/{p95!s:>7}"
            else:
                line += f"{p50:>14}"
        print(line)

    results["summary"] = summary
    RESULTS.mkdir(exist_ok=True)
    tag = label or "run"
    out = RESULTS / f"latency_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lava latency benchmark")
    ap.add_argument("--runs", type=int, default=3, help="runs per scenario")
    ap.add_argument("--port", type=int, default=8080, help="bridge port")
    ap.add_argument("--label", default="", help="tag for the results file (e.g. your name)")
    args = ap.parse_args()
    asyncio.run(run_benchmark(args.runs, args.port, args.label))
