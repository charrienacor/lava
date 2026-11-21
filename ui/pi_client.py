"""Lava with the Pi's own microphone and speaker — and the UI on another device.

Runs NEXT TO the server (usually on the Pi itself) and speaks the same /ws
voice protocol the browser page speaks (ui/server.py module docstring):

    this → server   binary: raw 16 kHz mono 16-bit LE PCM (Pi mic frames)
                    text:   {"type": "ready"}   Pi mic is live — VAD/turns on
    server → this   binary: [uint16 LE rate][16-bit LE mono PCM] → Pi speaker
                    text:   JSON events — printed as a terminal transcript
                    (and mirrored to every ?view=1 browser, where the UI lives)

So the session is owned HERE, with the hardware mic, while the pretty orb UI
opens on the laptop as a spectator:

    .venv/bin/python ui/server.py            # terminal 1 (Pi)
    .venv/bin/python ui/pi_client.py         # terminal 2 (Pi) — the voice
    # laptop browser:  http://<pi-ip>:8000/?view=1

Run:
    .venv/bin/python ui/pi_client.py                    # default devices
    .venv/bin/python ui/pi_client.py --list-devices     # find mic/speaker ids
    .venv/bin/python ui/pi_client.py --mic 2 --spk 1    # pick them
    .venv/bin/python ui/pi_client.py --url ws://<pi-ip>:8000   # client ≠ server host

Echo handling: the AEC twins (lava_aec) cancel her voice at the source when
it can converge — wired output does. Over Bluetooth it cannot (measured
~5 dB), so pi_client adds an energy-gated hold (ECHO_SUPPRESS=1, the
default): her echo never reaches the VAD, while a frame loud enough to be
you — not her echo — breaks through and barge-in fires mid-reply. The Pi
path stays full-duplex; speaking up interrupts her. Tune the line with
ECHO_BREAKIN_LEVEL, or set ECHO_SUPPRESS=0 when her output is wired and
the canceller converges.
"""

import argparse
import asyncio
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is missing — Lava's venv has it: run this with .venv/bin/python")

try:
    import websockets
except ImportError:
    sys.exit("websockets is missing — .venv/bin/python -m pip install websockets")

# The page batches 512-sample (32 ms) mic frames — match it: the server's
# Silero VAD consumes 512-sample frames, so this costs zero end-to-end latency.
MIC_RATE = 16000
MIC_BLOCK = 512


# Echo suppression (the raw-mic hardening): while reply audio is playing —
# and for a short tail after the last chunk — the mic does not stream. The
# AEC cannot converge over Bluetooth (measured ~5 dB cancellation), so her
# own voice otherwise comes back in the mic: the VAD fires on HER, barge-in
# cuts her speech (the gappy/choppy playback), Groq uploads the echo, and
# the speaker gate toasts "Not your voice" on every echo segment.
#
# But this stays FULL DUPLEX: a frame loud enough to be YOU — not her echo
# — breaks straight through the hold and barge-in fires mid-reply, exactly
# as the design promises. Measured anchors (post-AEC, through lava_aec):
# her echo peaks ~0.025 at normal volume; your speech peaks 0.1–1.0 —
# 30-40x apart, so the line sits at 0.08. Hysteresis keeps the gate open
# for ECHO_HOLD_SECS after any loud frame: one syllable opens the mic and
# the whole utterance flows continuously (per-frame gating would chop
# speech below the line and the VAD would never accumulate its start).
# Tune via ECHO_BREAKIN_LEVEL; ECHO_SUPPRESS=0 removes the hold entirely
# once her output is wired and the AEC converges (20+ dB).
ECHO_SUPPRESS = os.environ.get("ECHO_SUPPRESS", "1") not in ("0", "false", "no")
SUPPRESS_TAIL_SECS = float(os.environ.get("ECHO_SUPPRESS_TAIL_SECS", "0.8"))
ECHO_BREAKIN_LEVEL = float(os.environ.get("ECHO_BREAKIN_LEVEL", "0.08"))
ECHO_HOLD_SECS = float(os.environ.get("ECHO_HOLD_SECS", "2.0"))


def _peak(indata) -> float:
    """Peak level (0..1) of one raw int16 mic frame."""
    if not len(indata):
        return 0.0
    return float(np.abs(np.frombuffer(indata, dtype=np.int16)).max()) / 32768.0


def http_of(url: str) -> str:
    """The /api/greet REST twin of a ws:// URL."""
    return url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)


def fire_greeting(url: str, cid: str) -> None:
    """POST /api/greet — the instant warm-cache greeting, as the page does
    the moment its mic is live. Best-effort: without it the pipeline still
    greets on ready, one beat later."""
    try:
        req = urllib.request.Request(f"{http_of(url)}/api/greet?cid={cid}", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


class Speaker:
    """Plays the server's audio on the Pi's speaker.

    Chunks arrive as [uint16 LE rate][16-bit LE mono PCM] and the rate can
    change between replies (per voice), so the output stream is (re)opened on
    every rate change and audio accumulates in a locked byte buffer that the
    callback drains — silence when it runs dry (an underrun, not an error).
    """

    def __init__(self, device=None):
        self.device = device
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self._rate = None
        self._last_feed = 0.0  # echo suppression: monotonic time of last chunk
        self._mic_open_until = 0.0  # hysteresis: one loud frame opens the mic

    def feed(self, pcm: bytes, rate: int) -> None:
        if rate != self._rate or self._stream is None:
            self._reopen(rate)
        with self._lock:
            self._buf += pcm
        self._last_feed = time.monotonic()

    def holds_mic(self, peak: float) -> bool:
        """True while reply audio is playing or fresh — keep the mic dark so
        her own voice in the air cannot become VAD activity (see the
        ECHO_SUPPRESS note at the top of this file).

        FULL DUPLEX is preserved by hysteresis, not per-frame gating: one
        frame loud enough to be you (peak ≥ ECHO_BREAKIN_LEVEL) opens the
        mic for a rolling ECHO_HOLD_SECS window, so the whole utterance
        flows continuously — the VAD gets unbroken audio and barge-in
        fires mid-reply. Her echo (peak ~0.025) never opens it; your
        speech (peaks 0.1–1.0) opens it on the first loud syllable."""
        if not ECHO_SUPPRESS:
            return False
        now = time.monotonic()
        if peak >= ECHO_BREAKIN_LEVEL:
            self._mic_open_until = now + ECHO_HOLD_SECS
            return False  # you, speaking: frame flows, gate stays open
        if now < self._mic_open_until:
            return False  # hysteresis: your utterance continues, uninterrupted
        return (now - self._last_feed) < SUPPRESS_TAIL_SECS

    def _reopen(self, rate: int) -> None:
        with self._lock:
            self._buf.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._stream = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", device=self.device, callback=self._cb
        )
        self._stream.start()
        self._rate = rate

    def _cb(self, outdata, frames, time_info, status):  # noqa: ANN001 (sounddevice signature)
        need = frames * 2  # int16 mono → 2 bytes per frame
        with self._lock:
            take = bytes(self._buf[:need])
            del self._buf[:need]
        if len(take) < need:
            take += b"\x00" * (need - len(take))
        outdata[:] = take

    def flush(self) -> None:
        """Barge-in: cut whatever is queued/playing right now."""
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class TranscriptPrinter:
    """The terminal twin of the page's chat bubbles."""

    def __init__(self):
        self._reply: list[str] = []

    def event(self, m: dict) -> None:
        kind = m.get("type")
        if kind == "transcript":
            self._reply.clear()
            print(f"\n  you  ▸ {m.get('text', '')}")
        elif kind == "text":
            self._reply.append(m.get("t", ""))
            print(f" lava  ▸ {''.join(self._reply)}", end="\r", flush=True)
        elif kind == "done":
            reply = "".join(self._reply)
            self._reply.clear()
            if reply:
                print(f" lava  ▸ {reply}")
            else:
                print(" lava  ▸ (tool turn)")
        elif kind == "error":
            # A run died server-side (e.g. a malformed streamed tool call).
            # Silence here reads as "she ignored me" — say it happened.
            print(" lava  ▸ (something went wrong on my side — ask me again)")
        elif kind == "bye":
            print("· session timed out — still watching for the next one")


async def recv_loop(ws, speaker: Speaker, printer: TranscriptPrinter) -> None:
    """Server → Pi: audio to the speaker, events to the terminal."""
    async for raw in ws:
        if isinstance(raw, bytes):
            if len(raw) < 2:
                continue  # not a [rate][pcm] frame — ignore
            rate = struct.unpack("<H", raw[:2])[0]
            speaker.feed(raw[2:], rate)
            continue
        try:
            m = json.loads(raw)
        except (ValueError, TypeError):
            continue
        kind = m.get("type")
        if kind == "barge":
            speaker.flush()  # hard-cut playback, as the page's stopBotAudio
        elif kind == "busy":
            sys.exit("cid already in a session — pick another with --cid")
        else:
            printer.event(m)


async def mic_loop(ws, device, speaker: Speaker) -> None:
    """Pi mic → server: 512-sample int16 frames from a sounddevice callback,
    ferried to the websocket by an asyncio queue (the callback must never
    block, so drops — not stalls — are the overload policy).

    Echo suppression: while the speaker holds the mic (her reply playing
    or fresh in the air), frames are dropped here — her echo never reaches
    the server's VAD at all."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        if speaker.holds_mic(_peak(indata)):
            return  # her voice is in the air and this frame isn't you
        data = bytes(indata)

        def put():
            if q.full():
                return  # overload: drop the frame — never stall the mic thread
            q.put_nowait(data)

        try:
            loop.call_soon_threadsafe(put)
        except RuntimeError:
            pass  # event loop gone — shutting down

    stream = sd.RawInputStream(
        samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=MIC_BLOCK,
        device=device, callback=cb,
    )
    stream.start()
    try:
        while True:
            data = await q.get()
            await ws.send(data)
    finally:
        stream.stop()
        stream.close()


async def run(url: str, cid: str, mic, spk) -> None:
    speaker = Speaker(spk)
    printer = TranscriptPrinter()
    ws_url = f"{url.rstrip('/')}/ws?cid={cid}"
    print(f"· connecting to {ws_url} (mic 16 kHz, cid {cid})")
    async with websockets.connect(ws_url, max_size=None) as ws:
        # Mic is live the moment we connect — the page's two signals, both sent.
        await ws.send(json.dumps({"type": "ready"}))
        fire_greeting(url, cid)
        print("· connected — talk to Lava (Ctrl+C to quit). "
              "Spectators: http://<this-host>:8000/?view=1")
        mode = (
            f"on — echo dark, barge-in at peak ≥ {ECHO_BREAKIN_LEVEL:.2f} "
            f"(holds {ECHO_HOLD_SECS:.0f}s, +{SUPPRESS_TAIL_SECS:.1f}s tail)"
            if ECHO_SUPPRESS else
            "off — full duplex, no hold (ECHO_SUPPRESS=0)"
        )
        print(f"· echo suppression: {mode}")
        recv = asyncio.create_task(recv_loop(ws, speaker, printer))
        mic_task = asyncio.create_task(mic_loop(ws, mic, speaker))
        try:
            # Either end finishes the call: the server closing the session, or
            # a local failure (mic device gone). The survivor is cancelled —
            # nothing hangs on a half-dead call.
            done, _ = await asyncio.wait({recv, mic_task}, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                if t.exception() is not None:
                    raise t.exception()
        finally:
            mic_task.cancel()
            recv.cancel()
    speaker.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Lava on the Pi's own mic and speaker")
    ap.add_argument("--url", default="ws://127.0.0.1:8000", help="ui/server.py address")
    ap.add_argument("--cid", default="pi", help="session id (the page's ?cid twin)")
    ap.add_argument("--mic", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--spk", type=int, default=None, help="output device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="print sounddevice's device table and exit")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    u = urlparse(args.url)
    host = u.hostname or "127.0.0.1"
    shown = f"http://{host}:8000" if u.port == 8000 else http_of(args.url)
    try:
        asyncio.run(run(args.url, args.cid, args.mic, args.spk))
    except KeyboardInterrupt:
        print("\n· stopped")
    except (ConnectionRefusedError, OSError, websockets.exceptions.WebSocketException) as e:
        sys.exit(f"· could not reach {args.url} ({e}) — is ui/server.py running? (viewer: {shown}/?view=1)")


if __name__ == "__main__":
    main()
