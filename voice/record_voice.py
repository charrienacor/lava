"""Record voice samples from the mic — for ASR speaker enrollment.

Saves 16 kHz mono WAV clips (the same format the ASR pipeline sends) with a
live input-level meter so you can see the mic is picking you up. Then feed
the clips to the enrollment script:

    .venv/bin/python voice/record_voice.py --samples 3
    .venv/bin/python voice/enroll_voice.py data/recordings/voice_*.wav

Mic access: on macOS, allow your terminal microphone access when prompted
(System Settings → Privacy & Security → Microphone). On the RPi, install
PortAudio first: sudo apt install libportaudio2
"""

import argparse
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = ROOT / "data" / "recordings"

SAMPLE_RATE = 16000
CHANNELS = 1


def _ensure_input_device() -> None:
    try:
        sd.query_devices(kind="input")
    except Exception as e:
        sys.exit(
            f"No microphone available ({e}).\n"
            "On macOS, allow microphone access for your terminal; "
            "on the RPi: sudo apt install libportaudio2"
        )


def record_clip(seconds: float, out_path: Path) -> float:
    """Record one clip with a live level meter; returns the peak level."""
    frames: list[np.ndarray] = []
    peak = 0.0
    started = time.monotonic()
    last_draw = 0.0

    def callback(indata, _frames, _time, _status):
        nonlocal peak, last_draw
        frames.append(indata.copy())
        level = float(np.max(np.abs(indata))) / 32768.0
        peak = max(peak, level)
        now = time.monotonic()
        if now - last_draw < 0.1:  # redraw ~10x/s, not per audio block
            return
        last_draw = now
        elapsed = now - started
        bar = "#" * int(level * 30)
        print(
            f"\r  [{bar:<30}] {elapsed:5.1f}/{seconds:.0f}s  peak {peak:4.0%} ",
            end="",
            flush=True,
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=callback
    ):
        time.sleep(seconds)
    print()  # newline after the meter

    audio = np.concatenate(frames) if frames else np.zeros((0, CHANNELS), np.int16)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record mic clips for ASR speaker enrollment."
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="number of clips (default 3)"
    )
    parser.add_argument(
        "--seconds", type=float, default=8.0, help="seconds per clip (default 8)"
    )
    parser.add_argument(
        "--out", type=Path, help="output path for a single clip (default data/recordings/)"
    )
    args = parser.parse_args()

    _ensure_input_device()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    saved: list[Path] = []
    for i in range(args.samples):
        if i:
            input("  Press Enter for the next sample...")
        out = (
            args.out
            if args.out and args.samples == 1
            else RECORDINGS_DIR / f"voice_{stamp}_{i + 1}.wav"
        )
        print(f"Sample {i + 1}/{args.samples} — speak naturally for {args.seconds:.0f} s:")
        peak = record_clip(args.seconds, out)
        size_kb = out.stat().st_size / 1024
        print(f"  saved {out} ({size_kb:.0f} KB, peak {peak:.0%})")
        if peak < 0.02:
            print("  warning: input was almost silent — move closer to the mic")
        saved.append(out)

    print("\nNext — enroll with these samples:")
    paths = []
    for p in saved:
        try:
            paths.append(str(p.relative_to(ROOT)))
        except ValueError:  # clip saved outside the project — show the full path
            paths.append(str(p))
    print("  .venv/bin/python voice/enroll_voice.py " + " ".join(paths))


if __name__ == "__main__":
    main()
