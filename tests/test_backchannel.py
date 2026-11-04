"""Unit tests for the deferred barge-in backchannel classifier (web/server.py).

Run:  .venv/bin/python tests/test_backchannel.py
Imports web.server and exercises is_backchannel() — the Full-Duplex-Bench
(R)ESCUE the-floor logic: pure acknowledgments keep the bot talking, real
speech cuts.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import web.server as ws  # noqa: E402

POSITIVE = [
    "uh-huh", "uh huh", "mm-hmm", "mhm", "yeah", "yep", "right", "sure",
    "okay", "ok", "cool", "nice", "great", "got it", "gotcha", "understood",
    "that's right", "i see", "sounds good", "for sure", "oh okay",
    "Yeah, yeah.", "MM-HMM", "yes",
]
NEGATIVE = [
    # real interruptions / commands
    "wait, can you skip this song", "stop the music", "next", "louder",
    "no wait stop", "what time is it", "play Bohemian Rhapsody",
    # longer than BACKCHANNEL_MAX_WORDS
    "yes okay right sure cool great understood",
    # gibberish / unknown words
    "humina humina",
    "",
]


def main() -> int:
    fails = 0
    for text in POSITIVE:
        if not ws.is_backchannel(text):
            fails += 1
            print(f"FAIL  backchannel missed: {text!r}")
    for text in NEGATIVE:
        if ws.is_backchannel(text):
            fails += 1
            print(f"FAIL  false backchannel: {text!r}")
    n = len(POSITIVE) + len(NEGATIVE)
    print(f"{n - fails}/{n} cases passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
