"""pi_client echo suppression: her reply in the air must darken the mic.

Over Bluetooth the webrtc AEC cannot converge (measured ~5 dB), so her own
voice came back through the VAD: self-barge-in chopped her speech gappy,
Groq uploaded the echo, and the speaker gate toasted on every segment.
The mic hold (ECHO_SUPPRESS, default on) keeps the Pi path usable until
her output is wired.
"""

import sys
import time
from pathlib import Path

# ui/ sits at the repo root; agent/ is already on sys.path (conftest).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.pi_client as pc  # noqa: E402


def _fresh_speaker():
    return pc.Speaker()


def test_mic_held_while_she_speaks():
    spk = _fresh_speaker()
    assert spk.holds_mic(0.0) is False  # nothing has played yet
    spk._last_feed = time.monotonic()  # a chunk just arrived
    # her echo peaks ~0.025 at the mic — far under the line: stays dark
    assert spk.holds_mic(0.05) is True  # quiet frame within the tail: dark
    spk._last_feed = time.monotonic() - pc.SUPPRESS_TAIL_SECS - 0.1
    assert spk.holds_mic(0.05) is False  # tail elapsed: mic live again


def test_loud_syllable_opens_the_mic_and_hysteresis_keeps_it_open():
    """Full duplex preserved: one loud frame (a syllable peak) opens the
    mic for the rolling hold window, and the QUIET frames inside the
    utterance flow too — per-frame gating would chop speech below the
    line and the VAD would never accumulate its start."""
    spk = _fresh_speaker()
    spk._last_feed = time.monotonic()  # she is speaking right now
    loud = pc.ECHO_BREAKIN_LEVEL + 0.05
    assert spk.holds_mic(loud) is False  # the syllable breaks through
    assert spk.holds_mic(0.02) is False  # hysteresis: quiet frame flows
    assert spk.holds_mic(0.02) is False  # ...and the next one, uninterrupted
    # the window lapses with no further loud frames → echo-dark again
    spk._mic_open_until = time.monotonic() - 0.1
    assert spk.holds_mic(0.05) is True  # back to dark while she plays


def test_env_off_restores_full_duplex(monkeypatch):
    monkeypatch.setattr(pc, "ECHO_SUPPRESS", False)
    spk = _fresh_speaker()
    spk._last_feed = time.monotonic()  # audio playing right now
    assert spk.holds_mic(0.0) is False  # ECHO_SUPPRESS=0: never held
