"""Per-turn time-to-first-audio milestones — the honest clock.

TTFA is the number the user experiences: from the end of their speech to
the first reply audio. The ui/server.py tap publishes it per turn with a
stage split (endpointing+ASR vs LLM+TTS), and that log line is what the
Pi is tuned from — so the milestones must measure the WAIT, not whatever
happened to anchor the clock.

The clock this replaces armed once at the FIRST VAD stop and never
re-armed; a stray noise stop (TV, cough, the bot's own spill off a raw
mic) then anchored the whole measurement while the user went on talking —
logs showed "endpointing+ASR 120583 ms" for turns whose real wait was
~2 s. Re-arming on every stop fixes that: each VAD stop is a *candidate*
turn end, and when SmartTurn says INCOMPLETE the user keeps talking and
later stops move the anchor forward, so the published split starts at the
stop that actually committed the turn.

Milestones (all O(1), no locks — the tap calls them from one loop):

    vad_stop()    user's speech segment ended — RE-arm the clock and drop
                  any stale ASR split (a mid-thought pause discarded it)
    transcript()  final transcript in hand — stamps the endpointing+ASR
                  split; self-anchors if it races ahead of the stop frame
                  (ASR runs on a background task)
    typed()       a tray chip became a user turn — no VAD/ASR will fire,
                  so anchor at the typed request itself
    discard()     the segment produced nothing (noise, non-owner, error)
    first_audio() first reply audio is out — publish the split once and
                  re-arm for the next turn; None when nothing is pending

The split is clamped at zero: a transcript racing the stop frame can land
before it, which would otherwise print a negative "LLM+TTS" share.
"""

import time


class TurnClock:
    """TTFA milestones for one session turn (see the module docstring)."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._asr_ms: float | None = None

    def pending(self) -> bool:
        """A user turn is in flight (clock armed, no audio published yet)."""
        return self._t0 is not None

    def vad_stop(self) -> None:
        """User's speech just ended: re-arm, dropping any stale ASR split."""
        self._t0 = time.monotonic()
        self._asr_ms = None

    def transcript(self) -> None:
        """Final transcript in hand: stamp the endpointing+ASR split."""
        if self._t0 is None:  # raced ahead of the stop frame
            self._t0 = time.monotonic()
        self._asr_ms = (time.monotonic() - self._t0) * 1000.0

    def typed(self) -> None:
        """A tray chip became the user turn: anchor at the typed request."""
        self._t0 = time.monotonic()
        self._asr_ms = None

    def discard(self) -> None:
        """Turn ended with no reply coming (noise, non-owner, error)."""
        self._t0 = None
        self._asr_ms = None

    def first_audio(self) -> tuple[float, float | None, float | None] | None:
        """First reply audio: (ttfa_ms, asr_ms, llm_tts_ms), then re-arm.

        asr_ms / llm_tts_ms are None when no transcript landed for this
        turn (a typed tray chip). Returns None when no turn is pending.
        """
        if self._t0 is None:
            return None
        ttfa_ms = (time.monotonic() - self._t0) * 1000.0
        asr_ms = self._asr_ms
        llm_ms = max(ttfa_ms - asr_ms, 0.0) if asr_ms is not None else None
        self.discard()
        return ttfa_ms, asr_ms, llm_ms
