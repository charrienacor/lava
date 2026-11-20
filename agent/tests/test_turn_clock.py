"""turn_clock: the honest TTFA milestones.

The clock this module replaced armed once at the first VAD stop and never
re-armed, so a stray noise stop anchored the whole measurement — logs
showed "endpointing+ASR 120583 ms" for turns whose real wait was ~2 s.
These tests pin the re-arm semantics the tuning depends on.
"""

from turn_clock import TurnClock


def test_stop_rearms_and_drops_stale_asr():
    """A second stop moves the anchor: the wait is measured from the stop
    that commits the turn, not from a stray earlier one. The stale split
    from segment 1 is dropped — the next transcript re-stamps it from the
    LATER anchor."""
    import time

    c = TurnClock()
    c.vad_stop()          # mid-thought (or noise) stop
    time.sleep(0.05)
    c.transcript()        # its transcript — segment did not commit
    c.vad_stop()          # user continued and stopped again: re-armed
    c.transcript()        # the committing segment's transcript
    ttfa, asr, llm = c.first_audio()
    assert asr is not None and asr < 20.0  # from stop2, not the 50 ms old stop1
    assert llm is not None and 0.0 <= llm <= ttfa


def test_stop_after_transcript_races_clears_split():
    """The STT background task can race the stop frame; the stop re-arms
    anyway (honest TTFA, mislabeled split) — a few-ms cosmetic cost next
    to the huge inflated splits the single-anchor clock produced."""
    c = TurnClock()
    c.transcript()        # raced ahead of its own stop frame
    c.vad_stop()          # the stop lands microseconds later
    ttfa, asr, llm = c.first_audio()
    assert ttfa >= 0.0
    assert asr is None and llm is None  # logged as "(no transcript)"


def test_discard_rewinds():
    c = TurnClock()
    c.vad_stop()
    assert c.pending()
    c.discard()            # noise / non-owner / error: no reply is coming
    assert not c.pending()
    assert c.first_audio() is None


def test_transcript_self_anchors_when_first():
    """The STT background task can race the stop frame; a transcript
    arriving first anchors the clock itself (asr split ~0)."""
    c = TurnClock()
    c.transcript()
    ttfa, asr, llm = c.first_audio()
    assert asr is not None and asr < 50.0
    assert llm is not None


def test_typed_turn_has_no_asr_split():
    """Tray chips anchor at the typed request and publish without an ASR
    split — the old clock left them anchored at some stale VAD stop."""
    c = TurnClock()
    c.vad_stop()          # stale anchor from before the chip
    c.typed()
    ttfa, asr, llm = c.first_audio()
    assert ttfa >= 0.0
    assert asr is None and llm is None


def test_first_audio_publishes_once_then_rearms():
    c = TurnClock()
    c.vad_stop()
    c.transcript()
    assert c.first_audio() is not None
    assert c.first_audio() is None  # published and rewound
    c.vad_stop()                    # next turn arms fresh
    assert c.pending()


def test_split_clamped_at_zero():
    """A transcript that raced the stop must not print a negative share."""
    c = TurnClock()
    c.transcript()        # stamps asr ~0 against a self-anchored t0
    c.vad_stop()          # stop frame lands microseconds later
    c.transcript()        # ...and a second transcript re-stamps the split
    _, asr, llm = c.first_audio()
    assert llm is not None and llm >= 0.0
