"""Voice activity + turn detection — the front end of the voice loop.

Two cooperating detectors:

1. Silero VAD (on-device, 16 kHz mono): is someone speaking right now?
   Its timers slice the audio into speech segments; RNNoise cleans the
   audio first so it doesn't fire on music/TV.

2. SmartTurn v3.2 (pipecat.audio.turn.smart_turn, bundled 8.7 MB ONNX model):
   is the utterance FINISHED? A semantic end-of-turn detector — it reads the
   audio segment (whisper log-mel features) and classifies complete vs
   incomplete, so a mid-thought pause ("...um, the 2024 one") no longer
   commits a turn, and "yeah, go on" backchannels don't either.

Composition (pipecat 1.16 turns architecture): the VAD decides speech
boundaries; the SmartTurn stop strategy gates when the turn COMMITS to the
LLM. Because SmartTurn guards against premature ends, the VAD's stop window
can be tightened (0.3 → 0.2 s, pipecat's VAD_STOP_SECS which its safety-net
math assumes): real turns hand off faster, false ends are caught semantically.

SMART_TURN=0 falls back to the classic VAD-only behavior (longer stop window,
SpeechTimeout stop strategy) — no ML, lowest CPU.
"""

import os

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS, VADParams

# VAD tuning for a real room (other speakers, music): higher confidence and a
# longer onset confirm window than pipecat's defaults (0.7 / 0.2) cut false
# triggers. The stop window is the one parameter SmartTurn lets us tighten.
_VAD_CONFIDENCE = 0.8
_VAD_START_SECS = 0.25
_VAD_STOP_SECS_VAD_ONLY = 0.3  # old conservative window: no semantic guard

# SmartTurn's own hard fallback: if the user stays silent this long after
# speech, the turn completes WITHOUT another model run. This is the tail of
# every "SmartTurn said INCOMPLETE" pause — at 3.0 s the logs showed turns
# sitting in dead air while the user waited for a verdict that never came.
# 1.5 s still clears a thinking pause ("...um"), and when SmartTurn's verdict
# says COMPLETE the turn commits immediately regardless of this timer. This
# is the article-style tradeoff: tighten to the smallest value that doesn't
# truncate natural pauses, then watch the barge/cut-over rate in the logs.
_SMART_TURN_STOP_SECS = 1.5


def smart_turn_enabled() -> bool:
    """Semantic end-of-turn is on unless SMART_TURN=0 in the environment."""
    return os.environ.get("SMART_TURN", "1") not in ("0", "false", "no")


def create_vad(smart_turn: bool | None = None) -> SileroVADAnalyzer:
    """Silero VAD analyzer; runs on onnxruntime, no extra download.

    With SmartTurn active (the default) the stop window matches pipecat's
    VAD_STOP_SECS: the turn detector catches premature ends, so the VAD
    doesn't have to stay silent-proof. With SMART_TURN=0 the old, longer
    window returns — silence alone must then decide, so it needs the margin.
    """
    smart = smart_turn_enabled() if smart_turn is None else smart_turn
    stop_secs = VAD_STOP_SECS if smart else _VAD_STOP_SECS_VAD_ONLY
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=_VAD_CONFIDENCE,
            start_secs=_VAD_START_SECS,
            stop_secs=stop_secs,
        )
    )


def create_smart_turn() -> LocalSmartTurnAnalyzerV3:
    """Semantic end-of-turn analyzer: bundled smart-turn-v3.2 ONNX, no download.

    stop_secs is the analyzer's hard silence fallback (see above); pre_speech
    and max_duration keep the analyzed segment bounded on long utterances.
    The sample rate is set by the turn strategy during pipeline setup.
    """
    return LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=_SMART_TURN_STOP_SECS,
            pre_speech_ms=500,
            max_duration_secs=8,
        )
    )


def log_front_end_mode() -> None:
    """One startup line describing the active front end."""
    if smart_turn_enabled():
        logger.info("Front end: RNNoise → Silero VAD → SmartTurn v3.2 (semantic end-of-turn)")
    else:
        logger.info("Front end: RNNoise → Silero VAD (timers only; SMART_TURN=0)")
