"""Conversation context — system prompt + MCP tools + turn strategies.

User-turn end decision (pipecat 1.16 turns architecture):

    SMART_TURN=1 (default)  → semantic end-of-turn: SmartTurn v3.2 (bundled
                              8.7 MB ONNX) classifies the speech segment as
                              complete/incomplete; the turn commits to the LLM
                              only when the verdict says complete AND the final
                              transcript is in. VAD stop window: pipecat's
                              VAD_STOP_SECS (0.2 s) — SmartTurn guards against
                              premature ends, so the window can be tight.
    SMART_TURN=0            → classic timers: SpeechTimeout stop strategy
                              (commit shortly after the VAD stop window and
                              the first transcript), longer VAD window
                              (0.3 s) — silence alone decides.

Auto-summarization: compresses old turns into a summary instead of letting
the context grow into the TPM ceiling (Groq free tier: 8k tokens/minute).
"""

from loguru import logger
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
)

import time

import vad

# Auto-summarization thresholds sized for the Groq free tier's 8k TPM: the
# context (system prompt + tool schemas excluded from this estimate) must stay
# small enough that 2–3 requests landing in the same minute stay under quota.
# Tighter thresholds → more summarizer calls — each one is an extra request,
# so don't tighten past the point where compression itself pays the TPM.
SUMMARIZATION_CONFIG = LLMAutoContextSummarizationConfig(
    max_context_tokens=4000,
    max_unsummarized_messages=12,
)

# The web UI's summarize toast reads this: the on_summary_applied handler
# below stamps it (monotonic time, messages before, messages after) each
# time the context compresses — polled by ui/server.py _watch_summary.
LAST_SUMMARY: tuple[float, int, int] | None = None


async def _on_summary_applied(_summarizer, event) -> None:
    """Stamp LAST_SUMMARY when pipecat's summarizer applies a summary.

    Registered in create_context on the assistant aggregator's summarizer
    (pipecat event convention: handlers receive (emitter, event)).
    """
    global LAST_SUMMARY
    LAST_SUMMARY = (
        time.monotonic(),
        event.original_message_count,
        event.new_message_count,
    )


def build_user_params() -> LLMUserAggregatorParams:
    """User aggregator params: VAD analyzer + the active turn-stop strategy.

    Strategy and VAD window are kept in lockstep: the semantic detector and
    the tight window belong together, as do the plain timers and the longer
    window (see vad.py). A SmartTurn load failure degrades gracefully to
    timers — the agent keeps working, just with the old turn behavior.
    """
    stop_strategy = None
    if vad.smart_turn_enabled():
        try:
            stop_strategy = TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=vad.create_smart_turn()
            )
        except Exception as e:
            logger.warning(f"SmartTurn unavailable ({e}) — falling back to VAD timers")
    if stop_strategy is not None:
        vad_analyzer = vad.create_vad(smart_turn=True)
        strategies = UserTurnStrategies(stop=[stop_strategy])
    else:
        vad_analyzer = vad.create_vad(smart_turn=False)
        strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy()]
        )
    return LLMUserAggregatorParams(
        vad_analyzer=vad_analyzer,
        user_turn_strategies=strategies,
    )


def create_context(tools: ToolsSchema):
    """Build the LLM context and its user/assistant aggregators.

    The context holds only the conversation: the system prompt lives on the
    LLM service as ``system_instruction`` (pipeline.py create_llm) — a
    "system" message here is deprecated since pipecat 1.9 and bypasses the
    service's instruction composition. The merged MCP tool schemas ride on
    the context; the turn front end (VAD + SmartTurn or timers) drives the
    user turn events. Long conversations self-compress via auto context
    summarization.
    """
    user_params = build_user_params()
    mode = "SmartTurn (semantic end-of-turn)" if vad.smart_turn_enabled() else "VAD timers"
    logger.info(f"Turn detection: {mode}")
    llm_context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        llm_context,
        user_params=user_params,
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True,
            auto_context_summarization_config=SUMMARIZATION_CONFIG,
        ),
    )
    # The pair registers its own summarizer handlers; this one stamps
    # LAST_SUMMARY so the web UI can toast when compression lands. The
    # summarizer is pipecat-internal (no public accessor in 1.16) —
    # tolerate its absence rather than break the pipeline for a toast.
    summarizer = getattr(assistant_aggregator, "_summarizer", None)
    if summarizer is not None:
        summarizer.add_event_handler("on_summary_applied", _on_summary_applied)
    return llm_context, user_aggregator, assistant_aggregator
