"""Turn front end: SmartTurn wiring, VAD tuning, ASR finalization."""

import asyncio

import pytest
from pipecat.audio.turn.smart_turn.base_smart_turn import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

import asr
import context
import vad


def test_smart_turn_enabled_by_default(monkeypatch):
    monkeypatch.delenv("SMART_TURN", raising=False)
    assert vad.smart_turn_enabled() is True


def test_smart_turn_env_off(monkeypatch):
    for off in ("0", "false", "no"):
        monkeypatch.setenv("SMART_TURN", off)
        assert vad.smart_turn_enabled() is False


def test_vad_tight_window_with_smart_turn():
    analyzer = vad.create_vad(smart_turn=True)
    assert analyzer.params.stop_secs == 0.2  # pipecat VAD_STOP_SECS
    assert analyzer.params.start_secs == 0.25  # room protection unchanged


def test_vad_long_window_vad_only():
    analyzer = vad.create_vad(smart_turn=False)
    assert analyzer.params.stop_secs == 0.3  # old conservative window


def test_smart_turn_model_loads_and_runs():
    """Real bundled ONNX model: loads in ms, runs inference on a segment."""
    analyzer = vad.create_smart_turn()
    analyzer.set_sample_rate(16000)
    for _ in range(20):  # ~1.2 s of speech-ish tone
        analyzer.append_audio(
            ((__import__("numpy").sin(__import__("numpy").arange(1600) * 0.05)) * 8000)
            .astype(__import__("numpy").int16)
            .tobytes(),
            True,
        )
    for _ in range(5):  # 0.5 s pause
        analyzer.append_audio(__import__("numpy").zeros(1600, __import__("numpy").int16).tobytes(), False)
    state, prediction = asyncio.run(analyzer.analyze_end_of_turn())
    assert isinstance(state, EndOfTurnState)
    assert prediction is not None  # the model actually produced a verdict


def test_user_params_semantic_mode():
    params = context.build_user_params()
    stop = params.user_turn_strategies.stop
    assert len(stop) == 1
    assert isinstance(stop[0], TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(stop[0]._turn_analyzer, LocalSmartTurnAnalyzerV3)
    assert params.vad_analyzer.params.stop_secs == 0.2


def test_user_params_fallback_mode(monkeypatch):
    monkeypatch.setenv("SMART_TURN", "0")
    params = context.build_user_params()
    stop = params.user_turn_strategies.stop
    assert len(stop) == 1
    assert isinstance(stop[0], SpeechTimeoutUserTurnStopStrategy)
    assert params.vad_analyzer.params.stop_secs == 0.3


def test_smart_turn_load_failure_falls_back(monkeypatch):
    monkeypatch.delenv("SMART_TURN", raising=False)
    monkeypatch.setattr(vad, "create_smart_turn", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    params = context.build_user_params()
    assert isinstance(params.user_turn_strategies.stop[0], SpeechTimeoutUserTurnStopStrategy)
    assert params.vad_analyzer.params.stop_secs == 0.3


def test_context_wires_both_modes():
    llm_context, user_agg, assistant_agg = context.create_context(tools=[])
    assert user_agg is not None and assistant_agg is not None


class _FakeMCP:
    pass


def _run_stt(monkeypatch, transcript):
    async def fake_call(*a, **k):
        return transcript

    monkeypatch.setattr(asr, "call_mcp_tool", fake_call)
    service = asr.McpWhisperSTTService(mcp=_FakeMCP())

    async def collect():
        return [f async for f in service.run_stt(b"fake-wav-bytes")]

    return asyncio.run(collect())


def test_asr_frames_are_finalized(monkeypatch):
    frames = _run_stt(monkeypatch, "hello there")
    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "hello there"
    assert frames[0].finalized is True


def test_asr_empty_transcript_yields_drop_frame(monkeypatch):
    # A VAD segment the owner spoke but Whisper heard nothing in must not
    # hang the UI: run_stt emits UtteranceDroppedFrame, which the page
    # turns into its "silence" event. (A gate REJECT is a different case:
    # the not-owner marker yields VoiceNotRecognizedFrame below.)
    frames = _run_stt(monkeypatch, "")
    assert len(frames) == 1
    assert isinstance(frames[0], asr.UtteranceDroppedFrame)
    assert not any(isinstance(f, TranscriptionFrame) for f in frames)


def test_asr_not_owner_yields_voice_rejected_frame(monkeypatch):
    # The gate's reject marker must NOT look like silence: run_stt emits
    # VoiceNotRecognizedFrame, which the server turns into the throttled
    # "Not your voice" toast — the user sees the gate working.
    frames = _run_stt(monkeypatch, asr.NOT_OWNER)
    assert len(frames) == 1
    assert isinstance(frames[0], asr.VoiceNotRecognizedFrame)
    assert not any(isinstance(f, TranscriptionFrame) for f in frames)
