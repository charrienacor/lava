"""context: the summarize stamp the web UI's toast watcher reads."""

import asyncio

from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema

import context as context_mod


def test_summary_stamp_records_compression(monkeypatch):
    """The on_summary_applied handler stamps LAST_SUMMARY with the
    compression metrics — (time, messages before, messages after) —
    which ui/server.py _watch_summary polls for its toast."""
    monkeypatch.setattr(context_mod, "LAST_SUMMARY", None)

    class _Event:
        original_message_count = 34
        new_message_count = 6
        summarized_message_count = 28
        preserved_message_count = 5

    asyncio.run(context_mod._on_summary_applied(None, _Event()))

    stamp = context_mod.LAST_SUMMARY
    assert stamp is not None
    assert stamp[1] == 34 and stamp[2] == 6


def test_create_context_wires_the_summarize_handler(monkeypatch):
    """create_context registers the stamp handler on the pair's summarizer,
    so a real SummaryAppliedEvent reaches _on_summary_applied."""
    monkeypatch.setattr(context_mod, "LAST_SUMMARY", None)
    monkeypatch.setenv("SMART_TURN", "0")  # timers: no ONNX model in tests

    llm_context, _, assistant = context_mod.create_context(tools=ToolsSchema())

    summarizer = getattr(assistant, "_summarizer", None)
    assert summarizer is not None, "pipecat 1.16 always builds the summarizer"

    class _Event:
        original_message_count = 12
        new_message_count = 3
        summarized_message_count = 9
        preserved_message_count = 2

    asyncio.run(
        summarizer._call_event_handler("on_summary_applied", _Event())
    )
    assert context_mod.LAST_SUMMARY is not None
    assert context_mod.LAST_SUMMARY[1] == 12
    assert context_mod.LAST_SUMMARY[2] == 3
