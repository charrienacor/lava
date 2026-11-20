"""mcp_client policy loading and registry construction (offline, no connects)."""

from mcp_client import build_mcp_clients, load_allowlist


def test_allowlists():
    gh = load_allowlist("github")
    assert len(gh) == 10
    assert "merge_pull_request" in gh and "delete_file" in gh
    assert load_allowlist("tts") == ["speak"]
    assert load_allowlist("time") == ["convert_time", "get_current_time"]


def test_allowlist_missing_server_is_empty():
    assert load_allowlist("no_such_server") == []


def test_build_mcp_clients_offline():
    clients = build_mcp_clients()
    assert set(clients) == {"asr", "tts", "github", "spotify", "tavily", "time"}
    # Construction must not connect (no network here).
    for client in clients.values():
        assert client is not None


def test_stdio_servers_receive_their_tuning_env(monkeypatch):
    """ASR_*/TTS_* knobs reach their own stdio server (the SDK's sanitized
    env dropped them) — and never leak to a different server."""
    monkeypatch.setenv("ASR_SPEAKER_MIN_SIM", "0.7")
    monkeypatch.setenv("TTS_DEBUG_WAV", "1")
    clients = build_mcp_clients()
    asr_env = clients["asr"]._server_params.env
    tts_env = clients["tts"]._server_params.env
    assert asr_env["ASR_SPEAKER_MIN_SIM"] == "0.7"
    assert "TTS_DEBUG_WAV" not in asr_env
    assert tts_env["TTS_DEBUG_WAV"] == "1"


def test_tool_error_result_raises():
    """An is_error result (a tool that raised server-side) must raise, not
    come back as text — a Groq failure used to become the user's transcript."""
    import asyncio

    import pytest
    from mcp.types import CallToolResult, TextContent

    import mcp_client

    class _Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(
                content=[TextContent(type="text", text="rate limited")], is_error=True
            )

    class _Client:
        async def _open_session(self, as_owner):
            return _Session()

    with pytest.raises(mcp_client.McpToolError, match="rate limited"):
        asyncio.run(mcp_client.call_mcp_tool(_Client(), "transcribe", {}, server="asr"))
    assert mcp_client.HEALTH["asr"] is True  # reachable: the TOOL failed
