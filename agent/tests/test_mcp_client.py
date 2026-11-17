"""mcp_client policy loading and registry construction (offline, no connects)."""

from mcp_client import build_mcp_clients, load_allowlist


def test_allowlists():
    gh = load_allowlist("github")
    assert len(gh) == 9
    assert "merge_pull_request" in gh and "delete_file" not in gh
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
