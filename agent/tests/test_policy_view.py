"""policy_view + ui/server: the tray payload is live from the TOMLs."""

import importlib.util
import json
from pathlib import Path

import pytest

import policy_view


def test_snapshot_from_real_tomls():
    snap = policy_view.snapshot()
    names = [s["name"] for s in snap["servers"]]
    assert names == ["asr", "tts", "github", "spotify", "tavily", "time"]
    by = {s["name"]: s for s in snap["servers"]}
    # kinds from mcp_servers.json, roles from the bridge set
    assert by["github"]["kind"] == "remote" and by["github"]["role"] == "capability"
    assert by["asr"]["role"] == "bridge"
    # real policy facts: the confirm gate rides the enabled set
    for tool in ("spotify_add_tracks", "spotify_create_playlist"):
        assert tool in by["spotify"]["enabled"]
        assert tool in by["spotify"]["confirm"]
    # every tool entry is consistent with the aggregate fields
    for s in snap["servers"]:
        assert s["tools_total"] == len(s["tools"])
        assert s["enabled"] == [t["name"] for t in s["tools"] if t["enabled"]]


def test_toml_change_is_visible_without_restart(tmp_path, monkeypatch):
    servers = tmp_path / "mcp_servers.json"
    servers.write_text(json.dumps({"mcpServers": {"time": {"command": "uvx"}}}))
    tools = tmp_path / "tools"
    tools.mkdir()
    monkeypatch.setattr(policy_view, "SERVERS_JSON", servers)
    monkeypatch.setattr(policy_view, "TOOLS_DIR", tools)

    toml = tools / "time.toml"
    toml.write_text('[[tool]]\nname = "get_current_time"\nenabled = false\n')
    snap = policy_view.snapshot()
    assert snap["servers"][0]["enabled"] == []

    toml.write_text('[[tool]]\nname = "get_current_time"\nenabled = true\n')
    snap = policy_view.snapshot()
    assert snap["servers"][0]["enabled"] == ["get_current_time"]


def test_missing_toml_yields_empty_server():
    view = policy_view.server_view("ghost", {"command": "uvx"})
    assert view["tools_total"] == 0 and view["enabled"] == []


@pytest.fixture()
def client(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ui_server", Path(__file__).resolve().parents[2] / "ui" / "server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Stub the pipeline: the ws handler awaits run_bot; park until the page
    # disconnects, so the tray/greeting/busy flows are exercised without
    # booting the real voice stack (MCP connects, Piper, VAD) per test.
    async def fake_run_bot(transport, runner_args, session=None,
                           mcp_clients=None, base_tools=None):
        try:
            while True:
                await runner_args.websocket.receive_text()
        except Exception:
            pass  # page left — end the session like the real run_bot would

    monkeypatch.setattr(mod.pipeline, "run_bot", fake_run_bot)
    # Warm-cache greeting without the (networked) warm-up: one silent second.
    monkeypatch.setattr(mod, "GREETING_AUDIO", (b"\x00\x00" * 16000, 16000))

    from fastapi.testclient import TestClient

    return TestClient(mod.app), mod


def test_api_allowlist_and_pages(client):
    tc, _ = client
    r = tc.get("/api/allowlist")
    assert r.status_code == 200
    # The API serves policy_view's TOML truth augmented with a live
    # "connected" health flag (mcp_client.HEALTH) per server — compare
    # the TOML fields exactly, tolerating the extra flag.
    expected = policy_view.snapshot()["servers"]
    body = [{k: v for k, v in s.items() if k != "connected"}
            for s in r.json()["servers"]]
    assert body == expected
    assert "mcpGroups" in tc.get("/").text


def test_ws_pushes_allowlist_snapshot(client):
    tc, mod = client
    with tc.websocket_connect("/ws?cid=t1") as ws:
        first = ws.receive_json()
        assert first["type"] == "mcp"
        assert [s["name"] for s in first["servers"]] == [
            s["name"] for s in policy_view.snapshot()["servers"]
        ]


def test_ws_greets_on_ready(client):
    tc, mod = client
    with tc.websocket_connect("/ws?cid=t3") as ws:
        assert ws.receive_json()["type"] == "mcp"
        # Mic live: the page asks for the instant greeting from the cache —
        # text bubble first, then the cached audio, then the turn ends.
        assert tc.post("/api/greet?cid=t3").json()["ok"] is True
        greeting = ws.receive_json()
        assert greeting["type"] == "text"
        assert greeting["t"] == mod.GREETING_TEXT
        audio = ws.receive_bytes()
        assert int.from_bytes(audio[:2], "little") == 16000  # rate header
        assert len(audio) > 2  # actual PCM follows
        assert ws.receive_json()["type"] == "done"


def test_ws_busy_for_duplicate_cid(client):
    tc, mod = client
    # One conversation per tab id: the same cid from a second socket is told
    # plainly, then the server closes that socket. (A different cid is a
    # separate tab and runs its own session.)
    with tc.websocket_connect("/ws?cid=a") as ws1:
        assert ws1.receive_json()["type"] == "mcp"
        with tc.websocket_connect("/ws?cid=a") as ws2:
            assert ws2.receive_json()["type"] == "busy"
