"""UI view of the tool policy — what the web client's MCP tray renders.

Single source of truth stays the TOMLs (agent/tools/*.toml, same files
mcp_client.load_allowlist reads). This module just reshapes them for the
UI: per server, the full tool universe the TOML knows about, which tools
are allowlisted (enabled = true) and which require spoken confirmation
(confirm = true). Bridge servers (asr/tts) are included and marked so the
UI can label them — the LLM never sees their tools.

The web server re-reads this snapshot on every request/watch tick, so a
TOML edit shows up on the next poll — no process restart, no cache.
"""

import json
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = _ROOT / "agent" / "tools"
SERVERS_JSON = _ROOT / "mcp" / "mcp_servers.json"


# LLM never sees.
BRIDGE_SERVERS = {"asr", "tts"}


def server_specs() -> dict:
    """Ordered {name: spec} from mcp/mcp_servers.json."""
    data = json.loads(SERVERS_JSON.read_text())
    return data.get("mcpServers", {})


def _kind(spec: dict) -> str:
    return "stdio" if "command" in spec else "remote"


def server_view(name: str, spec: dict) -> dict:
    """One tray entry: tool universe, allowlist, and confirm-guarded set."""
    path = TOOLS_DIR / f"{name}.toml"
    tools: list[dict] = []
    if path.exists():
        data = tomllib.loads(path.read_text())
        tools = [
            {
                "name": t["name"],
                "enabled": bool(t.get("enabled", False)),
                "confirm": bool(t.get("confirm", False)),
                "description": str(t.get("description", "")),
            }
            for t in data.get("tool", [])
            if t.get("name")
        ]
    return {
        "name": name,
        "kind": _kind(spec),
        "role": "bridge" if name in BRIDGE_SERVERS else "capability",
        "tools_total": len(tools),
        "enabled": [t["name"] for t in tools if t["enabled"]],
        "confirm": [t["name"] for t in tools if t["enabled"] and t["confirm"]],
        "tools": tools,
    }


def snapshot() -> dict:
    """The full tray payload, in mcp_servers.json order."""
    specs = server_specs()
    return {"servers": [server_view(name, spec) for name, spec in specs.items()]}
