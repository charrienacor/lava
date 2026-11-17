"""MCP clients — one MCPClient per server in mcp/mcp_servers.json.

Connection concerns only:

- ${VAR} placeholders in server specs are expanded from the environment; a
  server whose required vars are missing is skipped with a warning, not a
  crash.
- stdio servers get the SDK's sanitized default environment merged with any
  declared env, so they keep PATH (for uv) and receive their keys.
- What the LLM sees of each server's tools (the allowlist) is tool policy
  and lives in agent/tools/<server>.toml: flip `enabled` per tool.
"""

import asyncio
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
from mcp.types import TextContent

from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient, _is_connection_lost

_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVERS_JSON = _ROOT / "mcp" / "mcp_servers.json"
TOOLS_DIR = Path(__file__).resolve().parent / "tools"
_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# Per-server tool schemas captured at connect (see
# connect_mcp_clients_report): {server: {tool: {"required": [...],
# "properties": [...]}}}. The tool gateway reads these to advertise EXACT
# argument names per action — with the full schemas hidden (progressive
# disclosure), the model was guessing (uri vs track_uri, uris lists) and
# burning turns on validation errors.
SERVER_TOOL_SCHEMAS: dict[str, dict[str, dict]] = {}

# Live per-server reachability for the web tray's health dot. True/False
# from the last definitive contact — the startup connect or the most recent
# tool call (whose invisible retry makes the final attempt the truth).
# Absent = never contacted (the UI shows a neutral dot).
HEALTH: dict[str, bool] = {}


def load_allowlist(name: str) -> list[str]:
    """Enabled tool names for a server, read from agent/tools/<name>.toml.

    The TOMLs are the single source of truth for what the LLM sees: every
    ``[[tool]]`` with ``enabled = true`` is exposed. Missing file → no tools.
    """
    path = TOOLS_DIR / f"{name}.toml"
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        logger.warning(f"MCP '{name}': no agent/tools/{name}.toml — no tools exposed")
        return []
    return [t["name"] for t in data.get("tool", []) if t.get("enabled")]


class _EnvExpander:
    """Expands `${VAR}` placeholders and records any vars that are missing."""

    def __init__(self) -> None:
        self.missing: set[str] = set()

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            def repl(match: re.Match) -> str:
                var = match.group(1)
                val = os.environ.get(var, "")
                if not val:
                    self.missing.add(var)
                return val

            return _ENV_VAR_RE.sub(repl, value)
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        return value


def build_mcp_clients() -> dict[str, MCPClient]:
    """Create one MCPClient per server in mcp/mcp_servers.json."""
    servers = json.loads(MCP_SERVERS_JSON.read_text())["mcpServers"]
    clients: dict[str, MCPClient] = {}
    for name, raw_spec in servers.items():
        expander = _EnvExpander()
        spec = expander(raw_spec)
        kind = spec.get("type", "stdio")
        try:
            if kind == "stdio":
                declared_env = spec.get("env") or {}
                params = StdioServerParameters(
                    command=spec["command"],
                    args=spec.get("args", []),
                    env={**get_default_environment(), **declared_env}
                    if declared_env
                    else None,
                )
            elif kind == "http":
                params = StreamableHttpParameters(
                    url=spec["url"],
                    headers=spec.get("headers"),
                    # 30 s, not the old 5 s: a remote search or extract can
                    # legitimately take longer than 5 s, and a timeout here
                    # surfaces as a tool error the LLM then retries — the
                    # wait doubles. The timeout bounds patience, not speed.
                    timeout=spec.get("timeout", 30),
                    sse_read_timeout=spec.get("sse_read_timeout", 300),
                    terminate_on_close=spec.get("terminate_on_close", True),
                )
            elif kind == "sse":
                params = SseServerParameters(
                    url=spec["url"],
                    headers=spec.get("headers"),
                    timeout=spec.get("timeout", 30),
                    sse_read_timeout=spec.get("sse_read_timeout", 300),
                )
            else:
                logger.warning(f"MCP '{name}': unknown type '{kind}', skipping")
                continue
        except KeyError as e:
            logger.warning(f"MCP '{name}': missing field {e}, skipping")
            continue

        if expander.missing:
            logger.warning(
                f"MCP '{name}': required env var(s) not set, skipping: "
                f"{sorted(expander.missing)} (add to .env)"
            )
            continue

        # Default-deny: a server with nothing enabled in its TOML contributes
        # no tools at all.
        allowlist = load_allowlist(name)
        if not allowlist:
            logger.warning(
                f"MCP '{name}': no enabled tools in agent/tools/{name}.toml — "
                f"skipping (flip `enabled` in the TOML)"
            )
            continue

        clients[name] = MCPClient(
            server_params=params,
            tools_filter=allowlist,
        )
    return clients


async def call_mcp_tool(
    client: MCPClient, name: str, arguments: dict[str, Any], server: str | None = None
) -> str:
    """Call an MCP tool directly on a client's session.

    Same session path Pipecat's own tool wrapper uses; needed because the
    voice loop (not the LLM) drives the asr/tts tools. Raises on failure and
    drops a dead session so the next call reconnects.

    One self-heal before an error reaches the model (retries ONCE on a fresh
    session — a second failure is a real error):

    - Dead transport: shared HTTP sessions die while idle ("Connection
      closed"); the retry re-establishes the session invisibly.
    """
    try:
        return await _call_mcp_tool_once(client, name, arguments, server)
    except asyncio.CancelledError:
        raise  # barge-in or shutdown: let the caller unwind
    except Exception as e:
        if _is_connection_lost(e) or "connection closed" in str(e).lower():
            logger.warning(f"MCP '{server or '?'}': dead session — reconnecting once")
            return await _call_mcp_tool_once(client, name, arguments, server)
        raise


async def _call_mcp_tool_once(
    client: MCPClient, name: str, arguments: dict[str, Any], server: str | None
) -> str:
    server = server or "unknown"
    session = None
    try:
        session = await client._open_session(as_owner=False)
        result = await session.call_tool(name, arguments=arguments)
    except asyncio.CancelledError:
        raise  # barge-in or shutdown: let the caller unwind
    except Exception as e:
        if session is not None:
            try:
                await client._drop_session(session)
            except Exception:
                pass
        HEALTH[server] = False
        raise RuntimeError(f"MCP tool '{name}' failed: {e}") from e
    text = "".join(
        c.text for c in (result.content or []) if isinstance(c, TextContent) and c.text
    )
    HEALTH[server] = True
    return text


async def ensure_server_schema(name: str, client: MCPClient) -> bool:
    """Capture a server's tool schema if not already captured (lazy).

    connect_mcp_clients_report captures every schema at startup — but if a
    remote server was down at that moment (the spotify worker has episodes),
    the capture is empty and the model is left guessing argument names. The
    gateway calls this before building its index, so the first successful
    session after a backend outage heals the schema.
    """
    if SERVER_TOOL_SCHEMAS.get(name):
        return True
    try:
        result = await client.tools()
    except Exception as e:
        logger.warning(
            f"MCP '{name}': schema capture failed ({e}) — the model will "
            f"guess argument names until a capture succeeds"
        )
        return False
    SERVER_TOOL_SCHEMAS[name] = {
        t.name: {
            "required": list(t.required or []),
            "properties": sorted((t.properties or {}).keys()),
        }
        for t in result.standard_tools
    }
    return True


async def connect_mcp_clients(
    clients: dict[str, MCPClient],
    exclude_from_schema: set[str] | None = None,
) -> ToolsSchema:
    """Connect every client and merge tools into one ToolsSchema.

    Thin wrapper over connect_mcp_clients_report that discards the failure
    report (see the docstring there for semantics).
    """
    tools, _failed = await connect_mcp_clients_report(clients, exclude_from_schema)
    return tools


async def connect_mcp_clients_report(
    clients: dict[str, MCPClient],
    exclude_from_schema: set[str] | None = None,
) -> tuple[ToolsSchema, list[str]]:
    """Connect every client and merge tools into one ToolsSchema.

    Every server connects (the voice loop needs its sessions), but only
    servers NOT in ``exclude_from_schema`` contribute tool definitions to
    the LLM schema — asr/tts are voice-loop-driven and their tools must
    never be model-callable (token bloat + attack surface). Each client's
    allowlist has already been applied by ``tools()``. Failing servers are
    skipped with an error log and reported in the returned list (health
    checks need to know; the voice loop does not care).
    """
    if not clients:
        raise RuntimeError("No MCP servers could be configured")

    names = list(clients)
    results = await asyncio.gather(
        *(clients[n].tools() for n in names), return_exceptions=True
    )

    merged = []
    failed: list[str] = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            failed.append(name)
            HEALTH[name] = False
            logger.error(f"MCP '{name}' connection failed: {result}")
            continue
        HEALTH[name] = True
        SERVER_TOOL_SCHEMAS[name] = {
            t.name: {
                "required": list(t.required or []),
                "properties": sorted((t.properties or {}).keys()),
            }
            for t in result.standard_tools
        }
        chars = sum(len(json.dumps(t.to_default_dict())) for t in result.standard_tools)
        tool_names = sorted(t.name for t in result.standard_tools)
        logger.info(
            f"MCP '{name}' connected: {len(tool_names)} tool(s), "
            f"~{chars // 4} tokens {tool_names}"
        )
        if exclude_from_schema is None or name not in exclude_from_schema:
            merged.extend(result.standard_tools)

    tools_schema = ToolsSchema(standard_tools=merged)
    chars = sum(len(json.dumps(t.to_default_dict())) for t in merged)
    connected = len(names) - len(failed)
    logger.info(
        f"MCP tools ready: {len(merged)} tool(s) in the LLM schema, from "
        f"{connected}/{len(names)} server(s) (~{chars // 4} tokens of context)"
    )
    return tools_schema, failed
