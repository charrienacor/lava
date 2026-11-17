"""Progressive disclosure + enforced confirm gate for MCP capability servers.

Groq's free tier allows 8,000 tokens per MINUTE. Full definitions for the 15
enabled GitHub tools cost ~5,600 tokens of that on every single request. This
module hides a wrapped server behind ONE generic tool whose definition is a
compact index — one line per action, generated from the same TOML allowlist —
and forwards calls to the real MCP server over its existing session. The
server's full tool schemas are never sent to the LLM; bad arguments surface
as an error message the model can correct on its next turn.

Confirm gate: TOML tools marked ``confirm = true`` cannot run from a single
call. The first call is refused with CONFIRMATION_REQUIRED (the model must ask
the user out loud); the re-call must carry ``confirmed = true`` AND the newest
user MESSAGE must have changed since the request, so the model cannot confirm
on the user's behalf. The message — not a turn count — is the anchor: the
auto-summarizer compresses old turns mid-conversation, and a count recorded
before a summarize could never be surpassed again (measured: four spoken
approvals, every one bounced with CONFIRMATION_STILL_PENDING). Once an
approval lands, the TARGET (repo/PR/file/playlist/tracks — see
_IDENTITY_KEYS) is granted for the rest of the session: same-target repeats
(iterating on the just-approved page) skip the handshake entirely, while a
different target is a new fingerprint and asks again. Actions without an
identity mapping fall back to full arguments.

Patience (mechanical, from conversation-design practice): Google's Dialogflow
ends a dialog after 3 no-input/no-match attempts, and ServiceNow's guide caps
clarifications at 2 ("then make your best guess and confirm"). Lava's logs
measured the uncapped behavior: one clarifying question asked 29 times,
spotify_play attempted 58. So: at most MAX_ASKS_PER_TARGET approval rounds per
target, at most MAX_ACTION_FAILS failed calls per action per session — then a
terminal tool result orders the model to tell the user plainly. Success resets
the counters; the caps are per session.

Design: the guards are data; the gateway is the only stateful piece.

    policy source     the TOMLs, re-read on every call (live policy)
    policy tables     identity keys, patience caps
    per-session state SessionState + _Call (fresh per session)
    guards            deny / confirm gate / strikeout / breaker — one concern each
    ServerGateway     owns client + state; handle() = guards → call → frame
    gateway_schemas   entry point: register one gateway per wrapped server
"""

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from mcp_client import (
    SERVER_TOOL_SCHEMAS,
    call_mcp_tool,
    ensure_server_schema,
)

TOOL_DIR = Path(__file__).resolve().parent / "tools"


# ---- Policy source: agent/tools/<server>.toml, re-read live ---------------
# The TOMLs are read at CALL time: an `enabled` flip or a `confirm`
# tightening applies to the running session on its very next tool call —
# no reconnect, no restart. The largest TOML parses in ~1 ms, noise next
# to a tool call.


def _catalogue(server: str) -> list[tuple[str, str, bool]]:
    """(action, one-line summary, confirm?) for every enabled tool of a server.

    A missing or unreadable TOML means no tools — policy gaps must never
    crash the agent.
    """
    try:
        with open(TOOL_DIR / f"{server}.toml", "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    entries = []
    for tool in data.get("tool", []):
        if not tool.get("enabled"):
            continue
        desc = tool.get("description", "").strip()
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
        if len(first) > 110:
            first = first[:107].rstrip() + "..."
        entries.append((tool["name"], first, bool(tool.get("confirm"))))
    return entries


def _known_tools(server: str) -> dict[str, str]:
    """Every tool the TOML lists for this server, enabled or not.

    name -> one-line summary. Lets the gateway tell a DISABLED tool apart
    from an UNKNOWN one — a disabled tool is a policy matter the model can
    raise with the user; an unknown one is a mistake to correct immediately.
    """
    try:
        with open(TOOL_DIR / f"{server}.toml", "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    known = {}
    for tool in data.get("tool", []):
        desc = tool.get("description", "").strip()
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
        if len(first) > 110:
            first = first[:107].rstrip() + "..."
        known[tool.get("name", "")] = first
    return known


# ---- What the LLM sees: ONE generic tool per heavy server ------------------


def _schema(server: str, catalogue: list[tuple[str, str, bool]]) -> FunctionSchema:
    """Build the generic '<server>' tool definition from the TOML index.

    The live tool schemas captured at connect (mcp_client
    SERVER_TOOL_SCHEMAS) add an exact-argument section: with the full schemas
    hidden, the model otherwise GUESSES parameter names — measured 27 identical
    failing spotify calls in one conversation because of `uri` vs `track_uri`.
    """
    names = [n for n, _, _ in catalogue]
    bullets = "\n".join(f"- {n}: {s}" for n, s, _ in catalogue)
    description = (
        f"Run any {server} action. Pass the action name and an arguments "
        f"object matching that action's parameters (e.g. owner, repo, "
        f"title); a failed call returns an error to fix and retry. Actions:\n"
        f"{bullets}"
    )
    schema_hint = SERVER_TOOL_SCHEMAS.get(server) or {}
    arg_lines = []
    for action, _, _ in catalogue:
        info = schema_hint.get(action)
        if not info:
            continue
        required = set(info.get("required") or [])
        args = ", ".join(
            f"{p}*" if p in required else p for p in info.get("properties") or []
        )
        if args:
            arg_lines.append(f"- {action}: {args}")
    if arg_lines:
        description += (
            "\n\nExact argument names per action (a * marks required; use "
            "these names exactly):\n" + "\n".join(arg_lines)
        )
    description += (
        f"\n\nCall shape: this tool is always invoked under the exact name "
        f"`{server}` — never as `{server}.<action>` or any dotted variant. "
        f"Pick the action from the list above and pass it as the `action` "
        f"string."
    )
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            # No enum: the advertised list is a session-start snapshot, and a
            # hard enum would block the model from retrying an action the
            # user enables mid-session (the gateway re-checks policy live).
            "description": f"{server} action to run — one of the actions "
                           f"listed above (use the exact names)",
        },
        "arguments": {
            "type": "object",
            "description": (
                "Arguments object for that action, matching its schema; "
                "omit or pass {} when the action takes none."
            ),
        },
        "confirmed": {
            "type": "boolean",
            "description": (
                "Set true ONLY to re-run an action after the user approved "
                "it following a CONFIRMATION_REQUIRED result."
            ),
        },
    }
    schema = FunctionSchema(
        name=server,
        description=description,
        properties=properties,
        required=["action"],
    )
    chars = len(json.dumps(schema.to_default_dict()))
    guarded = sum(1 for _, _, c in catalogue if c)
    logger.info(
        f"tool gateway '{server}': {len(names)} actions behind 1 tool "
        f"(~{chars // 4} tokens, {guarded} confirm-gated)"
    )
    return schema


# ---- Small shared helpers ---------------------------------------------------


def _user_turn_count(params) -> int:
    """Number of user messages currently in the LLM context."""
    messages = getattr(params.context, "messages", None) or []
    return sum(
        1
        for m in messages
        if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
    )


def _last_user_text(params) -> str:
    """The newest user message's text ("" when none).

    This is the confirm gate's "has the user spoken since the ask?" anchor.
    A turn COUNT breaks under auto-summarization: the summarizer replaces
    old user turns with one summary message, the count shrinks, and an
    approval recorded before the summarize can never be surpassed again
    (measured: four spoken approvals — "Push it!", "Yes.", "Go ahead.",
    "Yes." — every one bounced with CONFIRMATION_STILL_PENDING). The
    newest user MESSAGE survives summarization (summarization keeps the
    recent turns), so comparing it before/after is order-safe. Barge-in
    notes use role "developer", so an unheard interruption never counts
    as an answer.
    """
    messages = getattr(params.context, "messages", None) or []
    for m in reversed(messages):
        if isinstance(m, dict) and str(m.get("role", "")).lower() == "user":
            return str(m.get("content", ""))
    return ""


# ---- Protected-action policy: identity binding, patience -------------------

# The identity of a protected call: which resource it touches. Free-text
# payload keys (title, body, content, ...) are deliberately excluded, so
# wording tweaks after a spoken "yes" ride along; a different target —
# another PR, file, playlist or track list — is a new action needing new
# approval. Actions missing from this map fall back to full arguments.
_IDENTITY_KEYS = {
    "github": {
        "merge_pull_request": ("owner", "repo", "pullNumber"),
        "create_or_update_file": ("owner", "repo", "branch", "path"),
        "push_files": ("owner", "repo", "branch"),
        "create_repository": ("name", "org"),
    },
    "spotify": {
        "spotify_add_tracks": ("playlist_id", "uris"),
        "spotify_create_playlist": ("name",),
    },
}

# Patience caps (see module docstring). Terminal results tell the model to
# stop and speak plainly; they reset on success or a new session.
MAX_ASKS_PER_TARGET = 2  # approval rounds per target before terminal refusal
MAX_ACTION_FAILS = 3  # failed calls per action per session before terminal


def _identity(server: str, action: str, args: dict[str, Any]) -> str:
    """Fingerprint of a protected call: action + its target, nothing more."""
    keys = _IDENTITY_KEYS.get(server, {}).get(action)
    if keys is None:  # unmapped gated action: stay strict
        identity = args
    else:
        identity = {k: args.get(k) for k in keys if k in args}
        if server == "github" and action == "push_files":
            # The affected file paths are target, not payload.
            files = args.get("files")
            if isinstance(files, list):
                identity["files"] = sorted(
                    f.get("path", "?") for f in files if isinstance(f, dict)
                )
    return action + ":" + json.dumps(identity, sort_keys=True, default=str)


def _call_key(action: str, call_args: dict[str, Any]) -> str:
    """Fingerprint of an exact call (circuit-breaker key)."""
    return action + ":" + json.dumps(call_args, sort_keys=True, default=str)


# ---- Per-session state ------------------------------------------------------


@dataclass
class SessionState:
    """The gateway's per-session memory (a fresh instance per server).

    Keys bounded at 64 entries: these dicts only ever hold one entry per
    fingerprint/call key seen, but a long adversarial session could grow
    them — clear rather than grow.
    """

    pending: dict[str, str] = field(default_factory=dict)
    # fingerprint -> newest user message text at ask time
    granted: set[str] = field(default_factory=set)
    # fingerprints approved this session: same-target repeats skip the
    # handshake entirely (iterating on a just-approved page/playlist is
    # the normal pattern); a different target is a new fingerprint
    asks: dict[str, int] = field(default_factory=dict)
    # fingerprint -> approval rounds issued
    fails: dict[str, int] = field(default_factory=dict)
    # call key -> consecutive identical failures (circuit breaker)
    action_fails: dict[str, int] = field(default_factory=dict)
    # action -> failed calls this session (strikeout)
    ask_blocked: dict[str, str] = field(default_factory=dict)
    # fingerprint -> newest user message text at terminal refusal
    strike_blocked: dict[str, str] = field(default_factory=dict)
    # action -> newest user message text at terminal refusal


@dataclass
class _Call:
    """One gateway invocation, prepared.

    arguments   the model's raw arguments (carries the `confirmed` flag)
    call_args   the final arguments for the real MCP call
    fingerprint target identity — the confirm gate's key
    call_key    exact-call identity — the circuit breaker's key
    """

    action: str
    arguments: dict[str, Any]
    call_args: dict[str, Any]
    fingerprint: str
    call_key: str


# ---- Guards: run before anything executes; first denial wins ---------------
# Each guard takes (gateway, params, call), returns a denial string to speak
# UNFRAMED (an instruction to the model, not tool output) or None to proceed.


def _deny_unallowed(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """Denial for an action outside the LIVE policy — None if allowed."""
    allowed = {n for n, _, _ in gw.catalogue}
    if call.action in allowed:
        return None
    known = _known_tools(gw.server)
    if call.action in known:
        # A real tool of this server, switched off by policy. The model
        # cannot enable it itself — route the request to the user, then the
        # retry lands on the live policy.
        return (
            f"'{call.action}' is a real {gw.server} tool but it is DISABLED "
            f"in Lava's tool policy (agent/tools/{gw.server}.toml). You "
            f"cannot enable it yourself: ask the user to set "
            f"enabled = true for '{call.action}' in that file, then retry "
            f"this exact action. What it does: {known[call.action]}"
        )
    return (
        f"Error: unknown action '{call.action}' for {gw.server}. "
        f"Valid actions: {', '.join(sorted(allowed))}."
    )


def _confirm_gate(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """The enforced confirm gate for TOML-flagged actions — None to proceed.

    Order of checks (each returns a denial, or falls through):
      already granted → patience cap → not confirmed yet → no ask on
      record → user has not spoken since the ask → pass (approval
      consumed, target granted for the session).

    The "has the user answered" check anchors on the newest user MESSAGE,
    not a turn count: the auto-summarizer compresses old turns
    mid-conversation, and a count recorded before a summarize could never
    be surpassed again — the gate deadlocked (measured: four spoken
    approvals for one push_files, every one bounced).
    """
    if call.action not in {n for n, _, c in gw.catalogue if c}:
        return None
    state = gw.state

    # Session grant: this exact target was already approved once. Repeats
    # (wording tweaks, content iterations on the same repo/playlist) ride
    # through silently; a different target is a new fingerprint and asks.
    if call.fingerprint in state.granted:
        return None

    # Patience cap: after MAX_ASKS_PER_TARGET approval rounds for this
    # target, refuse terminally. This is what stops the ask-confirm-fail-
    # ask-again loop (measured 72 CONFIRMATION_REQUIRED and one question
    # re-asked 29 times). A fresh user turn after the refusal re-opens one
    # round — the user explicitly asked for another go (anchored on the
    # newest user message: summarization-proof).
    spoken_now = _last_user_text(params)
    blocked_at = state.ask_blocked.get(call.fingerprint)
    if state.asks.get(call.fingerprint, 0) >= MAX_ASKS_PER_TARGET:
        if blocked_at is not None and spoken_now != blocked_at:
            state.asks.pop(call.fingerprint, None)
            state.ask_blocked.pop(call.fingerprint, None)
        else:
            if blocked_at is None:
                state.ask_blocked[call.fingerprint] = spoken_now
            return (
                f"You have already asked the user to approve '{call.action}' "
                f"{MAX_ASKS_PER_TARGET} times and it still has not gone "
                f"through. Stop asking and stop calling {gw.server} tools "
                f"for this. Tell the user plainly that it is not working "
                f"right now."
            )
    if not bool(call.arguments.get("confirmed")):
        state.pending[call.fingerprint] = spoken_now
        state.asks[call.fingerprint] = state.asks.get(call.fingerprint, 0) + 1
        if len(state.pending) > 64:  # per-session state, keep bounded
            state.pending.clear()
            state.pending[call.fingerprint] = spoken_now
        return (
            f"CONFIRMATION_REQUIRED: '{call.action}' is a protected "
            f"action. Ask the user out loud to approve it "
            f"(state the key arguments), then call again with the "
            f"same target plus confirmed=true. Do not set "
            f"confirmed=true until the user has answered."
        )
    asked_text = state.pending.get(call.fingerprint)
    if asked_text is None:
        return (
            f"CONFIRMATION_REQUIRED: the user never approved this "
            f"target. Ask first; only after they answer, "
            f"re-call with the same target plus confirmed=true."
        )
    if spoken_now == asked_text:
        return (
            f"CONFIRMATION_STILL_PENDING: the user has not "
            f"answered yet. Ask the question and wait for their "
            f"reply before calling with confirmed=true."
        )
    # The approval is consumed and the target is granted for the session:
    # same-target repeats (content tweaks on the same repo/playlist) no
    # longer ask; a different target is a new fingerprint and does.
    state.pending.pop(call.fingerprint)
    state.granted.add(call.fingerprint)
    if len(state.granted) > 64:  # per-session state, keep bounded
        state.granted.clear()
        state.granted.add(call.fingerprint)
    return None


def _strikeout(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """Terminal refusal once an action failed MAX_ACTION_FAILS times.

    The user has waited through all of them — refuse, unless a fresh user
    turn says "try again", which re-opens one attempt (success resets the
    counter entirely).
    """
    state = gw.state
    if state.action_fails.get(call.action, 0) < MAX_ACTION_FAILS:
        return None
    spoken_now = _last_user_text(params)
    blocked_at = state.strike_blocked.get(call.action)
    if blocked_at is not None and spoken_now != blocked_at:
        state.action_fails.pop(call.action, None)
        state.strike_blocked.pop(call.action, None)
        return None
    if blocked_at is None:
        state.strike_blocked[call.action] = spoken_now
    return (
        f"'{call.action}' has already failed {MAX_ACTION_FAILS} times this "
        f"session. Do not call it again. Tell the user plainly that it "
        f"is not working right now; only try again if they ask you to."
    )


def _circuit_breaker(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """Circuit breaker: the identical call has already failed twice.

    Retrying a third time burns tokens and the user's patience — measured
    27 identical failing spotify calls in one conversation.
    """
    if gw.state.fails.get(call.call_key, 0) >= 2:
        return (
            f"Error: '{call.action}' already failed "
            f"{gw.state.fails[call.call_key]} times with these exact "
            f"arguments. Do not retry with the same arguments. Try a "
            f"different action or approach, or tell the user it is not "
            f"working right now."
        )
    return None


# ---- Execution and framing --------------------------------------------------


async def _run_tool(
    client: MCPClient, server: str, action: str, call_args: dict[str, Any]
) -> str:
    """One real MCP call; any failure becomes an 'Error: …' tool result."""
    try:
        return await call_mcp_tool(client, action, call_args, server=server)
    except Exception as e:
        return f"Error: {e}"


def _framed(server: str, action: str, result: str) -> str:
    """Delimit the result so the model can tell tool output from
    instructions (the system prompt declares this content untrusted)."""
    return (
        f"<tool_result server={server} action={action}>\n"
        f"{result or '(no output)'}\n"
        f"</tool_result>"
    )


# ---- The gateway: one stateful object per wrapped server -------------------


class ServerGateway:
    """One LLM-facing gateway tool for one wrapped server, one session.

    An instance owns its client and its per-session state (approvals,
    failure counters). ``gateway_schemas`` registers ``handle`` — a bound
    method, so nothing leaks across loop iterations and no
    closure-binding tricks are needed.
    """

    def __init__(self, server: str, client: MCPClient) -> None:
        self.server = server
        self.client = client
        self.state = SessionState()
        # The LIVE policy snapshot for the call in flight: prepare re-reads
        # it on every call; the guards read it from here.
        self.catalogue: list[tuple[str, str, bool]] = []

    async def prepare(self, params) -> _Call:
        """Parse the model's call into the pipeline's call record."""
        arguments = dict(params.arguments or {})
        action = str(arguments.get("action", ""))
        # LIVE policy: the TOML is re-read at call time, so an enable/disable
        # or confirm edit applies to this running session immediately — no
        # reconnect, no new tap.
        self.catalogue = _catalogue(self.server)
        call_args = dict(arguments.get("arguments") or {})
        return _Call(
            action=action,
            arguments=arguments,
            call_args=call_args,
            fingerprint=_identity(self.server, action, call_args),
            call_key=_call_key(action, call_args),
        )

    def record(self, call: _Call, result: str) -> None:
        """Bookkeeping after the attempt: failures count, success resets."""
        if isinstance(result, str) and result.startswith("Error"):
            self.state.fails[call.call_key] = (
                self.state.fails.get(call.call_key, 0) + 1
            )
            self.state.action_fails[call.action] = (
                self.state.action_fails.get(call.action, 0) + 1
            )
            if len(self.state.fails) > 64:
                self.state.fails.clear()
        else:
            self.state.fails.pop(call.call_key, None)
            self.state.action_fails.pop(call.action, None)  # patience resets
            self.state.asks.pop(call.fingerprint, None)
            self.state.pending.pop(call.fingerprint, None)

    async def handle(self, params) -> None:
        """The registered tool handler: guards, then the call pipeline."""
        call = await self.prepare(params)

        for guard in GUARDS:
            if denial := guard(self, params, call):
                await params.result_callback(denial)
                return

        result = await _run_tool(self.client, self.server, call.action,
                                 call.call_args)
        self.record(call, result)
        await params.result_callback(_framed(self.server, call.action, result))


# ---- The flow, as data -----------------------------------------------------
# The per-call flow (ServerGateway.handle) is fixed; its rules are data:
#
#   GUARDS    run in order before anything executes — the first denial
#             wins and is spoken unframed.
#
# Adding a rule = adding one function + one entry here.

GUARDS = (_deny_unallowed, _confirm_gate, _strikeout, _circuit_breaker)


# ---- Entry point: register one gateway per wrapped server ------------------


async def gateway_schemas(
    llm: LLMService,
    mcp_clients: dict[str, MCPClient],
    servers: set[str],
) -> list[FunctionSchema]:
    """Register one gateway per wrapped server and return its schema.

    The returned schemas must be added to the ToolsSchema the LLM context
    advertises; the handlers are registered on the LLM here. State is
    per-session by construction: a fresh ServerGateway (owning its
    SessionState) per wrapped server.

    Schemas are captured lazily per server: if the backend was down at
    startup (spotify worker episodes), the first session after it recovers
    heals the captured schema, and the model gets exact argument names again.
    """
    schemas = []
    for server in sorted(servers):
        client = mcp_clients.get(server)
        if client is None:
            logger.error(f"tool gateway '{server}': no MCP client, skipping")
            continue
        await ensure_server_schema(server, client)
        gateway = ServerGateway(server, client)
        # cancel_on_interruption=False: these are side-effectful commands,
        # and remote ones — a spotify or github call can take seconds. With
        # the sync default, a barge-in during the call CANCELLED it
        # (measured in the logs: the context got 'CANCELLED' and the UI's
        # command toast never resolved, so the user never saw the outcome
        # of a command they had asked for). As an async tool the call
        # survives barge-in, the result always lands (toast resolves;
        # pipecat injects it for the next turn), and a completed command is
        # never lost mid-flight.
        llm.register_function(server, gateway.handle, cancel_on_interruption=False)
        schemas.append(_schema(server, _catalogue(server)))
    return schemas
