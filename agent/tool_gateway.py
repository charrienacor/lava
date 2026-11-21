"""Progressive disclosure + enforced confirm gate for MCP capability servers.

Groq's free tier allows 8,000 tokens per MINUTE. The enabled GitHub tools'
full definitions cost ~2,540 tokens of that on every single request. This
module hides a wrapped server behind ONE generic tool whose definition is a
compact index — one line per action, generated from the same TOML allowlist —
and forwards calls to the real MCP server over its existing session. The
server's full tool schemas are never sent to the LLM; bad arguments surface
as an error message the model can correct on its next turn.

Confirm gate: TOML tools marked ``confirm = true`` cannot run from a single
call. The first call is refused with CONFIRMATION_REQUIRED (the model must ask
the user out loud); the re-call must carry ``confirmed = true`` AND the user
must have spoken since the ask, so the model cannot confirm on the user's
behalf. "Has the user spoken" is detected by EITHER the newest user message
text changing OR the user-turn count growing. Text alone fails when the user
repeats the same word ("Yes." twice); the count alone fails when the
auto-summarizer compresses old turns (the count shrinks and a count recorded
before a summarize could never be surpassed again). Using both covers each
other's blind spot.

Once an approval lands, the TARGET (repo/PR/file/playlist/tracks — the
TOMLs' `identity = [...]` keys) is granted for the rest of the session:
same-target repeats (iterating on the just-approved page) skip the
handshake entirely, while a different target is a new fingerprint and
asks again. Fingerprints are NORMALIZED (stringified, sorted lists,
`confirmed` stripped) so cosmetic drift between the ask and the re-call
does not look like a new target. Actions without an identity mapping
fall back to full (normalized) arguments.

Loop prevention (what was measured and what fixes it):

  * Cap only limits NEW asks. A legitimate confirmed re-call after the user
    answered is never blocked by the ask budget.
  * A call made while an ask is still unanswered does NOT burn budget; it is
    answered with CONFIRMATION_STILL_PENDING.
  * Arguments are validated against the captured schema BEFORE the gate, so
    the user is never asked to approve a call that was going to fail
    (ask-confirm-fail-fix-ask-again).
  * A ``confirmed`` flag nested inside ``arguments`` is honored.
  * Server-side argument incompatibilities (schema-valid individually,
    rejected combined) are stripped at call time — see
    PARAM_INCOMPATIBILITIES — so the model never pays a learning 400.

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
    policy tables     per-action identity keys (TOML `identity`), patience caps
    per-session state SessionState + _Call (fresh per session)
    guards            deny / validate / strikeout / breaker / confirm / cap
    ServerGateway     owns client + state; handle() = guards → call → frame
    gateway_schemas   entry point: register one gateway per wrapped server
"""

import asyncio
import json
import re
import time
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

# Session start waits at most this long for missing tool schemas to heal.
SCHEMA_HEAL_TIMEOUT_SECS = 3.0
_BACKGROUND: set[asyncio.Future] = set()  # late captures: keep a reference


# ---- Policy source: agent/tools/<server>.toml, re-read live ---------------
# The TOMLs are read at CALL time: an `enabled` flip or a `confirm`
# tightening applies to the running session on its very next tool call —
# no reconnect, no restart. The largest TOML parses in ~1 ms, noise next
# to a tool call.

# The one-line summary each action gets in the gateway index. Only
# ENABLED actions are advertised, so a longer line is affordable: it buys
# real operating knowledge per action (the <login>.github.io site route,
# tavily's recency params) for ~40 chars ≈ 10 tokens each — the github
# index grows ~504 → ~600 tokens, well inside the 8k TPM budget.
_CATALOGUE_MAX_DESC = 170


def _first_sentence(desc: str) -> str:
    """The description's first sentence — the model-visible line."""
    first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
    if len(first) > _CATALOGUE_MAX_DESC:
        first = first[:_CATALOGUE_MAX_DESC - 3].rstrip() + "..."
    return first


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
        entries.append((tool["name"], _first_sentence(desc), bool(tool.get("confirm"))))
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
        known[tool.get("name", "")] = _first_sentence(desc)
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
        enums = info.get("enums") or {}
        # Params the gateway strips unconditionally are never advertised:
        # naming them only invites the model to pass what cannot run.
        always_stripped = {
            param for param, rule in (
                PARAM_INCOMPATIBILITIES.get(server, {}).get(action) or {}
            ).items()
            if rule is None
        }

        def _advertised(p):
            if p in always_stripped:
                return False  # the gateway strips it — advertising invites waste
            # include_* display toggles stay off the wire: the model never
            # needs them, and naming include_raw_content invites multi-k
            # raw dumps into an 8k-TPM context. Required/enum params are
            # always kept.
            return p in required or p in enums or not p.startswith("include_")

        def _render(p):
            label = f"{p}*" if p in required else p
            values = enums.get(p)
            # Live enum values from the captured schema: with only names
            # advertised, the model invents values ("time_range":
            # "recent") and burns a turn on the server's rejection.
            return f"{label}({'|'.join(values)})" if values else label

        args = ", ".join(
            _render(p) for p in info.get("properties") or [] if _advertised(p)
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
                "it following a CONFIRMATION_REQUIRED result. This is a "
                "top-level field, NOT part of `arguments`."
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

    One half of the confirm gate's "has the user spoken since the ask?"
    anchor (the other half is the user-turn count, see _user_anchor).

    A turn COUNT alone breaks under auto-summarization: the summarizer
    replaces old user turns with one summary message, the count shrinks, and
    an approval recorded before the summarize can never be surpassed again
    (measured: four spoken approvals — "Push it!", "Yes.", "Go ahead.",
    "Yes." — every one bounced with CONFIRMATION_STILL_PENDING). The newest
    user MESSAGE survives summarization (summarization keeps the recent
    turns), so comparing it before/after is order-safe. Text alone breaks
    when the user repeats the identical words, which is why the count is
    checked too. Barge-in notes use role "developer", so an unheard
    interruption never counts as an answer.
    """
    messages = getattr(params.context, "messages", None) or []
    for m in reversed(messages):
        if isinstance(m, dict) and str(m.get("role", "")).lower() == "user":
            return str(m.get("content", ""))
    return ""


def _user_anchor(params) -> tuple[str, int]:
    """(newest user text, user-turn count): the 'has the user spoken' anchor."""
    return _last_user_text(params), _user_turn_count(params)


def _truthy(value: Any) -> bool:
    """Lenient bool: models sometimes send "true" as a string."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


# ---- Protected-action policy: identity binding, patience -------------------

def _identity_keys(server: str) -> dict[str, tuple[str, ...]]:
    """Per-action target keys for the confirm gate, from the TOML policy.

    ``identity = [...]`` on a ``[[tool]]`` names the arguments a spoken
    "yes" binds to (github.toml, spotify.toml); every other argument is
    payload — wording tweaks ride along after approval, a different
    target is a new fingerprint and asks again. Read live like the rest
    of tool policy: an identity edit applies to the running session's
    next call, no restart.

    A missing TOML, unreadable file, or tool without ``identity`` just
    means no mapping: gated actions fall back to strict full-args
    fingerprints (safe — more re-asking, never less protection).
    """
    try:
        with open(TOOL_DIR / f"{server}.toml", "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {
        t["name"]: tuple(t["identity"])
        for t in data.get("tool", [])
        if t.get("name") and isinstance(t.get("identity"), list)
    }

# Patience caps (see module docstring). Terminal results tell the model to
# stop and speak plainly; they reset on success or a new session.
MAX_ASKS_PER_TARGET = 2  # approval rounds per target before terminal refusal
MAX_ACTION_FAILS = 3  # failed calls per action per session before terminal

# Per-turn call caps: reflexive re-calls inside ONE user turn burn seconds
# of silence (measured: one midterms answer = 3 tavily_search calls, two
# of them 400-fix retries; weak snippets invite re-searches too). Guards
# are data: after this many runs of the same action with no new user turn
# in between, the call is refused terminally — answer from the results
# already in hand. A new user turn resets the budget.
PER_TURN_ACTION_CAPS = {"tavily_search": 3}


def _norm(value: Any) -> Any:
    """Normalize one argument value so cosmetic drift is not a new target.

    12 vs "12" compare equal; lists compare order-insensitively; dicts
    compare key-order-insensitively; absent/None stays None.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return sorted(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _identity(server: str, action: str, args: dict[str, Any]) -> str:
    """Fingerprint of a protected call: action + its target, nothing more.

    Always includes every identity key (None when absent), so omitting an
    optional key on the retry does not change the fingerprint. `confirmed`
    is stripped defensively so it can never leak into the identity.
    """
    args = {k: v for k, v in args.items() if k != "confirmed"}
    keys = _identity_keys(server).get(action)
    if keys is None:  # unmapped gated action: stay strict, but normalized
        identity = dict(args)
    else:
        identity = {k: args.get(k) for k in keys}
    identity = {k: _norm(v) for k, v in identity.items()}
    if server == "github" and action == "push_files":
        # The affected file paths are target, not payload.
        files = args.get("files")
        if isinstance(files, list):
            identity["files"] = sorted(
                str(f.get("path", "?")) for f in files if isinstance(f, dict)
            )
    return action + ":" + json.dumps(identity, sort_keys=True, default=str)


def _call_key(action: str, call_args: dict[str, Any]) -> str:
    """Fingerprint of an exact call (circuit-breaker key)."""
    return action + ":" + json.dumps(call_args, sort_keys=True, default=str)


# Server-side argument incompatibilities the model cannot know: both params
# are individually valid (the schema advertises them, the enum hints list
# them), but the server rejects the combination. Measured: three
# fix-retry ladders in one evening — country + search_depth=fast cost two
# wasted calls and two extra LLM requests per searched question before the
# model learns what the schema never said. The gateway strips the pair
# silently at call time; one log line per strip keeps it auditable.
PARAM_INCOMPATIBILITIES = {
    "tavily": {
        "tavily_search": {
            # Two measured failure classes in one evening: "Country parameter
            # is not supported for fast or ultra-fast search_depth" AND
            # "Invalid country. Must be a valid country name…" (the model
            # sent 'US', then 'USA', then 'United States' — burning a
            # round trip each time it guessed wrong). The query carries
            # geography better than this argument ever has, so country
            # goes unconditionally: stripped at call time (None = always)
            # and hidden from the advertised argument hints.
            "country": None,
        },
    },
}


def _strip_incompatible(
    server: str, action: str, call_args: dict[str, Any]
) -> dict[str, Any]:
    """Drop arguments the server rejects (see the table).

    A rule value of None strips the parameter unconditionally; a dict
    {trigger: (bad_values...)} strips it only in those combinations.
    In place, returns the same dict; one INFO line per strip so the
    pattern stays visible. Adding a rule = adding one table entry.
    """
    rules = PARAM_INCOMPATIBILITIES.get(server, {}).get(action) or {}
    for drop_param, triggers in rules.items():
        if drop_param not in call_args:
            continue
        if triggers is None:
            dropped = call_args.pop(drop_param)
            logger.info(
                f"{server}.{action}: dropped '{drop_param}'={dropped!r} "
                f"(server rejects this argument — the query carries it)"
            )
            continue
        for trigger, bad_values in triggers.items():
            trigger_value = str(call_args.get(trigger, "")).lower()
            if trigger_value in {str(v).lower() for v in bad_values}:
                dropped = call_args.pop(drop_param)
                logger.info(
                    f"{server}.{action}: dropped '{drop_param}'={dropped!r} "
                    f"(server rejects it with {trigger}={call_args.get(trigger)!r})"
                )
                break
    return call_args


# ---- Per-session state ------------------------------------------------------


@dataclass
class SessionState:
    """The gateway's per-session memory (a fresh instance per server).

    Keys bounded at 64 entries: these dicts only ever hold one entry per
    fingerprint/call key seen, but a long adversarial session could grow
    them — clear rather than grow.
    """

    pending: dict[str, tuple[str, int]] = field(default_factory=dict)
    # fingerprint -> (newest user text, user-turn count) at ask time
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
    action_calls: dict[str, int] = field(default_factory=dict)
    # action -> calls run within the CURRENT user turn (per-turn caps)
    action_anchor: tuple[str, int] | None = None
    # (newest user text, turn count) when the current turn's budget opened


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


def _validate_args(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """Reject calls missing required arguments BEFORE the confirm gate.

    Without this, the gate asks the user to approve a call that the MCP
    server then rejects for bad arguments; the model fixes them, the call
    is re-gated, and the user is asked again (ask-confirm-fail-ask-again).
    Validating first means approvals are only ever requested for calls that
    can actually run. No schema captured for the action → nothing to check.
    """
    info = (SERVER_TOOL_SCHEMAS.get(gw.server) or {}).get(call.action)
    if not info:
        return None
    required = list(info.get("required") or [])
    missing = [
        p for p in required
        if p not in call.call_args or call.call_args[p] in (None, "")
    ]
    if not missing:
        return None
    return (
        f"Error: '{call.action}' is missing required argument(s): "
        f"{', '.join(missing)}. Pass them inside `arguments` using these "
        f"exact names, then retry. Do not ask the user for approval yet."
    )


def _confirm_gate(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """The enforced confirm gate for TOML-flagged actions — None to proceed.

    Decision order:
      1. Not a gated action, or target already granted      → pass.
      2. Confirmed AND the user has spoken since the ask     → pass
         (approval consumed, target granted for the session).
      3. An ask is on record but the user has not spoken     → STILL_PENDING
         (costs no budget, so same-turn retries cannot exhaust the cap).
      4. A NEW ask is needed: the patience cap applies HERE only, so it can
         never block a legitimate confirmed re-call (step 2 runs first).

    "Has the user spoken" = newest user text changed OR user-turn count
    grew. Text alone misses a repeated "Yes."; count alone deadlocks after
    the summarizer shrinks old turns. Either signal is enough.
    """
    if call.action not in {n for n, _, c in gw.catalogue if c}:
        return None
    state, fp = gw.state, call.fingerprint

    # 1) Session grant: this target was already approved once. Repeats
    # (wording tweaks, content iterations on the same repo/playlist) ride
    # through silently; a different target is a new fingerprint and asks.
    if fp in state.granted:
        return None

    text, turns = _user_anchor(params)
    pending = state.pending.get(fp)  # (text, turn count) at ask time
    answered = pending is not None and (
        text != pending[0] or turns > pending[1]
    )
    confirmed = _truthy(call.arguments.get("confirmed"))
    logger.debug(
        f"confirm gate {gw.server}.{call.action} fp={fp} "
        f"confirmed={confirmed} pending={pending} answered={answered} "
        f"asks={state.asks.get(fp, 0)} text={text!r} turns={turns}"
    )

    # 2) Valid approval: consume it and grant the target for the session.
    if confirmed and answered:
        state.pending.pop(fp, None)
        state.granted.add(fp)
        if len(state.granted) > 64:  # per-session state, keep bounded
            state.granted.clear()
            state.granted.add(fp)
        return None

    # 3) Asked already, user has not replied: do not count, do not re-ask.
    if pending is not None and not answered:
        return (
            "CONFIRMATION_STILL_PENDING: you already asked the user and "
            "they have not answered yet. Wait for their reply; do not call "
            "this action again until they have spoken."
        )

    # 4) A (new) ask is needed. The patience cap applies here only. After
    # MAX_ASKS_PER_TARGET rounds for this target, refuse terminally. This
    # stops the ask-confirm-fail-ask-again loop (measured: 72
    # CONFIRMATION_REQUIRED and one question re-asked 29 times). A fresh
    # user message after the refusal re-opens one round — the user asked
    # for another go (anchored on the newest user message: summary-proof).
    blocked_at = state.ask_blocked.get(fp)
    if state.asks.get(fp, 0) >= MAX_ASKS_PER_TARGET:
        if blocked_at is not None and text != blocked_at:
            state.asks.pop(fp, None)
            state.ask_blocked.pop(fp, None)
        else:
            state.ask_blocked.setdefault(fp, text)
            return (
                f"You have already asked the user to approve "
                f"'{call.action}' {MAX_ASKS_PER_TARGET} times and it still "
                f"has not gone through. Stop asking and stop calling "
                f"{gw.server} tools for this. Tell the user plainly that "
                f"it is not working right now."
            )

    state.pending[fp] = (text, turns)
    state.asks[fp] = state.asks.get(fp, 0) + 1
    if len(state.pending) > 64:  # per-session state, keep bounded
        state.pending.clear()
        state.pending[fp] = (text, turns)
    return (
        f"CONFIRMATION_REQUIRED: '{call.action}' is a protected action. "
        f"Ask the user ONCE, out loud, to approve it (state the key "
        f"arguments). When they answer yes, call again with the same "
        f"target plus confirmed=true as a top-level field. Do not ask "
        f"again if they have already said yes."
    )


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


def _per_turn_cap(
    gw: "ServerGateway", params, call: _Call
) -> str | None:
    """Terminal refusal once an action ran PER_TURN_ACTION_CAPS times
    inside ONE user turn — None to proceed (and count the call).

    The rungs of the measured ladder each cost the user a second of
    silence: a rejected call re-issued with fixed arguments, a re-search
    after weak snippets. Guards are data, not prompt coaching — the cap
    stops the ladder and orders the model to answer from what it has.
    Runs LAST (after the confirm gate) so it counts only calls that
    actually execute; the budget reopens on a new user turn, anchored on
    the same (text, count) pair the confirm gate uses, so summarization
    cannot wedge it.
    """
    cap = PER_TURN_ACTION_CAPS.get(call.action)
    if cap is None:
        return None
    anchor = _user_anchor(params)
    if anchor != gw.state.action_anchor:
        gw.state.action_anchor = anchor
        gw.state.action_calls = {}  # new user turn: budget reopens
    made = gw.state.action_calls.get(call.action, 0)
    if made < cap:
        gw.state.action_calls[call.action] = made + 1
        return None
    return (
        f"'{call.action}' has already run {cap} times this turn. Do not "
        f"call it again. Answer from the results you already have, or "
        f"tell the user plainly what is missing."
    )


# ---- Execution and framing --------------------------------------------------


async def _run_tool(
    client: MCPClient, server: str, action: str, call_args: dict[str, Any]
) -> str:
    """One real MCP call; any failure becomes an 'Error: …' tool result.

    Every call logs its wall time: "tool tavily.tavily_search: 930 ms".
    That line is the ground truth for tool latency questions — measured
    against the LLM round trips around it, a searched answer's wait is
    almost never the search itself (tavily's engine reported 0.13 s on a
    turn whose retry ladder cost ~3.5 s in LLM re-fixes).
    """
    t0 = time.monotonic()
    try:
        result = await call_mcp_tool(client, action, call_args, server=server)
    except Exception as e:
        logger.warning(
            f"tool {server}.{action} FAILED after "
            f"{(time.monotonic() - t0) * 1000:.0f} ms: {e}"
        )
        return f"Error: {e}"
    took_s = time.monotonic() - t0
    log = logger.warning if took_s > 2.0 else logger.info
    log(f"tool {server}.{action}: {took_s * 1000:.0f} ms")
    return result


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
        # Server-side incompatibilities die here, before any guard or wire:
        # the model keeps passing what the schema told it was valid.
        call_args = _strip_incompatible(self.server, action, call_args)
        # Models sometimes nest `confirmed` inside `arguments`. Honor it as
        # the top-level flag and strip it so it never reaches the real MCP
        # call or the fingerprints.
        if "confirmed" in call_args:
            if _truthy(call_args.pop("confirmed")):
                arguments["confirmed"] = True
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
#             wins and is spoken unframed. Cheap rejections come first;
#             the confirm gate and the per-turn cap run LAST so the user
#             is only ever asked to approve a call that is allowed,
#             well-formed, not struck out and not already known to fail —
#             and even an approved call cannot loop past its per-turn
#             budget.
#
# Adding a rule = adding one function + one entry here.

GUARDS = (
    _deny_unallowed,
    _strikeout,
    _circuit_breaker,
    _validate_args,
    _confirm_gate,
    _per_turn_cap,
)


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
    present = []
    for server in sorted(servers):
        if mcp_clients.get(server) is None:
            logger.error(f"tool gateway '{server}': no MCP client, skipping")
        else:
            present.append(server)
    # Heal missing schemas concurrently and on a short leash: this runs on
    # every session start, and a backend that is still down would otherwise
    # hold the pipeline boot for its full connect timeout (30 s), once per
    # down server. A capture that misses the window is not cancelled (that
    # would abort a connect mid-handshake): it finishes in the background
    # and the NEXT session gets the exact argument hints.
    missing = [s for s in present if not SERVER_TOOL_SCHEMAS.get(s)]
    if missing:
        tasks = {
            asyncio.ensure_future(ensure_server_schema(s, mcp_clients[s]))
            for s in missing
        }
        _, late = await asyncio.wait(tasks, timeout=SCHEMA_HEAL_TIMEOUT_SECS)
        if late:
            _BACKGROUND.update(late)
            for task in late:
                task.add_done_callback(_BACKGROUND.discard)
            logger.warning(
                f"tool gateway: schema capture still running after "
                f"{SCHEMA_HEAL_TIMEOUT_SECS:.0f} s — starting without exact "
                f"argument hints for the slow server(s)"
            )

    schemas = []
    for server in present:
        client = mcp_clients[server]
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