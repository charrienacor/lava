"""tool_gateway: index, identity binding, and the enforced confirm gate."""

import asyncio

import tool_gateway
from tool_gateway import _catalogue, _identity, _schema, gateway_schemas

import pytest


# --- catalogue / schema (pure, offline) -------------------------------------

def test_catalogue_flags():
    gh = dict((n, c) for n, _, c in _catalogue("github"))
    assert len(gh) == 10
    assert gh["merge_pull_request"] and gh["push_files"]
    assert gh["delete_file"]  # destructive: gated like the other writers
    assert not gh["search_repositories"]
    assert not gh["get_me"]  # read-only profile lookup, no gate
    sp = dict((n, c) for n, _, c in _catalogue("spotify"))
    assert sp["spotify_add_tracks"] and not sp["spotify_play"]
    assert all(not c for _, _, c in _catalogue("tavily"))


def test_schema_shape():
    schema = _schema("github", _catalogue("github"))
    d = schema.to_default_dict()
    assert d["name"] == "github"
    assert d["parameters"]["required"] == ["action"]
    # No hard enum: mid-session policy flips must stay callable. Enabled
    # actions are named in the description instead.
    action = d["parameters"]["properties"]["action"]
    assert "enum" not in action
    assert "merge_pull_request" in d["description"]
    assert "delete_file" in d["description"]  # enabled policy: advertised
    assert "confirmed" in d["parameters"]["properties"]


def test_schema_arg_hints_show_live_enums(monkeypatch):
    """Enum-valued params advertise their VALUES, not just names.

    With only names visible the model invents values ("time_range":
    "recent") and burns a turn on the server's rejection — the values are
    captured live from the server's schema (mcp_client._param_enums).
    """
    monkeypatch.setattr(
        tool_gateway, "SERVER_TOOL_SCHEMAS",
        {"tavily": {"tavily_search": {
            "required": ["query"],
            "properties": [
                "query", "topic", "time_range", "max_results",
                "include_images", "include_raw_content",
            ],
            "enums": {
                "topic": ["general", "news"],
                "time_range": ["day", "week", "month", "year"],
            },
        }}},
    )
    schema = _schema("tavily", [("tavily_search", "Search the web.", False)])
    d = schema.to_default_dict()
    assert "topic(general|news)" in d["description"]
    assert "time_range(day|week|month|year)" in d["description"]
    assert "query*" in d["description"]
    assert "max_results" in d["description"]
    assert "max_results(" not in d["description"]  # no enum -> no parens
    # include_* display toggles stay off the wire (token economy; naming
    # include_raw_content invites multi-k raw dumps into an 8k-TPM context)
    assert "include_images" not in d["description"]
    assert "include_raw_content" not in d["description"]


def test_catalogue_advertises_operating_knowledge():
    """The gateway advertises each action's FIRST description sentence, so
    multi-step flows must live THERE — not in the system prompt. Pinned:
    the github.io personal-site route (with the autoInit detail that makes
    push_files work on a fresh repo) and tavily's recency params."""
    gh = {n: s for n, s, _ in _catalogue("github")}
    assert ".github.io" in gh["get_me"] and "FIRST" in gh["get_me"]
    assert "autoInit" in gh["create_repository"]
    assert ".github.io" in gh["create_repository"]
    assert ".github.io" in gh["push_files"] and "raw text" in gh["push_files"]
    assert "user:<login>" in gh["search_repositories"]
    tv = {n: s for n, s, _ in _catalogue("tavily")}
    # tavily stays AS SHIPPED: the summary line is the stock description —
    # no extra prose. Recency params ride the LIVE enum hints
    # (test_schema_arg_hints_show_live_enums), and the search-first rule
    # lives in prompt.py's short awareness block — neither is "prompt
    # coaching" on the tool itself.
    assert tv["tavily_search"] == "Search the web for current information on any topic."
    assert tv["tavily_extract"] == "Extract content from URLs."
    # every advertised line stays within the cap (no mid-word truncation)
    for server in ("github", "tavily", "spotify", "time"):
        for _, summary, _ in _catalogue(server):
            assert len(summary) <= tool_gateway._CATALOGUE_MAX_DESC


def test_identity_binding():
    merge = {"owner": "a", "repo": "b", "pullNumber": 1}
    tweaked = {**merge, "commit_title": "new title"}
    assert _identity("github", "merge_pull_request", merge) == _identity(
        "github", "merge_pull_request", tweaked
    )
    assert _identity("github", "merge_pull_request", merge) != _identity(
        "github", "merge_pull_request", {**merge, "pullNumber": 2}
    )
    # push_files: file paths are target, contents are payload
    files_a = {"owner": "a", "repo": "b", "branch": "main",
               "files": [{"path": "x.py", "content": "1"}]}
    files_b = {**files_a, "files": [{"path": "x.py", "content": "2"}]}
    assert _identity("github", "push_files", files_a) == _identity(
        "github", "push_files", files_b
    )
    # unmapped gated action: strict full-args fallback
    assert _identity("nowhere", "act", {"a": 1}) != _identity("nowhere", "act", {"a": 2})


# --- handler flow (confirm gate enforced; no network) ------------------------

class FakeLLM:
    def __init__(self):
        self.fns = {}

    def register_function(self, name, fn, *, cancel_on_interruption=True):
        self.fns[name] = fn
        self.cancel_on_interruption = cancel_on_interruption


class Params:
    def __init__(self, arguments, messages):
        self.arguments = arguments
        self.context = type("Ctx", (), {"messages": messages})()
        self.captured = None

    async def result_callback(self, text):
        self.captured = text


@pytest.fixture()
def gh_handler(monkeypatch):
    async def fake_call(client, name, args, server=None):
        return f"executed:{name}"

    monkeypatch.setattr(tool_gateway, "call_mcp_tool", fake_call)
    llm = FakeLLM()
    asyncio.run(gateway_schemas(llm, {"github": object()}, {"github"}))
    return llm.fns["github"]


def test_gateway_is_async_tool():
    """The gateway registers as an async tool (cancel_on_interruption=False).

    A barge-in mid-call used to cancel the call outright: the context got
    'CANCELLED', the UI's command toast never resolved, and a completed
    command's result was lost (measured: spotify_play's ~8 s playback
    verification was interrupted again and again). As an async tool the
    result always lands.
    """
    llm = FakeLLM()
    asyncio.run(gateway_schemas(llm, {"github": object()}, {"github"}))
    assert llm.cancel_on_interruption is False


def call(handler, arguments, messages):
    async def run():
        p = Params(arguments, messages)
        await handler(p)
        return p.captured
    return asyncio.run(run())


def test_gate_full_protocol(gh_handler):
    merge = {"action": "merge_pull_request",
             "arguments": {"owner": "a", "repo": "b", "pullNumber": 1}}
    msgs = [{"role": "user", "content": "merge PR one"}]

    assert "CONFIRMATION_REQUIRED" in call(gh_handler, merge, msgs)
    assert "CONFIRMATION_STILL_PENDING" in call(
        gh_handler, {**merge, "confirmed": True}, msgs)
    msgs.append({"role": "user", "content": "yes"})
    out = call(gh_handler, {**merge, "confirmed": True}, msgs)
    assert "executed:merge_pull_request" in out and "<tool_result" in out


def test_gate_never_asked_and_target_change(gh_handler):
    msgs = [{"role": "user", "content": "do it"},
            {"role": "user", "content": "yes"}]
    args = {"action": "merge_pull_request",
            "arguments": {"owner": "a", "repo": "b", "pullNumber": 1}}
    assert "CONFIRMATION_REQUIRED" in call(
        gh_handler, {**args, "confirmed": True}, msgs)
    # approved PR 1, then swap target → refused
    msgs.append({"role": "user", "content": "ok"})
    call(gh_handler, args, msgs)
    swapped = {"action": "merge_pull_request",
               "arguments": {"owner": "a", "repo": "b", "pullNumber": 2}}
    assert "CONFIRMATION_REQUIRED" in call(
        gh_handler, {**swapped, "confirmed": True}, msgs)


def test_gate_survives_summarization(gh_handler):
    """The measured deadlock: auto-summarization compressed the context
    between the ask and the answer, shrinking the user-turn count, and
    the old count comparison bounced four spoken approvals with
    CONFIRMATION_STILL_PENDING. The newest user MESSAGE is the anchor —
    it survives summarization, so approval lands."""
    push = {"action": "push_files",
            "arguments": {"owner": "a", "repo": "b", "branch": "main",
                          "files": [{"path": "index.html",
                                     "content": "<html>"}]}}
    msgs = [{"role": "user", "content": "push it to my repo"}]
    assert "CONFIRMATION_REQUIRED" in call(gh_handler, push, msgs)

    # the summarizer compresses the old turns away; the user answers after
    msgs[:] = [
        {"role": "assistant",
         "content": "Conversation summary: user wants index.html pushed."},
        {"role": "user", "content": "yes"},
    ]
    out = call(gh_handler, {**push, "confirmed": True}, msgs)
    assert "executed:push_files" in out and "<tool_result" in out


def test_gate_grants_target_for_the_session(gh_handler):
    """One spoken approval covers same-target repeats for the rest of the
    session — iterating on a just-approved page never re-asks — while a
    DIFFERENT target (another repo) is a new fingerprint and asks again."""
    push = {"action": "push_files",
            "arguments": {"owner": "a", "repo": "b", "branch": "main",
                          "files": [{"path": "index.html",
                                     "content": "v1"}]}}
    msgs = [{"role": "user", "content": "push the page"}]
    assert "CONFIRMATION_REQUIRED" in call(gh_handler, push, msgs)
    msgs.append({"role": "user", "content": "yes"})
    assert "executed:push_files" in call(
        gh_handler, {**push, "confirmed": True}, msgs)

    # content iteration, same target: no handshake, rides through
    iterate = {"action": "push_files",
               "arguments": {"owner": "a", "repo": "b", "branch": "main",
                             "files": [{"path": "index.html",
                                        "content": "v2"}]}}
    assert "executed:push_files" in call(gh_handler, iterate, msgs)

    # a different repo is a new target: the full handshake again
    other = {"action": "push_files",
             "arguments": {"owner": "a", "repo": "c", "branch": "main",
                           "files": [{"path": "index.html",
                                      "content": "v1"}]}}
    assert "CONFIRMATION_REQUIRED" in call(gh_handler, other, msgs)


def test_ungated_runs_immediately_and_unknown_denied(gh_handler):
    msgs = [{"role": "user", "content": "hi"}]
    out = call(gh_handler, {"action": "search_repositories",
                            "arguments": {"query": "x"}}, msgs)
    assert "executed:search_repositories" in out
    assert "unknown action" in call(gh_handler, {"action": "banana",
                                                 "arguments": {}}, msgs)


def test_tavily_capped_per_user_turn(monkeypatch):
    """One question, at most PER_TURN_ACTION_CAPS searches: the measured
    ladder (a rejected call re-issued with fixed arguments, a re-search
    after weak snippets) ends in a terminal refusal that orders the model
    to answer from what it has. A NEW user turn reopens the budget."""
    async def fake_call(client, name, args, server=None):
        return f"executed:{name}"

    monkeypatch.setattr(tool_gateway, "call_mcp_tool", fake_call)
    llm = FakeLLM()
    asyncio.run(gateway_schemas(llm, {"tavily": object()}, {"tavily"}))
    h = llm.fns["tavily"]
    msgs = [{"role": "user", "content": "who won the midterms"}]
    args = {"action": "tavily_search", "arguments": {"query": "x"}}

    cap = tool_gateway.PER_TURN_ACTION_CAPS["tavily_search"]
    for _ in range(cap):  # the ladder itself fits the budget
        out = call(h, args, msgs)
        assert "executed:tavily_search" in out and "<tool_result" in out
    # one past the cap: terminal, unframed, nothing executed
    out = call(h, args, msgs)
    assert "already run" in out and "<tool_result" not in out
    assert "results you already have" in out
    # the user spoke again: budget reopens
    msgs.append({"role": "user", "content": "and the senate?"})
    out = call(h, args, msgs)
    assert "executed:tavily_search" in out
    # uncapped actions (e.g. tavily_extract) are untouched
    out = call(h, {"action": "tavily_extract",
                   "arguments": {"urls": ["https://x"]}}, msgs)
    assert "executed:tavily_extract" in out


def test_tavily_country_fast_incompatibility_stripped(monkeypatch):
    """The fix-retry ladder, killed at the source. Tavily's country
    argument failed in TWO classes in one evening (rejected with
    fast/ultra-fast depth; 'US'/'USA' rejected as invalid names), each
    guess burning a call + an LLM round trip. The gateway strips it
    unconditionally — and hides it from the advertised args, so the
    model never learns to want it."""
    captured = {}

    async def fake_call(client, name, args, server=None):
        captured.clear()
        captured.update(args)
        return "executed"

    monkeypatch.setattr(tool_gateway, "call_mcp_tool", fake_call)
    llm = FakeLLM()
    asyncio.run(gateway_schemas(llm, {"tavily": object()}, {"tavily"}))
    h = llm.fns["tavily"]
    msgs = [{"role": "user", "content": "who won the midterms"}]

    # stripped at fast depth, at basic depth, and with any value the
    # model guesses ('US', 'USA', full names — all of it). Each call
    # gets a fresh turn anchor: the per-turn cap (3) would otherwise
    # deny the fourth search of a single test "turn".
    for i, depth in enumerate(("fast", "ultra-fast", "basic", None)):
        args = {"action": "tavily_search",
                "arguments": {"query": "x", "country": "US"}}
        if depth:
            args["arguments"]["search_depth"] = depth
        out = call(h, args, msgs)
        assert "executed" in out and "<tool_result" in out
        assert "country" not in captured, f"country survived at depth={depth}"
        assert captured.get("query") == "x"
        msgs.append({"role": "user", "content": f"and again {i}"})

    # and it is no longer advertised: the schema the model sees must not
    # name the argument the gateway will silently remove
    d = tool_gateway._schema("tavily", _catalogue("tavily")).to_default_dict()
    assert "country" not in d["description"]


def test_disabled_tool_is_recognized_and_live_flips(monkeypatch, tmp_path):
    """Policy is live: a disabled tool is named (not 'unknown'), and flipping
    enabled = true mid-session lets the SAME session run it."""
    toml = tmp_path / "spotify.toml"
    toml.write_text(
        '[[tool]]\nname = "spotify_shuffle"\ndescription = "Shuffle playback. "\nenabled = false\n'
    )
    monkeypatch.setattr(tool_gateway, "TOOL_DIR", tmp_path)

    async def fake_call(client, name, args, server=None):
        return f"executed:{name}"

    monkeypatch.setattr(tool_gateway, "call_mcp_tool", fake_call)
    llm = FakeLLM()
    asyncio.run(tool_gateway.gateway_schemas(llm, {"spotify": object()}, {"spotify"}))
    h = llm.fns["spotify"]
    msgs = [{"role": "user", "content": "shuffle"}]

    # disabled: recognized as a real tool, with the fix spelled out
    out = call(h, {"action": "spotify_shuffle", "arguments": {}}, msgs)
    assert "DISABLED" in out and "spotify_shuffle" in out
    assert "enabled = true" in out

    # the user flips the TOML mid-session -> same handler picks it up,
    # since policy is re-read on every call
    toml.write_text(
        '[[tool]]\nname = "spotify_shuffle"\ndescription = "Shuffle playback. "\nenabled = true\n'
    )
    out2 = call(h, {"action": "spotify_shuffle", "arguments": {}}, msgs)
    assert "executed:spotify_shuffle" in out2

    # and a genuinely unknown action is still told apart from a disabled one
    out3 = call(h, {"action": "banana", "arguments": {}}, msgs)
    assert "unknown action" in out3
