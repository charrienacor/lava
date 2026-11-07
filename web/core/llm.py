"""Groq LLM: the streaming completion with its tool loop, the model fallback
chain with per-model cooldowns, and the post-ASR proofread pass."""

import asyncio
import json
import os
import re
import time

from loguru import logger
from openai import AsyncOpenAI

from web.core.config import GROQ_API_KEY, LLM_CHAIN
from web.core.mcp_client import exec_tool
from web.core.state import STATE

# LLM (Groq, streaming, with the tool loop)

llm = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    timeout=30.0,
)

# 0.8 (down from 1.0): a voice agent pays for every quirky generation with
# several seconds of TTS, so consistency beats sampling diversity.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
# Bounds a runaway generation; voice replies are two short sentences.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))

# Post-ASR correction pass. Groq's Whisper endpoint rejects `initial_prompt`
# (400 unknown param), so the model can't be biased toward the user's
# vocabulary at the API level. Instead, a fast LLM call fixes obvious
# mis-hearings (proper nouns, homophones) using the recent conversation as
# context. It never changes meaning and falls back to the raw transcript on
# any error, so it can only help. Set ASR_CORRECT=false to disable.
ASR_CORRECT = os.getenv("ASR_CORRECT", "true").lower() in ("1", "true", "yes", "on")
# Skip the pass for very short turns ("hi", "next") where there is nothing to
# fix and a call would only add latency.
ASR_CORRECT_MIN_WORDS = int(os.getenv("ASR_CORRECT_MIN_WORDS", "3"))
# How many recent messages of context to hand the corrector.
ASR_CORRECT_CTX = int(os.getenv("ASR_CORRECT_CTX", "6"))
# Hard cap so a slow corrector can never stall a turn; on timeout the raw
# transcript is used.
ASR_CORRECT_TIMEOUT = float(os.getenv("ASR_CORRECT_TIMEOUT", "2.5"))
# But the turn never waits this long for the proofread regardless of the
# timeout above: the transcript is shown instantly and the LLM starts on the
# raw text after the grace expires, so a throttled corrector costs ~nothing
# instead of up to ASR_CORRECT_TIMEOUT of dead air. 0.5 s is the sweet spot:
# a healthy corrector answers in ~300-400 ms, and the common case in the
# logs was a full-grace timeout that bought nothing - every extra 100 ms
# here is dead air on the critical path of every turn.
ASR_CORRECT_GRACE = float(os.getenv("ASR_CORRECT_GRACE_SECS", "0.5"))
# Which model does the proofread. Defaults to the first NON-reasoning model in
# the chain: a simple correction task should not pay the reasoning tax (or
# burn the daily quota) of the heavy primary. Override to force one.
ASR_CORRECT_MODEL = os.getenv("ASR_CORRECT_MODEL", "")
# Whisper's own confidence signal (avg_logprob, lower = less sure) decides
# WHETHER to proofread at all. Proofreading every turn cost ~500 ms of dead
# air on the critical path and, per the logs, was a net loss (more timeouts
# than fixes) because a healthy transcript needs no repair. Run the pass only
# when Whisper itself is unsure (avg_logprob at or below this); confident
# transcripts skip it for free. Set to a very negative number to always
# proofread, or 0 to never.
ASR_PROOFREAD_LOGPROB_MAX = float(os.getenv("ASR_PROOFREAD_LOGPROB_MAX", "-0.7"))

RETRYABLE = (429, 500, 502, 503, 524, 529)
# Word-bounded, so "4290 chars" or "500ms" do not match.
_RETRYABLE_RE = re.compile(rf"\b({'|'.join(map(str, RETRYABLE))})\b")


def _is_throttled(e: Exception) -> bool:
    # openai exceptions carry the HTTP status on .status_code - use it
    # instead of string-matching, which false-positives on any error text
    # containing a code (e.g. a 500-char limit in the message).
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc in RETRYABLE
    return bool(_RETRYABLE_RE.search(str(e)))


# A model that just refused a request (429) refuses the next one too -
# Groq's free-tier quota is per model. Retrying it every turn paid a full
# round trip + a 429 + a 1 s sleep before falling back, on EVERY turn.
# Putting the model on cooldown makes the fallback instant for the rest of
# the outage window; the last model is always tried even while cooling down
# (a 429 from it is better than no brain at all).
_model_cooldown: dict[str, float] = {}
LLM_COOLDOWN_SECS = float(os.getenv("LLM_COOLDOWN_SECS", "60"))


async def _open_stream(start: int, msgs: list[dict]):
    """Open a streaming completion, walking the model chain.

    On a 429 the next model in LLM_CHAIN still has a fresh quota (Groq's
    daily quota is per model) and the refused model goes on cooldown so
    the next turn skips it without paying for the refusal again.
    Returns (stream, model_index_used)."""
    last_err: Exception | None = None
    now = time.monotonic()
    for i in range(start, len(LLM_CHAIN)):
        model = LLM_CHAIN[i]
        if _model_cooldown.get(model, 0.0) > now and i + 1 < len(LLM_CHAIN):
            logger.debug(f"[web] {model} in cooldown - skipping")
            continue
        extra = (
            {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None
        )
        try:
            stream = await llm.chat.completions.create(
                model=model,
                messages=msgs,
                tools=STATE["tools"] or None,
                stream=True,
                temperature=LLM_TEMPERATURE,
                top_p=0.95,
                max_tokens=LLM_MAX_TOKENS,
                extra_body=extra,
            )
            if i > start:
                logger.info(f"[web] LLM fell back to {model}")
            return stream, i
        except Exception as e:
            last_err = e
            if not _is_throttled(e):
                raise
            if i + 1 < len(LLM_CHAIN):
                _model_cooldown[model] = time.monotonic() + LLM_COOLDOWN_SECS
                logger.warning(
                    f"[web] {model} rate-limited "
                    f"(cooldown {LLM_COOLDOWN_SECS:.0f} s); trying "
                    f"{LLM_CHAIN[i + 1]}"
                )
            else:
                logger.error(f"[web] all LLM models rate-limited: {str(e)[:200]}")
    assert last_err is not None
    raise last_err


async def llm_stream(messages: list[dict], notify=None):
    """Yield ('text', str) and ('tool_call', dict) tuples.

    Runs the tool loop: execute any tool calls, feed results back, keep
    streaming until prose. `notify` (optional) is forwarded to exec_tool so
    the bridge can push command notifications to the UI as tools run."""
    msgs = [dict(m) for m in messages]
    model_idx = 0
    while True:
        stream, model_idx = await _open_stream(model_idx, msgs)
        text_buf = ""
        pending: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_buf += delta.content
                yield "text", delta.content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    e = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        e["id"] = tc.id
                    if tc.function and tc.function.name:
                        e["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        e["args"] += tc.function.arguments
        if pending:
            # Parse every streamed call first, then run them concurrently:
            # models occasionally request two independent tools (time +
            # weather), and running them serially paid both round trips.
            calls: list[tuple[dict, dict]] = []
            for idx in sorted(pending):
                e = pending[idx]
                try:
                    args = json.loads(e["args"]) if e["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append((e, args))
            for e, args in calls:
                yield "tool_call", {"name": e["name"], "arguments": args}
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": e["id"],
                            "type": "function",
                            "function": {"name": e["name"], "arguments": e["args"]},
                        }
                        for e, _ in calls
                    ],
                }
            )
            results = await asyncio.gather(
                *(exec_tool(e["name"], args, notify=notify) for e, args in calls)
            )
            for (e, _), result in zip(calls, results):
                msgs.append(
                    {"role": "tool", "tool_call_id": e["id"], "content": result}
                )
            continue
        return


def _recent_context(history: list[dict]) -> str:
    """Compact transcript of the last few turns, for the ASR corrector."""
    msgs = [m for m in history if m.get("role") in ("user", "assistant")][
        -ASR_CORRECT_CTX:
    ]
    lines = []
    for m in msgs:
        role = "user" if m["role"] == "user" else "lava"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _corrector_models() -> list[str]:
    """Ordered models for the proofread: the configured corrector first, then
    the rest of the chain as fallback (deduplicated)."""
    primary = ASR_CORRECT_MODEL or next(
        (m for m in LLM_CHAIN if not m.startswith("openai/gpt-oss")), LLM_CHAIN[0]
    )
    models = [primary] + [m for m in LLM_CHAIN if m != primary]
    return models


async def _correct_asr(
    text: str,
    history: list[dict],
    stats: dict,
    timeout: float | None = None,
    avg_logprob: float | None = None,
) -> str:
    """Fix obvious ASR mis-hearings using recent conversation context.

    Whisper (via Groq) can't take an `initial_prompt`, so proper nouns and
    unusual words get snapped to phonetically-similar common words ("Clara
    Benin" -> "Sarah Benin"). This fast LLM pass repairs those without
    changing what the user meant. It is conservative: if the transcript
    already looks right it is returned unchanged, and any error or timeout
    falls back to the original text, so the pass can only ever help.

    It runs only when Whisper's own confidence (avg_logprob) says the
    transcript is doubtful - proofreading a confident transcript is pure
    latency (see ASR_PROOFREAD_LOGPROB_MAX)."""
    if not ASR_CORRECT or not text:
        return text
    if len(text.split()) < ASR_CORRECT_MIN_WORDS:
        return text
    if avg_logprob is not None and avg_logprob > ASR_PROOFREAD_LOGPROB_MAX:
        logger.debug(f"[asr-correct] skipped (confident: logprob {avg_logprob:.2f})")
        return text
    ctx = _recent_context(history)
    system = (
        "You proofread a voice transcript. The text is what a speech "
        "recognizer heard; it may contain mis-heard words, especially names, "
        "places, brands and spelled-out words. Using the recent conversation "
        "as context, fix only obvious recognition errors. Keep the user's "
        "exact meaning, tone and brevity. Never answer the user, never add or "
        "remove words, never fix grammar they actually used, never translate. "
        "Always output a non-empty corrected version of the transcript - if "
        "it already looks correct, output it exactly as-is. Output ONLY the "
        "corrected text: no quotes, no labels, no explanation."
    )
    user_msg = f"Recent conversation:\n{ctx or '(none)'}\n\nTranscript to fix:\n{text}"
    t0 = time.perf_counter()
    effective = timeout or ASR_CORRECT_TIMEOUT
    try:
        return await asyncio.wait_for(
            _correct_asr_inner(text, system, user_msg, stats, t0),
            timeout=effective,
        )
    except TimeoutError:
        logger.warning(f"[asr-correct] timed out after {effective:.1f}s; using raw")
        return text


async def _correct_asr_inner(
    text: str, system: str, user_msg: str, stats: dict, t0: float
) -> str:
    last_err: Exception | None = None
    for model in _corrector_models():
        extra = (
            {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None
        )
        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=120,
                extra_body=extra,
            )
            fixed = (resp.choices[0].message.content or "").strip()
            stats["asr_correct_ms"] = (time.perf_counter() - t0) * 1000
            if not fixed:
                return text
            # Sanity: a recognition fix should not balloon the text.
            if len(fixed) > max(len(text) * 3, len(text) + 40):
                logger.info(f"[asr-correct] rejected (too long): {fixed!r}")
                return text
            if fixed != text:
                logger.info(f"[asr-correct] {text!r} -> {fixed!r}")
            return fixed
        except Exception as e:
            last_err = e
            if not _is_throttled(e):
                break
            logger.warning(f"[asr-correct] {model} rate-limited; trying next")
            await asyncio.sleep(0.5)
    logger.debug(f"[asr-correct] passed through unchanged ({last_err})")
    return text
