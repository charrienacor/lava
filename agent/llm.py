"""LLM service — Groq GPT-OSS-120B (streaming) with a GPT-OSS-20B fallback.

The free tier's 8k TPM makes the 120b saturate in bursts: the OpenAI SDK
retries 429s silently with backoff and the user hears a long mute. Two layers
handle that:

- visibility: every 429 on the wire fires on_rate_limit listeners (the UI
  toast), and a fallback switch fires on_fallback_model listeners;
- resilience: when the primary model fails after the SDK's own retries
  (429 / 5xx / transport), requests switch to the small model for a cooldown
  window — the 20b has far higher limits, so the conversation keeps flowing.
  The window is sticky (subsequent requests go straight to the fallback) and
  the primary is probed again when it lapses.
"""

import asyncio
import os
import time
from typing import Awaitable, Callable

import openai
from loguru import logger
from pipecat.services.groq.llm import GroqLLMService

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
FALLBACK_COOLDOWN_SECS = 60.0

# Errors a different model can plausibly survive. Deliberately excluded:
# auth errors (a key problem affects both models) and CancelledError
# (barge-in — must propagate untouched).
_FALLBACK_ERRORS = (
    openai.RateLimitError,       # 429, after the SDK's own retries
    openai.InternalServerError,  # 5xx
    openai.APIConnectionError,   # transport
    openai.APITimeoutError,      # SDK-level timeout
    asyncio.TimeoutError,        # pipecat-level timeout, if enabled
)

# Rate-limit visibility: the OpenAI SDK retries 429s silently with backoff
# (honoring Retry-After) — the user just experiences a long mute. Listeners
# registered here fire the moment a 429 hits the wire, so the UI can say
# "rate limited" while the retry is still running. Registry is plain and
# host-agnostic: ui/server.py subscribes a broadcast to its sessions.
RateLimitListener = Callable[[], Awaitable[None] | None]
_RATE_LIMIT_LISTENERS: list[RateLimitListener] = []


def on_rate_limit(listener: RateLimitListener) -> None:
    """Subscribe a listener invoked on every Groq 429 seen on the wire."""
    _RATE_LIMIT_LISTENERS.append(listener)


async def _notify_rate_limit() -> None:
    for listener in _RATE_LIMIT_LISTENERS:
        try:
            result = listener()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass  # a broken listener must never break the request stream


# Fallback visibility: fired when requests switch to the small model.
FallbackListener = Callable[[], Awaitable[None] | None]
_FALLBACK_LISTENERS: list[FallbackListener] = []


def on_fallback_model(listener: FallbackListener) -> None:
    """Subscribe a listener invoked when the fallback model takes over."""
    _FALLBACK_LISTENERS.append(listener)


async def _notify_fallback() -> None:
    for listener in _FALLBACK_LISTENERS:
        try:
            result = listener()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (add it to .env)")
    return key


def groq_model() -> str:
    return os.environ.get("GROQ_LLM_MODEL", DEFAULT_LLM_MODEL)


def fallback_model() -> str:
    """The small model that takes over when the primary is unavailable."""
    return os.environ.get("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)


def create_llm(system_instruction: str | None = None) -> GroqLLMService:
    # max_retries rides through to the OpenAI-compatible client: the SDK
    # retries 429/5xx with backoff honoring Retry-After. Two is the right
    # budget for a voice agent — quick retries soak the transient blips, and
    # a sustained 429 surfaces to the model-fallback wrapper early instead of
    # stretching into a long mute (4 retries on the 8k-TPM free tier bought
    # ~11 s of silence before the first token).
    #
    # reasoning_effort="low": GPT-OSS is a reasoning model, and its thinking
    # ran 0.2–5.1 s before the first speakable token in real turns — TTFA
    # pays all of it. A voice agent needs tool sense, not essays; "low" keeps
    # tool planning while cutting most of the think time (override with
    # LLM_REASONING_EFFORT if a task ever needs deeper reasoning).
    #
    # system_instruction: the persona lives here, on the service — pipecat
    # 1.9 deprecated a "system" message in LLMContext (removed in 2.0), and
    # the service-level instruction is where the framework composes its own
    # guidance around it.
    llm = GroqLLMService(
        api_key=groq_api_key(),
        settings=GroqLLMService.Settings(
            model=groq_model(),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", "low"),
            system_instruction=system_instruction,
        ),
        max_retries=2,
    )
    _wire_rate_limit_watch(llm)
    return _wire_model_fallback(llm, groq_model(), fallback_model())


async def keep_warm(llm: GroqLLMService) -> None:
    """Open the LLM's HTTPS connection now and keep it open for the session.

    Each session builds a fresh client, so its first turn paid DNS + TCP +
    TLS to Groq before the first token could arrive; and a connection idle
    between turns can be closed by Groq's edge, charging the handshake
    again. A models.list() at session start and every LLM_KEEPALIVE_SECS
    after (default 45; 0 = warm once, no pings) keeps the socket hot. The
    pings use no tokens. Run as a task; the session cancels it on exit.
    """
    interval = float(os.environ.get("LLM_KEEPALIVE_SECS", "45"))
    while True:
        try:
            await llm._client.models.list()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"LLM keep-warm ping failed: {e}")
        if interval <= 0:
            return
        await asyncio.sleep(interval)


def _wire_rate_limit_watch(llm: GroqLLMService) -> GroqLLMService:
    """Fire _notify_rate_limit on every 429 the OpenAI client receives.

    The hook hangs off the underlying httpx client (openai SDK keeps it at
    ``._client``) and only inspects the status line — it never touches the
    body, so streaming responses are unaffected. Every retry attempt fires
    separately; listeners throttle.
    """
    try:
        httpx_client = llm._client._client

        async def _on_response(response):
            if getattr(response, "status_code", None) == 429:
                await _notify_rate_limit()

        httpx_client.event_hooks.setdefault("response", []).append(_on_response)
    except Exception:
        pass  # monitoring is best-effort; never block service creation
    return llm


def _model_fallback_wrapper(
    original, state: dict, primary: str, fallback: str
):
    """Wrap the SDK's create() with a sticky switch to the fallback model.

    state["until"] is the monotonic time until which requests skip the
    primary entirely: once a failure switches models, the cooldown keeps the
    next turns off the saturated model instead of paying its retry backoff
    every time. When the window lapses, the primary is probed again.

    Malformed tool-call completions (Groq rejects the whole request with
    "Tool call validation failed" when the model invents a tool name, e.g.
    `spotify.get_devices`) are retried ONCE — a resample usually produces a
    correctly named call. Two in a row raise, as any other real error.
    """
    async def create(**kwargs):
        if time.monotonic() < state["until"] and kwargs.get("model") == primary:
            kwargs["model"] = fallback
        for attempt in (0, 1):
            try:
                return await original(**kwargs)
            except _FALLBACK_ERRORS:
                if kwargs.get("model") != primary:
                    raise  # the fallback itself failed — a genuine outage
                state["until"] = time.monotonic() + FALLBACK_COOLDOWN_SECS
                await _notify_fallback()
                kwargs["model"] = fallback
            except openai.BadRequestError as e:
                malformed = "tool call validation failed" in str(e).lower()
                if attempt == 1 or not malformed:
                    raise
                # resample once: same request, fresh completion
        raise RuntimeError("unreachable: completion loop exhausted")

    return create


def _wire_model_fallback(llm: GroqLLMService, primary: str, fallback: str) -> GroqLLMService:
    """Install the fallback wrapper on the SDK client (covers every call site:
    streaming chat, timeout-retry, and the summarizer)."""
    if primary == fallback:
        return llm
    try:
        state = {"until": 0.0}
        llm._model_fallback_state = state
        completions = llm._client.chat.completions
        completions.create = _model_fallback_wrapper(
            completions.create, state, primary, fallback
        )
    except Exception:
        pass  # resilience is best-effort; never block service creation
    return llm
