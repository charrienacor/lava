"""Shared runtime state: live session counters, per-tab conversation history,
the TTS speech cache, and the Full-Duplex-Bench metric windows."""

import time

from web.core.config import HISTORY_TURNS, LLM_CHAIN

# Per-tab history (CONVERSATIONS): tapping the orb after an idle close
# resumes the same conversation.
CONVERSATIONS: dict[str, list[dict]] = {}
CONVERSATION_TTL = 60 * 60 * 24


STATE = {
    "sessions": 0,
    "tools": [],
    "model": LLM_CHAIN[0],
    "chain": LLM_CHAIN,
    "live_sessions": set(),
    "by_ws": {},
    # Spotify playback mirror (drives music-aware mic gating). Kept fresh by
    # the 1 Hz infra loop via the tools MCP's cached music_state tool.
    "music_playing": False,
    "music_seen_at": 0.0,
    "music_mode_until": 0.0,
    # Full-duplex overlap accounting (Full-Duplex-Bench style): backchannels
    # recognized while the bot talked (kept talking) vs real barge-ins (cut).
    "backchannels": 0,
    "barge_ins": 0,
    # Full-Duplex-Bench v1.5: per-scenario behavior counts plus rolling
    # t_stop / t_resp samples (ms). See classify_overlap() and fd_summary().
    "fd_behaviors": {
        "interruption": 0,
        "backchannel": 0,
        "talk_to_other": 0,
        "background": 0,
    },
    "fd_stop_ms": [],
    "fd_resp_ms": [],
}


def _fd_record(key: str, ms: float) -> None:
    """Append one t_stop / t_resp sample (rolling window of 200)."""
    buf = STATE[key]
    buf.append(round(ms))
    if len(buf) > 200:
        del buf[: len(buf) - 200]


def fd_summary() -> dict:
    """Rolling Full-Duplex-Bench metrics for /api/status and the bench."""

    def stat(buf: list[int]) -> dict | None:
        if not buf:
            return None
        xs = sorted(buf)
        return {
            "n": len(xs),
            "p50": xs[len(xs) // 2],
            "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
        }

    return {
        "behaviors": dict(STATE["fd_behaviors"]),
        "t_stop_ms": stat(STATE["fd_stop_ms"]),
        "t_resp_ms": stat(STATE["fd_resp_ms"]),
    }


# Text -> (pcm, sr) cache. Seeded with the greeting and apologies at startup;
# short repeated lines (acknowledgments, timer announcements) accumulate at
# runtime. FIFO eviction, capped.
SPEECH_CACHE: dict[str, tuple[bytes, int]] = {}
SPEECH_CACHE_MAX = 40
SPEECH_CACHE_MAX_CHARS = 80


def _prune_conversations() -> None:
    """Drop conversations older than CONVERSATION_TTL."""
    now = time.time()
    stale = [
        cid for cid, (_h, ts) in CONVERSATIONS.items() if now - ts > CONVERSATION_TTL
    ]
    for cid in stale:
        CONVERSATIONS.pop(cid, None)


def _trim_history(history: list[dict]) -> None:
    """Keep the system prompt plus the last HISTORY_TURNS exchanges."""
    if len(history) > 1 + HISTORY_TURNS * 2:
        del history[1 : -(HISTORY_TURNS * 2)]


def done_stats(stats: dict, *, ttfa_ms: float | None = None) -> dict:
    """Timing payload for a `done` frame (turns and the music fast path).

    `ttfa_ms` (time to first audio) is written by the Speaker for LLM turns;
    callers that speak without a Speaker pass their own measurement."""
    out = {
        "asr_ms": round(stats.get("asr_ms", 0)),
        "llm_ms": round(stats.get("llm_ms", 0)),
        "tts_ms": round(stats.get("tts_ms", 0)),
        "ttfa_ms": round(ttfa_ms if ttfa_ms is not None else stats.get("ttfa_ms", 0)),
        "total_ms": round(
            stats.get("asr_ms", 0) + stats.get("llm_ms", 0) + stats.get("tts_ms", 0)
        ),
    }
    # Full-Duplex-Bench v1.5: surface this turn's overlap metrics (present
    # only when this turn was the response to a barge-in).
    if stats.get("fd_stop_ms") is not None:
        out["fd_stop_ms"] = round(stats["fd_stop_ms"])
    if stats.get("fd_resp_ms") is not None:
        out["fd_resp_ms"] = round(stats["fd_resp_ms"])
    return out
