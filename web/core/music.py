"""Spotify music commands: the deterministic fast-path parser (no LLM) and the
voice acknowledgments, for sub-second control while music plays."""

import json
import re
import time

from aiohttp import web
from loguru import logger

from web.core.config import FAST_MUSIC_COMMANDS, MUSIC_MODE_GRACE
from web.core.mcp_client import exec_tool
from web.core.speech import send, speak_text
from web.core.state import STATE, _trim_history, done_stats

# Per-turn override, appended ONLY while music plays (see run_turn). It exists
# to settle a direct conflict with the base system prompt, which says "when
# unsure, treat a plausible direct question or request as real and answer it."
# That generosity is right for normal talk but is exactly what made lyric
# fragments ("I don't think much.") get chatty replies during playback. So this
# note flips the burden of proof for the music-mode turn: default to SKIP, and
# answer only what is unmistakably a real question or request. It is scoped to
# this turn (swept in run_turn's finally), so non-music behavior is untouched.
MUSIC_NOTE = (
    "OVERRIDE FOR THIS TURN ONLY: Spotify music is playing out loud in this "
    "room and its vocals reach the microphone, so most of what you hear "
    "right now is the song, not the user. This reverses your usual "
    "'when unsure, treat it as real and answer it' default: while music "
    "plays, treat an utterance as the music UNLESS it is clearly a direct "
    "question or request to you. Reply SKIP (and nothing else) to anything "
    "that is or could be song lyrics, a statement, a track or artist name, "
    "or a fragment - even if it reads like a plausible sentence. Do answer "
    "an unambiguous direct question ('what time is it?', 'what's the "
    "weather?', 'who sings this song?', 'what is playing?') or a clear "
    "request. Plain music commands (stop, pause, next, louder, volume ...) "
    "are normally handled before you reach them; if one slips through, "
    "honor it. Never call music_play for a lyric or song-name fragment."
)


# --- deterministic music commands (fast path) --------------------------------
# While music plays, control words ("stop", "next", "louder", "volume 40")
# are routed straight to their tool, bypassing the LLM: sub-second acks, and
# the lyrics-vs-command ambiguity never reaches a model that once PLAYED a
# phantom request. Matched strictly enough that ordinary lyrics never hit it.

_MUSIC_FILLERS = (
    "hey",
    "ok",
    "okay",
    "please",
    "could",
    "can",
    "would",
    "will",
    "you",
    "just",
    "kindly",
    "lava",
    "the",
)
_MUSIC_WORDS = re.compile(
    r"\b(music|song|songs|spotify|track|tracks|album|playlist|shuffle|repeat|"
    r"rewind|skip|pause|resume|unpause|play|playing|volume|louder|quieter|"
    r"next|previous|queue|devices?)\b",
    re.IGNORECASE,
)
# Phrases that look like commands but belong to other tools/the LLM.
_MUSIC_NOT_OURS = re.compile(
    r"\b(timer|timers|remind(?:er)?|alarm|notes?|weather|time)\b", re.IGNORECASE
)


def parse_music_command(text: str) -> dict | None:
    """Map a spoken music command -> {"tool", "args"} for the tools MCP.

    Returns None for anything that is not a clean music control. Never
    touches timers/notes/etc (see _MUSIC_NOT_OURS) - the LLM keeps those."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    while words and words[0] in _MUSIC_FILLERS:
        words.pop(0)
    while words and words[-1] in _MUSIC_FILLERS:
        words.pop()
    t = " ".join(words)
    if not t:
        return None
    # Restart first: its triggers contain "time"/"song" which NOT_OURS would
    # otherwise block ("play it one more time").
    if re.match(
        r"^(restart|rewind|start (this )?(song )?(over|again|from the beginning)"
        r"|beginning|play it (again|one more time|from the (top|start|beginning))"
        r"|one more time)\b",
        t,
    ):
        return {"tool": "music_restart", "args": {}}
    if _MUSIC_NOT_OURS.search(t):
        return None
    if len(t) > 80:
        return None
    num = re.search(r"\b\d{1,3}\b", t)

    # now playing / devices --------------------------------------------
    if re.match(
        r"^(what(?:'?s| is| s| are)?(?: (?:currently|right now))?"
        r"(?: playing| song|track)"
        r"|which (?:song|track)|now playing|tell me the (?:song|track)|"
        r"this (?:song|track))\b",
        t,
    ) or re.match(
        r"^(what(?:'?s| is| s| are)? (?:this|that) "
        r"(?:song|track))\b",
        t,
    ):
        return {"tool": "music_now_playing", "args": {}}
    if re.search(r"\bdevices?\b", t):
        return {"tool": "music_devices", "args": {}}

    # stop / pause -------------------------------------------------------
    if re.match(r"^(stop|pause)\b", t):
        return {"tool": "music_pause", "args": {}}

    # shuffle / repeat ---------------------------------------------------
    if re.match(r"^shuffl(e|es|ing)?\b", t):
        state = (
            "on"
            if re.search(r"\bon\b", t)
            else "off"
            if re.search(r"\boff\b", t)
            else ""
        )
        return {"tool": "music_shuffle", "args": {"state": state}}
    if re.match(r"^(repeat|loop)\b", t):
        mode = (
            "off"
            if re.search(r"\b(off|stop|cancel|no)\b", t)
            else "track"
            if re.search(r"\b(song|track|this|current|one)\b", t)
            else "context"
            if re.search(r"\b(album|playlist|queue|context|list)\b", t)
            else ""
        )
        return {"tool": "music_repeat", "args": {"mode": mode}}

    # previous -----------------------------------------------------------
    if re.match(r"^(previous|prev|last(?: song| one| track)?|go back|back)\b", t):
        return {"tool": "music_skip", "args": {"direction": "previous"}}

    # next ---------------------------------------------------------------
    if re.match(r"^(next|skip)\b", t):
        musicish = re.search(r"\b(song|track|music|spotify|tune|one|this|it)\b", t)
        if musicish or len(t.split()) == 1:  # bare "next"/"skip"; not "next Tuesday"
            return {"tool": "music_skip", "args": {"direction": "next"}}
    # "play the next song" / "move to the next" mean SKIP, not play a track
    # literally named "Next Song" - the log caught music_play searching for
    # it. These phrasings never carry a real track title.
    if re.match(
        r"^(?:play|put on|move to|go to|switch to|jump to|take me to) "
        r"(?:the |my )?next\b",
        t,
    ):
        return {"tool": "music_skip", "args": {"direction": "next"}}

    # resume / bare play ---------------------------------------------------
    if re.match(
        r"^(resume|unpause|continue|carry on|go ahead|go on|keep playing)\b", t
    ) or t in ("play", "music", "spotify"):
        return {"tool": "music_resume", "args": {}}

    # volume ---------------------------------------------------------------
    # Direction words come in two strengths. Strong words ("louder", "crank")
    # are unambiguous anywhere in the utterance. Bare "up"/"down"/"higher"
    # are common LYRIC words ("wake me up, wake me up", "take me down to the
    # river", "take me higher") - they only count as a volume command when
    # paired with a volume idiom ("turn it up", "bring the volume down",
    # "up the music"). Verbs are limited to volume idioms: "take"/"set" are
    # far too common in speech and lyrics ("take me down", "set it up").
    vol_verb = r"\b(turn|bring|push|dial)\b"
    vol_obj = r"\b(volume|music|sound)\b"
    dir_up = bool(
        re.search(r"\b(louder|boost|blare|blast|crank)\b", t)
        or (
            re.search(r"\b(up|higher)\b", t)
            and re.search(vol_verb + "|" + vol_obj, t)
        )
        or (re.search(r"\braise\b", t) and re.search(vol_obj, t))
    )
    dir_down = bool(
        re.search(r"\b(quieter|softer|lower|dip)\b", t)
        or (
            re.search(r"\bdown\b", t)
            and re.search(vol_verb + "|" + vol_obj, t)
        )
    )
    vol_word = "volume" in t.split()
    if (
        vol_word
        or dir_up
        or dir_down
        or re.search(r"\b(max|maximum|full|mute|silent)\b", t)
        or (num and (dir_up or dir_down))
    ):
        if re.search(r"\bmute\b|\bsilent\b", t):
            level = "mute"
        elif re.search(r"\b(max|maximum|full)\b", t):
            level = "max"
        elif num is not None:
            n = num.group(0)
            if re.search(r"\b(?:up|down)\s+to\b", t):
                level = n  # "turn it up to 40" -> absolute 40
            else:
                level = (
                    "up "
                    if dir_up and not dir_down
                    else "down "
                    if dir_down and not dir_up
                    else ""
                ) + n
        elif dir_up and not dir_down:
            level = "up"
        elif dir_down and not dir_up:
            level = "down"
        else:
            level = ""  # "what's the volume?" -> report current
        return {"tool": "music_volume", "args": {"level": level}}

    # queue ------------------------------------------------------------------
    m = re.match(r"^queue\s+(.+)$", t)
    if m and len(m.group(1)) >= 2:
        return {"tool": "music_queue", "args": {"query": m.group(1)}}
    m = re.match(
        r"^add\s+(.+?)\s+to\s+(?:the |my )?(?:play )?queue(?: (?:up )?next)?$", t
    )
    if m and len(m.group(1)) >= 2:
        return {"tool": "music_queue", "args": {"query": m.group(1)}}

    # play <query> -------------------------------------------------------------
    m = re.match(r"^(?:play|put on)\s+(.+)$", t)
    if m:
        q = m.group(1).strip()
        if q in ("it", "that", "this", "something", "some music", "the music"):
            return {"tool": "music_resume", "args": {}}
        if len(q) >= 2 and 1 <= len(q.split()) <= 10:
            kind = (
                "playlist"
                if q.startswith("playlist ")
                else "album"
                if q.startswith("album ")
                else "track"
            )
            q = re.sub(r"^(playlist|album|artist) ", "", q).strip()
            q = re.sub(r"^(?:the )?(?:song|track|tune) ", "", q).strip() or q
            return {"tool": "music_play", "args": {"query": q, "kind": kind}}
    return None


def maybe_music_command(text: str) -> dict | None:
    """Gate for the fast path: music mode, or an explicit music word.

    Outside music mode a bare "next" must not hijack general talk ("next
    Tuesday") - only explicit music vocabulary unlocks the fast path."""
    if not FAST_MUSIC_COMMANDS:
        return None
    if _MUSIC_NOT_OURS.search(text):
        return None
    in_music = time.monotonic() < STATE["music_mode_until"]
    if not in_music and not _MUSIC_WORDS.search(text):
        return None
    return parse_music_command(text)


def _voice_of_track(line: str) -> str:
    """TTS-friendly track name: 'X — Y' -> 'X, by Y'."""
    parts = (line or "").split(" — ")
    if len(parts) >= 2:
        return f"{parts[0].strip()}, by {parts[-1].strip()}"
    return (line or "").strip()


def _web_player_note(name: str) -> str:
    return (
        ", in the Spotify web player tab"
        if "web player" in (name or "").lower()
        else ""
    )


def _music_error_speech(err: str) -> str:
    """Short spoken form of a tool error (details live in the logs)."""
    low = (err or "").lower()
    if "not linked" in low or "authorization expired" in low:
        return (
            "Spotify isn't linked to me. Open Lava's Spotify link page in "
            "this browser and approve once."
        )
    if "reach spotify" in low or "rate-limiting" in low or "network" in low:
        return "I can't reach Spotify right now. Give it a few seconds and try again."
    if "premium" in low:
        return "Spotify Premium is needed to control playback."
    if "no spotify device named" in low:
        return "I can't find a Spotify device with that name."
    if "several spotify devices" in low:
        return "Several Spotify devices are available. Which one should I use?"
    if "open spotify" in low or "no active spotify device" in low:
        return (
            "No Spotify device is active. Open Spotify somewhere, press "
            "play once, then ask me again."
        )
    if "nothing is playing" in low:
        return "Nothing is playing right now."
    cleaned = re.sub(r"https?://\S+", "the Lava link page", err or "")
    return "Hmm - " + cleaned[:140]


def _music_reply(tool: str, args: dict, res: dict) -> str:
    """Voice confirmation for the fast path, straight from the tool result."""
    err = res.get("error")
    if err:
        return _music_error_speech(err)
    webn = _web_player_note(res.get("device", ""))
    if tool == "music_pause":
        return "Nothing was playing." if res.get("paused") is False else "Paused."
    if tool == "music_resume":
        return "Resuming."
    if tool == "music_restart":
        return f"Starting {_voice_of_track(res.get('restarted', 'this song'))} over."
    if tool == "music_skip":
        tr = _voice_of_track(res.get("playing", "")) if res.get("playing") else ""
        return (
            f"Now on {tr}{webn}."
            if tr
            else "Back a track."
            if args.get("direction") == "previous"
            else "Skipped."
        )
    if tool == "music_play":
        tr = _voice_of_track(res.get("playing", ""))
        if not tr or tr.lower() == "started":
            return "Playback started."
        return f"Playing {tr}{webn}."
    if tool == "music_queue":
        return f"Queued {res.get('queued', 'that')}."
    if tool == "music_volume":
        pct = res.get("volume_pct")
        if pct is None:
            return "There's no volume to read right now."
        if not res.get("action"):
            return f"The volume is at {pct} percent."
        extra = " It was already there." if "already" in res.get("note", "") else ""
        return f"Volume {res['action']} to {pct} percent.{extra}"
    if tool == "music_shuffle":
        return res.get("note") or f"Shuffle is {res.get('shuffle', 'unknown')}."
    if tool == "music_repeat":
        return res.get("note") or f"Repeat mode is {res.get('repeat', 'unknown')}."
    if tool == "music_now_playing":
        if not res.get("playing"):
            return "Nothing is playing right now."
        name = _voice_of_track(res["playing"])
        return (
            f"{name} is paused." if res.get("paused") else f"Now playing {name}{webn}."
        )
    if tool == "music_devices":
        devs = res.get("devices") or []
        if not devs:
            return "No Spotify devices are online."
        return "Devices online: " + ", ".join(d.get("name", "?") for d in devs) + "."
    return "Done."


async def run_music_command(
    ws: web.WebSocketResponse, sess: dict, text: str, stats: dict, cmd: dict
):
    """Execute a matched music command directly (no LLM) and ack by voice."""
    tool, args = cmd["tool"], cmd["args"]
    logger.info(f"[web] music fast path {tool}({json.dumps(args)}) <- {text!r}")
    await send(ws, {"type": "tool", "name": tool, "args": json.dumps(args)[:120]})
    history = sess["history"]
    history.append({"role": "user", "content": text})
    result = await exec_tool(tool, args)
    try:
        res = json.loads(result)
    except ValueError:
        res = {}
    reply = _music_reply(tool, args, res)
    history.append({"role": "assistant", "content": reply})
    _trim_history(history)
    # Mirror into STATE so the VAD reacts the same tick as Spotify does.
    now = time.monotonic()
    if tool == "music_pause" and res.get("paused", True):
        STATE["music_playing"] = False
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
    elif (tool == "music_resume" and res.get("resumed")) or (
        tool == "music_play" and "error" not in res and res.get("playing")
    ):
        STATE["music_playing"] = True
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
        STATE["music_seen_at"] = now
    await send(ws, {"type": "text", "t": reply})
    try:
        await speak_text(ws, sess, reply, stats)
    except Exception as e:
        logger.warning(f"[web] music fast path TTS failed: {e}")
    # No Speaker on this path, so ttfa (time to first audio) is measured
    # here: wall clock from turn start to now that the reply is spoken.
    await send(
        ws,
        {
            "type": "done",
            "reply": reply,
            "stats": done_stats(
                stats, ttfa_ms=(time.perf_counter() - stats["turn_t0"]) * 1000
            ),
        },
    )
