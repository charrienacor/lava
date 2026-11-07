"""Command notifications: map a tool call to one of the ten command
categories the UI shows as a transient toast (banner) when it fires.

The bridge is the single source of truth for *what* category a tool call
belongs to (it knows the tool name, the parsed arguments, and - on the
"done" phase - the result); the browser only handles presentation. This
keeps the classification in one place and makes the toast copy easy to
tune without touching the UI.

Categories (the ten command families, plus a fallback):
  music       play a song/album/playlist/artist
  query       time, weather, web search, wikipedia
  lights      turn lights on/off (no brightness or color in the request)
  dim         adjust brightness or color of lights
  timer       start / list / cancel a countdown timer
  alarm       set an alarm for a clock time
  thermostat  set or adjust the room temperature
  media       pause / resume / skip / volume / shuffle / repeat / queue
  reminder    add / list / delete notes and reminders
  calls       Telegram call-back requests and messages (out through the
              Telegram Bot API; recipients learned from incoming messages)
  system      Raspberry Pi health (CPU temp, load, memory, uptime, IP)
  unknown     anything else (internal tools, future additions)
"""

import hashlib
import json
import re

# --- category constants (mirrored in web/index.html NOTIFY_MAP) -----------
CAT_MUSIC = "music"
CAT_QUERY = "query"
CAT_LIGHTS = "lights"
CAT_DIM = "dim"
CAT_TIMER = "timer"
CAT_ALARM = "alarm"
CAT_THERMOSTAT = "thermostat"
CAT_MEDIA = "media"
CAT_REMINDER = "reminder"
CAT_CALLS = "calls"
CAT_SYSTEM = "system"
CAT_UNKNOWN = "unknown"


# --- small text helpers -----------------------------------------------------

def _clean(text, maxlen: int = 300) -> str:
    """Trim and ellipsize a free-form argument for a toast. The cap is high
    enough that a normal request is shown in full (the toast wraps to a few
    lines now); only pathological inputs still get an ellipsis."""
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip()
    if len(text) > maxlen:
        text = text[: maxlen - 1].rstrip() + "…"
    return text


def _human_duration(value) -> str | None:
    """Render a duration (seconds) for display: 300 -> "5 min"."""
    if isinstance(value, str):
        return _clean(value, 24) or None
    try:
        secs = int(value)
    except (TypeError, ValueError):
        return None
    if secs <= 0:
        return None
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hr" + ("s" if h != 1 else ""))
    if m:
        parts.append(f"{m} min" + ("s" if m != 1 else ""))
    if s and not h:
        parts.append(f"{s} s")
    return " ".join(parts) or None


# --- intent patterns (cross-tool: the LLM may route these through any
# free-text tool, e.g. web_search, until dedicated tools land) --------------

_LIGHT_WORD = re.compile(r"\blights?\b|\blamps?\b|\bbulbs?\b", re.I)
_BRIGHT_COLOR = re.compile(
    r"\b(dim|dimmer|brighter|darker|softer|fainter|glow|warm|cool|color|colour|"
    r"red|orange|yellow|green|blue|purple|pink|white|amber)\b|\d+\s*(%|percent)",
    re.I,
)
_CLOCK_TIME = re.compile(r"\b\d{1,2}\s*(am|pm)\b", re.I)
_THERMOSTAT = re.compile(
    r"\bthermostat\b|\btemperature\b|\bdegrees?\b|\bset (?:it|the room) to \d+|"
    r"\b(?:heat|cool)(?:ing|er)?\b",
    re.I,
)
_CALLS = re.compile(r"\b(?:call|calls|text|texts|message|messages|dial|ring)\s+\w+", re.I)

# Tool-name keywords for categories that have no dedicated tool yet. When a
# light/thermostat/call tool is added, it is caught by name even if nobody
# remembers to register it in _BUILDERS. Checked only for UNMAPPED tools so a
# real web_search("how to dim lights") stays an honest "Web search" toast.
_NAME_CAT = [
    (re.compile(r"\b(lights?|lighting|lamps?|bulbs?|dimmer?s?)\b", re.I), CAT_LIGHTS),
    (re.compile(r"\b(thermostat|temperature|climate|hvac|heating|cooling)\b", re.I), CAT_THERMOSTAT),
    (re.compile(r"\b(call|calls|phone|phones|sms|text|texts|message|messages|dial)\b", re.I), CAT_CALLS),
]

# The text-intent heuristics below only fire when the request READS like a
# command ("turn on the lights"), not a question about the domain ("how do I
# dim lights?"). Voice transcripts of real commands are near-imperative;
# polite prefixes are stripped first.
_CMD_PREFIX = re.compile(
    r"^(?:please|can you|could you|will you|would you|hey lava|lava|ok|okay)\s+", re.I
)
_CMD_VERB = re.compile(
    r"^(?:turn|switch|set|make|dim|raise|lower|increase|decrease|adjust|change|"
    r"call|text|message|dial|ring|start|stop|pause|resume|cancel|add|remind|"
    r"play|put|give|bring|check|show|tell)\b",
    re.I,
)


def _is_command(text: str) -> bool:
    t = _CMD_PREFIX.sub("", text.strip())
    return bool(_CMD_VERB.match(t))


def _free_text(args: dict) -> str:
    return " ".join(
        str(args[k]) for k in ("query", "topic", "text", "label", "location") if args.get(k)
    ).strip()


# --- per-category title/subtitle builders -----------------------------------
# Each returns (title, subtitle). Subtitle may be "" (nothing to add).

def _t_music(name: str, args: dict, res: dict) -> tuple[str, str]:
    if name == "music_queue":
        return "Queue", _clean(args.get("query", ""))
    title = "Play music"
    sub = _clean(args.get("query", ""))
    track = res.get("playing")
    if isinstance(track, str) and track and track.lower() != "started":
        sub = _clean(track)
    if res.get("device"):
        sub = f"{sub} · {res['device']}" if sub else str(res["device"])
    return title, sub


def _t_query(name: str, args: dict, res: dict) -> tuple[str, str]:
    titles = {
        "get_time": "Time",
        "get_weather": "Weather",
        "web_search": "Web search",
        "wikipedia": "Wikipedia",
    }
    title = titles.get(name, "Look up")
    sub = ""
    if name == "get_weather":
        sub = _clean(args.get("location", ""), 32)
    elif name == "web_search":
        sub = _clean(args.get("query", ""))
    elif name == "wikipedia":
        sub = _clean(args.get("topic", ""))
    return title, sub


def _t_lights(args: dict, res: dict) -> tuple[str, str]:
    # Done phase: the hub's answer (matched friendly name + final state).
    # Start phase: the requested room + intended state.
    room = res.get("lights") or _clean(args.get("room", ""), 32)
    state = res.get("on", args.get("on", True))
    sub = f"{room} · {'on' if state else 'off'}" if room else ("on" if state else "off")
    return "Lights", sub


def _t_dim(args: dict, res: dict) -> tuple[str, str]:
    room = res.get("lights") or _clean(args.get("room", ""), 32)
    bits = [room] if room else []
    bright = res.get("brightness_pct") or args.get("brightness_pct")
    if bright:
        bits.append(f"{bright}%")
    color = res.get("color") or args.get("color")
    if color:
        bits.append(_clean(color, 20))
    return "Dim / color", " · ".join(bits)


def _t_timer(name: str, args: dict, res: dict) -> tuple[str, str]:
    if name == "set_timer":
        label = _clean(args.get("label", ""), 32)
        title = f"Timer: {label}" if label and label.lower() != "timer" else "Timer"
        sub = _human_duration(args.get("seconds")) or ""
    elif name == "cancel_timer":
        title, sub = "Cancel timer", _clean(args.get("label_or_id", ""), 32)
    else:  # list_timers
        title, sub = "Timers", "listing"
    return title, sub


def _t_alarm(args: dict, res: dict) -> tuple[str, str]:
    sub = _clean(args.get("label", ""), 32)
    dur = _human_duration(args.get("seconds"))
    if sub and dur:
        sub = f"{sub} · {dur}"
    elif dur:
        sub = dur
    return "Alarm", sub


def _t_thermostat(name: str, args: dict, res: dict) -> tuple[str, str]:
    # Read (get_temperature): the hub's CURRENT reading is the headline; the
    # target only as a trailing note when it differs.
    if name == "get_temperature":
        cur = res.get("current")
        unit = res.get("unit", "°C")
        room = res.get("thermostat") or _clean(args.get("room", ""), 32)
        if cur is None:
            return "Temperature", _clean(_free_text(args), 32)
        sub = f"{cur} {unit}"
        tgt = res.get("target")
        if tgt is not None and tgt != cur:
            sub += f" · set to {tgt}"
        if room:
            sub = f"{room} · {sub}"
        return "Temperature", sub
    # Set (set_thermostat): the requested target is the headline.
    temp = res.get("temperature", args.get("temperature"))
    unit = res.get("unit", args.get("unit", "C"))
    if temp is None:
        return "Thermostat", _clean(_free_text(args), 32)
    mark = "°C" if str(unit).upper().startswith("C") else "°F"
    room = res.get("thermostat") or _clean(args.get("room", ""), 32)
    sub = f"set to {temp} {mark}"
    if room:
        sub = f"{room} · {sub}"
    return "Thermostat", sub


def _t_media(name: str, args: dict, res: dict) -> tuple[str, str]:
    if name == "music_pause":
        return "Pause", ""
    if name == "music_resume":
        return "Resume", ""
    if name == "music_restart":
        return "Restart track", ""
    if name == "music_skip":
        return ("Previous", "") if args.get("direction") == "previous" else ("Next track", "")
    if name == "music_volume":
        lvl = _clean(str(args.get("level", "")), 16)
        if res.get("volume_pct") is not None:
            lvl = f"{res['volume_pct']}%"
        return "Volume", lvl
    if name == "music_shuffle":
        return "Shuffle", _clean(args.get("state", ""), 16)
    if name == "music_repeat":
        return "Repeat", _clean(args.get("mode", ""), 16)
    if name == "music_now_playing":
        track = res.get("playing")
        return "Now playing", (_clean(track, 48) if isinstance(track, str) else "")
    if name == "music_devices":
        return "Devices", ""
    return "Media", _clean(args.get("query", ""), 32)


def _t_calls(name: str, args: dict, res: dict) -> tuple[str, str]:
    if name == "send_message":
        who = _clean(res.get("to") or args.get("recipient", ""), 40)
        body = _clean(args.get("message", ""), 300)
        if res.get("to"):
            # Done: the message went out to this person - show the full body.
            sub = f"{who}: “{body}”" if who else f"“{body}”"
            return "Message", sub
        # Start: still composing/sending.
        sub = f"{who}: “{body}”" if who else f"“{body}”"
        return "Message", (f"{sub} · sending" if sub else "")
    who = _clean(res.get("calling") or args.get("contact", ""), 32)
    if res.get("calling"):
        # Done: the call-back request went out on Telegram for this person.
        return "Call", (f"{who} · Telegram call request" if who else "")
    return "Call", (f"{who} · call request" if who else "")


def _t_system(name: str, args: dict, res: dict) -> tuple[str, str]:
    # Pi health: the readings ARE the answer, so the done phase leads with
    # them; the start phase just says we are checking.
    bits = []
    cpu = res.get("cpu_temp_c")
    if cpu is not None:
        bits.append(f"CPU {cpu}\u00b0C")
    mem = res.get("memory_used_pct")
    if mem is not None:
        bits.append(f"mem {mem}%")
    load = res.get("load_1m")
    if load is not None:
        bits.append(f"load {load}")
    up = res.get("uptime_hours")
    if up is not None:
        bits.append(f"up {up}h")
    ip = res.get("ip")
    if ip:
        bits.append(ip)
    sub = " \u00b7 ".join(bits)
    if not sub:
        sub = _clean(_free_text(args), 32)
    return "System", sub


def _t_reminder(name: str, args: dict, res: dict) -> tuple[str, str]:
    if name == "add_note":
        return "Reminder", _clean(args.get("text", ""))
    if name == "delete_note":
        return "Remove reminder", _clean(args.get("match", ""))
    if name == "clear_notes":
        return "Clear reminders", ""
    return "Reminders", "listing"  # list_notes


# --- the classifier ---------------------------------------------------------

# Mapped tools: name -> (category, builder). Music play/queue and the
# explicit media controls are handled before this table so song titles like
# "Turn On The Lights" can never be misread as a light command.
_BUILDERS = {
    "get_time": (CAT_QUERY, _t_query),
    "get_weather": (CAT_QUERY, _t_query),
    "web_search": (CAT_QUERY, _t_query),
    "wikipedia": (CAT_QUERY, _t_query),
    "get_temperature": (CAT_THERMOSTAT, _t_thermostat),
    "set_timer": (CAT_TIMER, _t_timer),
    "list_timers": (CAT_TIMER, _t_timer),
    "cancel_timer": (CAT_TIMER, _t_timer),
    "system_status": (CAT_SYSTEM, _t_system),
    "add_note": (CAT_REMINDER, _t_reminder),
    "list_notes": (CAT_REMINDER, _t_reminder),
    "delete_note": (CAT_REMINDER, _t_reminder),
    "clear_notes": (CAT_REMINDER, _t_reminder),
    "music_pause": (CAT_MEDIA, _t_media),
    "music_resume": (CAT_MEDIA, _t_media),
    "music_restart": (CAT_MEDIA, _t_media),
    "music_skip": (CAT_MEDIA, _t_media),
    "music_volume": (CAT_MEDIA, _t_media),
    "music_shuffle": (CAT_MEDIA, _t_media),
    "music_repeat": (CAT_MEDIA, _t_media),
    "music_now_playing": (CAT_MEDIA, _t_media),
    "music_devices": (CAT_MEDIA, _t_media),
    "call_contact": (CAT_CALLS, _t_calls),
    "send_message": (CAT_CALLS, _t_calls),
}


def describe(name: str, args: dict, result: str | None = None) -> dict:
    """Describe a tool call for the UI toast.

    Returns {"cat", "title", "sub", "err"}. Call it once with result=None
    (optimistic, shown while the tool runs) and again with the raw result
    string (final, shown on completion - this is where the track title,
    device, volume % and errors come from).
    """
    args = args or {}
    res: dict = {}
    err: str | None = None
    if result is not None:
        try:
            parsed = json.loads(result)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            res = parsed
            if parsed.get("error"):
                err = _clean(parsed["error"], 64)
        elif isinstance(parsed, str) and parsed.strip():
            err = _clean(parsed, 64)

    # Explicit tools always win: a song titled "Turn On The Lights" played
    # via music_play is music, and "remind me to call mom" stored via
    # add_note is a reminder - the tool name is the intent.
    if name in ("music_play", "music_queue"):
        cat, (title, sub) = CAT_MUSIC, _t_music(name, args, res)
    elif name == "set_timer":
        if _CLOCK_TIME.search(str(args.get("label", ""))):
            cat, (title, sub) = CAT_ALARM, _t_alarm(args, res)
        else:
            cat, (title, sub) = CAT_TIMER, _t_timer(name, args, res)
    elif name == "set_lights":
        # One tool, two families: brightness/colour requests are "dim",
        # plain on/off stays "lights".
        if args.get("brightness_pct") or args.get("color"):
            cat, (title, sub) = CAT_DIM, _t_dim(args, res)
        else:
            cat, (title, sub) = CAT_LIGHTS, _t_lights(args, res)
    elif name == "set_thermostat":
        cat, (title, sub) = CAT_THERMOSTAT, _t_thermostat(name, args, res)
    elif name == "get_temperature":
        cat, (title, sub) = CAT_THERMOSTAT, _t_thermostat(name, args, res)
    elif name == "call_contact":
        cat, (title, sub) = CAT_CALLS, _t_calls(name, args, res)
    else:
        # Everything else: read the intent out of the request text first -
        # the toast should name the COMMAND the user gave, not the plumbing
        # tool the LLM reached for ("turn on the lights" answered via
        # web_search is still a light command on screen). The gate keeps
        # genuine questions ("how do I dim lights?") in their honest
        # category, and "remind me to ..." stays a reminder.
        text = _free_text(args)
        if re.search(r"^remind\b", text, re.I):
            # "remind me to call mom" is a reminder, not a call.
            cat, (title, sub) = CAT_REMINDER, ("Reminder", _clean(text))
        elif _is_command(text) and _LIGHT_WORD.search(text):
            if _BRIGHT_COLOR.search(text):
                cat, (title, sub) = CAT_DIM, _t_dim(args, res)
            else:
                cat, (title, sub) = CAT_LIGHTS, _t_lights(args, res)
        elif _is_command(text) and _THERMOSTAT.search(text):
            cat, (title, sub) = CAT_THERMOSTAT, _t_thermostat(name, args, res)
        elif _is_command(text) and _CALLS.search(text):
            cat, (title, sub) = CAT_CALLS, ("Call", _clean(text))
        elif name in _BUILDERS:
            # Mapped tool with no special intent in the text (web_search for
            # a fact, get_weather, ...): its own category.
            cat, build = _BUILDERS[name]
            title, sub = build(name, args, res)
        else:
            # Unmapped tool (internal, future): recognize the category from
            # the tool NAME (a future set_light / set_thermostat / call_phone
            # tool lands in the right family before anyone registers it),
            # else a quiet generic toast.
            norm = name.replace("_", " ")
            cat = CAT_UNKNOWN
            for pat, c in _NAME_CAT:
                if pat.search(norm):
                    cat = c
                    break
            label = name.replace("_", " ").strip().capitalize() or "Command"
            if cat == CAT_LIGHTS:
                title = "Lights"
                on = args.get("on", args.get("state"))
                sub = _clean(str(on), 16) if on is not None else _clean(text)
            elif cat == CAT_THERMOSTAT:
                title = "Thermostat"
                temp = args.get("temperature", args.get("temp", args.get("degrees")))
                sub = f"set to {temp} {args.get('unit', '°C')}" if temp is not None else _clean(text, 32)
            elif cat == CAT_CALLS:
                title = "Call"
                sub = _clean(args.get("contact", args.get("to", args.get("number", ""))), 32) or _clean(text, 44)
            else:
                title, sub = label, _clean(text)

    return {"cat": cat, "title": title, "sub": sub, "err": err}


def tool_key(name: str, args: dict) -> str:
    """Stable identity for one tool call: same name + same arguments -> the
    same key, so the UI can collapse repeats (two "pause"s) into one toast
    and match the "done" frame to its "start" frame."""
    h = hashlib.sha1(json.dumps([name, args or {}], sort_keys=True).encode()).hexdigest()
    return f"{name}:{h[:10]}"


def tool_frame(phase: str, name: str, args: dict, result: str | None) -> dict:
    """Build the WebSocket frame for one tool phase. `phase` is "start"
    (optimistic toast while the tool runs) or "done" (final, with result).
    Used by both the LLM tool loop (turn.py) and the music fast path
    (music.py) so the two paths emit byte-identical frames."""
    d = describe(name, args, result)
    return {
        "type": "tool_start" if phase == "start" else "tool_done",
        "key": tool_key(name, args),
        "cat": d["cat"],
        "title": d["title"],
        "sub": d["sub"],
        "err": d["err"],
    }
