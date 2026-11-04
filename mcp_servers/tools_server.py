"""Real-world task tools as an MCP server.

Every tool here uses free services or the Pi itself, with no API keys:

  get_time        local date/time in any timezone
  get_weather     current weather + today's forecast (Open-Meteo)
  web_search      web search (DuckDuckGo via the `ddgs` package)
  wikipedia       short encyclopedia summary (Wikipedia REST API)
  set_timer etc.  countdown timers; the agent announces them when they finish
  notes           add / list / delete / clear a persistent to-do / notes list
  set_light       switch an LED or relay on a GPIO pin (simulated if no GPIO)
  system_status   Pi CPU temperature, load, memory, uptime, IP
  music_*         Spotify search / play / pause / skip / queue / volume /
                  shuffle / repeat / now-playing / devices
                  (needs SPOTIFY_CLIENT_ID/SECRET in .env; playback needs the
                  one-time /spotify/auth link and a Premium account)

Run:  python -m mcp_servers.tools_server          (port 8003 by default)
"""

import asyncio
import json
import os
import re
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from loguru import logger
from mcp.server.mcpserver import MCPServer

from mcp_servers import spotify

from mcp_servers.common import ROOT, setup_logging

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("TOOLS_PORT", "8003"))
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Manila")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Quezon City")
NOTES_FILE = ROOT / "logs" / "notes.json"

UA = {"User-Agent": "Lava-voice-agent/1.0 (Raspberry Pi)"}

# One pooled client for every outbound HTTP tool call. A fresh client per call
# paid a full TLS handshake (~100-300 ms on the Pi) on every weather lookup;
# pooling keeps the connection warm between calls.
_http: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=10, headers=UA, follow_redirects=True)
    return _http


# City -> (lat, lon, display name). Geocoding is a stable lookup and costs a
# full network round trip (~1.5 s); the weather tool is called repeatedly for
# the same few cities, so caching it halves the tool's latency.
_geo_cache: dict[str, tuple[float, float, str]] = {}


@asynccontextmanager
async def _lifespan(_server):
    yield {}
    global _http
    if _http is not None and not _http.is_closed:
        await _http.aclose()


mcp = MCPServer(
    "lava-tools",
    instructions="Everyday tools for a voice assistant running on a Raspberry Pi.",
    lifespan=_lifespan,
)

# time


@mcp.tool()
def get_time(timezone: str = DEFAULT_TZ) -> dict:
    """Get the current local date and time.

    Args:
        timezone: IANA timezone name, e.g. "Asia/Manila", "America/New_York", "Europe/London".
    """
    now = datetime.now(ZoneInfo(timezone))
    return {
        "timezone": timezone,
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "date": now.strftime("%A, %B %d, %Y"),
        "iso": now.isoformat(timespec="seconds"),
    }


# weather

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "light showers", 81: "showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


@mcp.tool()
async def get_weather(location: str = DEFAULT_CITY) -> dict:
    """Get the current weather and today's forecast for a city.

    Args:
        location: City name, e.g. "Manila" or "Tokyo".
    """
    try:
        http = get_http()
        key = location.strip().lower()
        cached = _geo_cache.get(key)
        if cached:
            lat, lon, place = cached
        else:
            geo_resp = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en"},
            )
            geo_resp.raise_for_status()
            geo = geo_resp.json()
            if not geo.get("results"):
                return {"error": f"Could not find a place called {location}"}
            hit = geo["results"][0]
            lat, lon = hit["latitude"], hit["longitude"]
            place = f"{hit['name']}, {hit.get('country', '')}".strip(", ")
            _geo_cache[key] = (lat, lon, place)
        wx_resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
        )
        wx_resp.raise_for_status()
        wx = wx_resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return {"error": f"weather lookup failed: {e}"}
    cur, day = wx["current"], wx["daily"]
    return {
        "place": place,
        "conditions": WMO_CODES.get(cur["weather_code"], "unknown"),
        "temperature_c": cur["temperature_2m"],
        "feels_like_c": cur["apparent_temperature"],
        "humidity_pct": cur["relative_humidity_2m"],
        "wind_kmh": cur["wind_speed_10m"],
        "today_high_c": day["temperature_2m_max"][0],
        "today_low_c": day["temperature_2m_min"][0],
        "rain_chance_pct": day["precipitation_probability_max"][0],
    }


# search / knowledge


@mcp.tool()
async def web_search(query: str, max_results: int = 4) -> dict:
    """Search the web for current information (news, prices, facts, events).

    Args:
        query: What to search for.
        max_results: Number of results to return (1-8).
    """
    from ddgs import DDGS

    max_results = max(1, min(int(max_results or 4), 8))
    query = (query or "").strip()
    if not query:
        return {"error": "Give me something to search for."}

    def _search():
        return DDGS().text(query, max_results=max_results)

    try:
        hits = await asyncio.to_thread(_search)
    except Exception as e:
        # ddgs raises "No results found." when the engine simply has no hits -
        # that is an empty answer, not a failure.
        if "no results" in str(e).lower():
            return {"query": query, "results": [],
                    "note": f"No web results for {query!r} - try different wording"}
        return {"error": f"search failed: {str(e)[:200]}"}
    return {
        "query": query,
        "results": [{"title": h.get("title"), "snippet": h.get("body"), "url": h.get("href")}
                    for h in hits if h],
    }


@mcp.tool()
async def wikipedia(topic: str) -> dict:
    """Get a short encyclopedia summary of a person, place, thing or concept.

    Args:
        topic: The subject, e.g. "Jose Rizal" or "Raspberry Pi".
    """
    try:
        http = get_http()
        search = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
        )
        search.raise_for_status()
        try:
            search_json = search.json()
        except ValueError:
            return {"error": f"Wikipedia returned a non-JSON response (status {search.status_code}); try again later"}
        if not search_json[1]:
            return {"error": f"No Wikipedia article found for {topic}"}
        title = search_json[1][0]
        page_resp = await http.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        )
        page_resp.raise_for_status()
        try:
            page = page_resp.json()
        except ValueError:
            return {"error": f"Wikipedia summary lookup failed (status {page_resp.status_code}); try again later"}
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return {"error": f"Wikipedia lookup failed: {e}"}
    return {"title": page.get("title", title), "summary": page.get("extract", "")}


# timers

_timers: dict[str, dict] = {}
_finished: list[dict] = []


async def _run_timer(timer_id: str):
    t = _timers[timer_id]
    await asyncio.sleep(max(0.0, t["ends_at"] - time.time()))
    if _timers.pop(timer_id, None):
        logger.info(f"Timer done: {t['label']}")
        _finished.append({"id": timer_id, "label": t["label"], "seconds": t["seconds"]})


@mcp.tool()
async def set_timer(seconds: int, label: str = "timer") -> dict:
    """Start a countdown timer. The assistant will announce it when it finishes.

    Args:
        seconds: Duration in seconds (e.g. 300 for five minutes).
        label: Short name such as "pasta" or "break".
    """
    try:
        seconds = max(1, int(seconds))   # "in 0 seconds" or negatives rot in place
    except (TypeError, ValueError):
        return {"error": f"Timer length {seconds!r} is not a number of seconds."}
    timer_id = uuid.uuid4().hex[:6]
    _timers[timer_id] = {"label": label, "seconds": seconds, "ends_at": time.time() + seconds}
    _timers[timer_id]["task"] = asyncio.create_task(_run_timer(timer_id))
    return {"id": timer_id, "label": label, "seconds": seconds, "status": "started"}


@mcp.tool()
def list_timers() -> dict:
    """List running timers and how long each has left."""
    return {
        "timers": [
            {"id": k, "label": v["label"], "seconds_left": max(0, round(v["ends_at"] - time.time()))}
            for k, v in _timers.items()
        ]
    }


@mcp.tool()
def cancel_timer(label_or_id: str) -> dict:
    """Cancel a running timer by its label or id.

    Args:
        label_or_id: The timer's label (e.g. "pasta") or id.
    """
    for k, v in list(_timers.items()):
        if label_or_id.strip().lower() in (k.lower(), v["label"].lower()):
            v["task"].cancel()
            del _timers[k]
            return {"cancelled": v["label"]}
    return {"error": f"No running timer called {label_or_id}"}


@mcp.tool()
def pop_finished_timers() -> dict:
    """Internal: return timers that finished since the last call (polled by the agent)."""
    done = list(_finished)
    _finished.clear()
    return {"finished": done}


# notes


def _load_notes() -> list[str]:
    try:
        return json.loads(NOTES_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_notes(notes: list[str]):
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


@mcp.tool()
def add_note(text: str) -> dict:
    """Save a note or to-do item that persists across restarts.

    Args:
        text: The note, e.g. "buy eggs".
    """
    notes = _load_notes()
    notes.append(text)
    _save_notes(notes)
    return {"saved": text, "count": len(notes)}


@mcp.tool()
def list_notes() -> dict:
    """Read back all saved notes / to-do items."""
    return {"notes": _load_notes()}


@mcp.tool()
def delete_note(match: str) -> dict:
    """Delete saved notes whose text matches a phrase.

    Case-insensitive substring match against every saved note; every note
    that matches is removed. Ideal for voice requests like "delete the
    reminder about eggs". Non-matching text returns an error.

    Args:
        match: Text to look for inside a saved note, e.g. "eggs".
    """
    notes = _load_notes()
    m = match.strip().lower()
    if not m:
        return {"error": "match must be non-empty"}
    keep, deleted = [], []
    for n in notes:
        if m in n.lower():
            deleted.append(n)
        else:
            keep.append(n)
    if not deleted:
        return {"deleted": [], "remaining": notes,
                "error": f"No note matching {match!r}"}
    _save_notes(keep)
    return {"deleted": deleted, "remaining": keep}


@mcp.tool()
def clear_notes() -> dict:
    """Delete all saved notes."""
    n = len(_load_notes())
    _save_notes([])
    return {"deleted": n}


# system

# Simulated light state; wire a relay to real GPIO before trusting it for
# anything physical.
_light_on = False


@mcp.tool()
def set_light(state: str) -> dict:
    """Turn the light on or off.

    Args:
        state: "on" or "off".
    """
    global _light_on
    s = state.strip().lower()
    if s not in ("on", "off"):
        return {"error": f"state must be 'on' or 'off', got {state!r}"}
    _light_on = s == "on"
    return {"light": s}


@mcp.tool()
def system_status() -> dict:
    """Report the Raspberry Pi's health: CPU temperature, load, memory, uptime and IP address."""
    out: dict = {}
    try:
        out["cpu_temp_c"] = round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except OSError:
        out["cpu_temp_c"] = None
    out["load_1m"] = round(os.getloadavg()[0], 2)
    try:
        mem = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total = int(mem["MemTotal"].split()[0]) / 1024
        avail = int(mem["MemAvailable"].split()[0]) / 1024
        out["memory_used_pct"] = round(100 * (1 - avail / total))
        out["uptime_hours"] = round(float(open("/proc/uptime").read().split()[0]) / 3600, 1)
    except (OSError, KeyError):
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            out["ip"] = s.getsockname()[0]
    except OSError:
        out["ip"] = None
    out["light"] = "on" if _light_on else "off"
    return out


# music (Spotify)

WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

_SPOTIFY_KINDS = ("track", "artist", "album", "playlist")

# Step for direction-only volume words ("louder", "turn it down"...).
VOLUME_STEP = int(os.getenv("MUSIC_VOLUME_STEP", "10"))


def _spotify_error(res: dict) -> dict | None:
    """Translate Spotify API failures into voice-friendly messages."""
    status = res.get("status", 0)
    if status == -1:
        # Network/transport failure inside the spotify client (it never
        # raises; -1 is its "couldn't talk to Spotify" code).
        return {"error": "I couldn't reach Spotify just now - it may be a "
                         "network blip; try again in a moment."}
    if status == 0:
        return {"error": "Spotify is not linked. Open "
                         f"http://127.0.0.1:{WEB_PORT}/spotify/auth in a "
                         "browser to connect it."}
    if status == 401:
        return {"error": "Spotify authorization expired. Re-link at "
                         f"http://127.0.0.1:{WEB_PORT}/spotify/auth."}
    if status == 429:
        return {"error": "Spotify is rate-limiting this account right now - "
                         "wait a few seconds and try again."}
    if status == 403:
        msg = (res.get("json", {}).get("error", {}) or {}).get("message", "") or ""
        low = msg.lower()
        if "volume" in low or "restrict" in low:
            # e.g. the web player: Spotify accepts the account but the device
            # has no remotely controllable volume.
            return {"error": "This Spotify device does not support remote volume "
                             "control - the web player is the usual case. Use the "
                             "volume keys on the device itself."}
        return {"error": "Spotify Premium is required to control playback."}
    if status == 404:
        return {"error": "No active Spotify device found. Open Spotify on "
                         "your phone or computer and press play once, then "
                         "try again."}
    if status >= 400:
        msg = res.get("json", {}).get("error", {}).get("message", "")
        return {"error": f"Spotify said: {msg or status}"}
    return None


def _track_line(t: dict) -> str:
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    return f"{t['name']} — {artists}"


def _device_hint(name: str) -> str:
    """Voice-friendly hint when playback lands somewhere users overlook."""
    low = name.lower()
    if "web player" in low or any(b in low for b in ("safari", "chrome", "firefox", "edge")):
        return ("Playing in the Spotify WEB PLAYER browser tab - remind the "
                "user to bring that tab to the front and check its mute "
                "button; the desktop app and other apps stay silent.")
    return ""


# Playback state cache. The bridge polls music_state in its 1 Hz infra loop,
# so the mirror rides along cheap: knowing that music is playing lets the
# bridge tighten its mic gates so speaker-to-mic feedback (lyrics becoming
# phantom turns, music barging the bot) stays out. Every playback tool sets
# it instantly; _refresh_playback catches external changes (pause from the
# phone) without hammering the Spotify API.
PLAYBACK_REFRESH_SECS = 4.0
_playback = {"is_playing": False, "checked_at": 0.0}


def _set_playing(flag: bool) -> None:
    _playback["is_playing"] = flag
    _playback["checked_at"] = time.monotonic()


async def _refresh_playback() -> bool:
    """Re-read is_playing from Spotify if the cache is stale; never raises."""
    if not spotify.linked():
        return False
    if time.monotonic() - _playback["checked_at"] < PLAYBACK_REFRESH_SECS:
        return _playback["is_playing"]
    _playback["checked_at"] = time.monotonic()   # claim the slot even on failure
    try:
        res = await spotify.api("GET", "/me/player")
        if res["status"] == 200:
            _playback["is_playing"] = bool(res["json"].get("is_playing"))
        elif res["status"] == 204:
            _playback["is_playing"] = False
    except Exception:
        pass
    return _playback["is_playing"]


@mcp.tool()
async def music_state() -> dict:
    """Internal mirror: whether Spotify is currently playing.

    Not exposed to the LLM - the web bridge polls this (it is a ~ms cached
    read, refreshed from Spotify at most every PLAYBACK_REFRESH_SECS) to
    keep its music-aware mic gating in sync."""
    return {"is_playing": await _refresh_playback()}


@mcp.tool()
async def music_search(query: str, kind: str = "track") -> dict:
    """Search Spotify for tracks, artists, albums, or playlists.

    Args:
        query: What to search for, e.g. "Here Comes The Sun".
        kind: One of "track", "artist", "album", "playlist".
    """
    kind = kind if kind in _SPOTIFY_KINDS else "track"
    res = await spotify.api("GET", "/search", require_user=False, params={
        "q": query, "type": kind, "limit": 5})
    err = _spotify_error(res)
    if err:
        return err
    items = (res["json"].get(f"{kind}s", {}) or {}).get("items", [])
    if not items:
        return {"results": [], "note": f"No {kind} matching {query!r}"}
    out = []
    for it in items:
        if not it:                     # Spotify occasionally returns null entries
            continue
        row = {"name": it["name"], "uri": it["uri"]}
        if kind == "track":
            row["artist"] = ", ".join(a["name"] for a in it.get("artists", []))
        elif kind in ("album",):
            row["artist"] = ", ".join(a["name"] for a in it.get("artists", []))
        elif kind == "playlist":
            row["owner"] = it.get("owner", {}).get("name", "")
        out.append(row)
    return {"results": out}


async def _resolve(query: str, kind: str) -> dict:
    """Search and return the best match's uri and display name."""
    res = await spotify.api("GET", "/search", require_user=False, params={
        "q": query, "type": kind, "limit": 1})
    items = [it for it in (res["json"].get(f"{kind}s", {}) or {}).get("items", [])
             if it]                    # Spotify occasionally returns null entries
    if not items:
        return {"error": f"No {kind} matching {query!r} on Spotify."}
    it = items[0]
    name = it["name"]
    out = {"uri": it["uri"], "name": name}
    if kind == "track" and it.get("artists"):
        out["artist"] = it["artists"][0]["name"]
    return out


async def _start_playback(body: dict, device: str, fallback_name: str = "") -> dict:
    """PUT /me/player/play, auto-transferring to a lone device when needed."""
    qs = ""
    if device:
        d = await spotify.api("GET", "/me/player/devices")
        match = next((it for it in d.get("json", {}).get("devices", [])
                      if device.lower() in it["name"].lower()), None)
        if not match:
            return {"error": f"No Spotify device named {device!r}"}
        qs = f"?device_id={match['id']}"
    res = await spotify.api("PUT", f"/me/player/play{qs}", json=body)
    if res["status"] == 404 and not device:
        # nothing active: if exactly one device exists, move playback there
        d = await spotify.api("GET", "/me/player/devices")
        devs = d.get("json", {}).get("devices", [])
        if len(devs) == 1:
            res = await spotify.api(
                "PUT", f"/me/player/play?device_id={devs[0]['id']}", json=body)
        elif devs:
            names = ", ".join(it["name"] for it in devs)
            return {"error": f"Several Spotify devices are available ({names}) "
                             "- say which one to use."}
    err = _spotify_error(res)
    if err:
        return err
    _set_playing(True)
    # Read back what is actually on air. Spotify lags a beat in registering
    # the change, so poll briefly: one early read can still show the OLD
    # track. 0.25 s steps confirm in ~0.5-0.8 s (the old 6 x 0.35 s loop
    # made every play feel a second slower), and if it never confirms we
    # report the requested item rather than the stale one.
    want = (body.get("uris") or [None])[0]
    confirmed = False
    out = {"playing": fallback_name or "started"}
    dev_name = ""
    # 0.2 s first, then 0.2 s steps, 5 tries (~1.2 s max): Spotify registers
    # the change within ~0.3-0.9 s, and this wait sits directly in front of
    # the spoken acknowledgement. (Was 8 x 0.25 s = up to 2 s.)
    for n in range(5):
        await asyncio.sleep(0.2)
        now = await spotify.api("GET", "/me/player")
        if now["status"] != 200:
            break
        item = now["json"].get("item") or {}
        dev_name = (now["json"].get("device") or {}).get("name", "")
        if item:
            out = {"playing": _track_line(item)}
            if want is None or item.get("uri") == want:
                confirmed = True   # the requested track is on air
                break
    if dev_name:
        out["device"] = dev_name   # so the LLM can say WHERE it plays
        note = _device_hint(dev_name)
        if not confirmed and not note:
            name = fallback_name or "your request"
            note = (f"Spotify accepted the command on {dev_name} but has not "
                    f"confirmed {name} yet - check that the app on that "
                    "device is open, unlocked, and not muted.")
        if note:
            out["note"] = note
    return out


@mcp.tool()
async def music_play(query: str, kind: str = "track", device: str = "") -> dict:
    """Search Spotify and immediately play the best match on the user's
    Spotify device. Requires the user to be linked (one-time /spotify/auth)
    and Spotify Premium.

    Args:
        query: Song, artist, album, or playlist to play.
        kind: "track", "artist", "album", or "playlist" (default track).
        device: Optional device name to play on, e.g. "Living room".
    """
    kind = kind if kind in _SPOTIFY_KINDS else "track"
    hit = await _resolve(query, kind)
    if "error" in hit:
        return hit
    body = ({"uris": [hit["uri"]]} if kind == "track"
            else {"context_uri": hit["uri"]})
    out = await _start_playback(body, device)
    if "error" not in out:
        out["matched"] = hit["name"]
    return out


@mcp.tool()
async def music_pause() -> dict:
    """Pause Spotify playback."""
    res = await spotify.api("PUT", "/me/player/pause")
    if res.get("status") == 404:
        # "Stop the music" with nothing active is a no-op, not an error.
        return {"paused": False, "note": "Nothing is playing right now."}
    err = _spotify_error(res)
    if err:
        return err
    _set_playing(False)   # the mic gates react to music the instant it stops
    return {"paused": True}


@mcp.tool()
async def music_resume() -> dict:
    """Resume Spotify playback."""
    res = await spotify.api("PUT", "/me/player/play")
    err = _spotify_error(res)
    if err:
        return err
    _set_playing(True)
    return {"resumed": True}


@mcp.tool()
async def music_restart() -> dict:
    """Restart the currently playing track from the beginning."""
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"error": "Nothing is playing right now."}
    err = _spotify_error(res)
    if err:
        return err
    item = res["json"].get("item") or {}
    if not item:
        return {"error": "Nothing is playing right now."}
    put = await spotify.api("PUT", "/me/player/seek", params={"position_ms": 0})
    err = _spotify_error(put)
    if err:
        return err
    _set_playing(True)
    return {"restarted": _track_line(item)}


@mcp.tool()
async def music_skip(direction: str = "next") -> dict:
    """Skip to the next or previous track.

    Spotify silently ignores next/previous when the queue is empty (always
    the case right after playing a single song), so if the player does not
    change tracks this falls back to playing the next song by the same
    artist (or restarting the song, for "previous").

    Args:
        direction: "next" or "previous".
    """
    if direction not in ("next", "previous"):
        return {"error": "direction must be 'next' or 'previous'"}
    before = await spotify.api("GET", "/me/player")
    cur = ((before["json"] or {}).get("item") or {}) if before["status"] == 200 else {}
    cur_uri = cur.get("uri", "")
    res = await spotify.api("POST", f"/me/player/{direction}")
    err = _spotify_error(res)
    if err and res.get("status") != 403:
        return err
    # (403 "Restriction violated" happens on some tracks — fall through to
    # the fallback below instead of reporting a raw Spotify error.)
    # Verify the player actually changed tracks — Spotify accepts the call
    # even when there is nothing to skip to, so trust the read-back, not the
    # 2xx response. 4 x 0.3 s is enough: Spotify registers a skip within
    # ~0.3-0.9 s, and the old 5 x 0.8 s loop added up to 4 s of dead wait
    # whenever the queue was empty (the 7-8 s skips in the logs).
    for _ in range(4):
        await asyncio.sleep(0.3)
        now = await spotify.api("GET", "/me/player")
        if now["status"] != 200:
            break
        item = (now["json"] or {}).get("item") or {}
        if item and item.get("uri") != cur_uri:
            out = {"playing": _track_line(item)}
            dev = ((now["json"] or {}).get("device") or {}).get("name", "")
            if dev:
                out["device"] = dev
            _set_playing(True)
            return out
    # Nothing changed: the queue was empty. Fall back.
    if not cur:
        return {"error": "Nothing is playing right now - play a song first."}
    if direction == "previous":
        # "Previous" with an empty queue: restart the current track.
        await spotify.api("PUT", "/me/player/seek", params={"position_ms": 0})
        _set_playing(True)
        return {"restarted": _track_line(cur),
                "note": "No previous track in the queue - restarted the current song."}
    # Try each credited artist in turn (featured artists often have no
    # catalog of their own), then fall back to track-name search. The old
    # queries were malformed: `track:` takes a track NAME, so searching
    # `track:<artist>` returned nothing half the time.
    items = []
    for a in cur.get("artists") or []:
        name = a.get("name", "")
        if not name:
            continue
        res2 = await spotify.api("GET", "/search", require_user=False, params={
            "q": f'artist:"{name}"', "type": "track", "limit": 10})
        items = [it for it in ((res2["json"].get("tracks", {}) or {}).get("items", []))
                 if it and it["uri"] != cur_uri]
        if items:
            break
    if not items:
        res2 = await spotify.api("GET", "/search", require_user=False, params={
            "q": f'{cur.get("name", "")}',
            "type": "track", "limit": 6})
        items = [it for it in ((res2["json"].get("tracks", {}) or {}).get("items", []))
                 if it and it["uri"] != cur_uri]
    if not items:
        return {"skipped": direction,
                "note": "No other tracks found by this artist on Spotify - "
                        "ask me to play a different song."}
    nxt = items[0]
    play = await spotify.api("PUT", "/me/player/play",
                             json={"uris": [nxt["uri"]]})
    err = _spotify_error(play)
    if err:
        return err
    _set_playing(True)
    out = {"playing": _track_line(nxt),
           "note": "Queue was empty, so played the next song by the same artist."}
    return out


def _volume_target(raw: str) -> tuple[str, int | None, str]:
    """Parse a spoken volume request -> (sign, value, error).

    sign is "+" or "-" for relative moves, "" for an absolute set.
    value is None only when the parser could not understand the request."""
    if re.search(r"\bup\s*to\s+(\d{1,3})\b", raw):
        # "turn it up to 40" = set 40, not a + relative bump. The old parser
        # matched both "up" and 40 and added 40 on top of the current level.
        to_abs = re.search(r"\b(?:up|down)?\s*to\s+(\d{1,3})\b", raw)
        return "", int(to_abs.group(1)), ""
    if re.search(r"\b(max|maximum|full|fuller|blasting|cranked|loudest)\b", raw):
        return "", 100, ""
    if re.search(r"\b(mute|silent|silence|sound off|zero)\b", raw):
        return "", 0, ""
    num = re.search(r"(\d{1,3})", raw)
    up = re.search(r"\b(up|louder|higher|more|boost|blare|blast|crank)\b", raw)
    down = re.search(r"\b(down|quieter|quiet|softer|soft|lower|less|dip)\b", raw)
    if num is not None:
        val = int(num.group(1))
        if val > 100:
            return "", None, f"Volume {val} percent is out of range - the maximum is 100."
        if "%" in raw or "percent" in raw:
            return "", val, ""           # "40 percent" is always absolute
        if re.search(r"\+", raw):
            return "+", val, ""
        if re.search(r"[-]", raw) or down and not up:
            return "-", val, ""
        if up and not down:
            return "+", val, ""
        return "", val, ""
    if up and not down:
        return "+", VOLUME_STEP, ""      # "turn it up" — no number given
    if down and not up:
        return "-", VOLUME_STEP, ""
    if up and down:
        return "", None, "I got both 'up' and 'down' in there - say one direction."
    return "", None, ""                  # fall through: report current volume


async def _spotify_current_volume() -> dict:
    """GET /me/player -> {"error"} | {"volume_pct": int|None}."""
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"volume_pct": None,
                "note": "No active Spotify device, so there is no volume to read."}
    err = _spotify_error(res)
    if err:
        return err
    pct = (res["json"].get("device") or {}).get("volume_percent")
    return {"volume_pct": pct,
            "note": "That is the device's own volume, as reported by Spotify."}


@mcp.tool()
async def music_volume(level: str = "") -> dict:
    """Set, adjust, or read the Spotify playback volume.

    Args:
        level: An absolute percentage like "40" or "40 percent", a relative
            adjustment like "+10", "-20", "up", "up 10" or "down 5", a word
            like "max", "mute", "louder" or "softer", or "" to just report
            the current volume.
    """
    raw = level.strip().lower().replace("%", "").replace("percent", " ")
    if not raw:
        return await _spotify_current_volume()
    sign, val, err_msg = _volume_target(raw)
    if val is None:
        return {"error": err_msg or
                f"I could not understand the volume '{level}'. Say a "
                "percentage like '40', an adjustment like 'up' or 'down 5', "
                "'max', or 'mute'."}
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"error": "No active Spotify device to change the volume on. "
                         "Open Spotify somewhere and press play once, then "
                         "try again."}
    err = _spotify_error(res)
    if err:
        return err
    cur = (res["json"].get("device") or {}).get("volume_percent")
    if sign == "+":
        target = min(100, (cur or 0) + val)
    elif sign == "-":
        target = max(0, (cur or 0) - val)
    else:
        target = val
    put = await spotify.api("PUT", "/me/player/volume",
                            params={"volume_percent": target})
    err = _spotify_error(put)
    if err:
        return err
    verb = "raised" if sign == "+" else ("lowered" if sign == "-" else "set")
    out = {"volume_pct": target, "action": verb,
           "note": f"Volume {verb} to {target} percent."}
    if cur is not None and target == cur:
        out["note"] += " It was already at that level."
    return out


@mcp.tool()
async def music_shuffle(state: str = "") -> dict:
    """Turn Spotify shuffle on or off, or report its state.

    Args:
        state: "on" or "off"; "" to report whether shuffle is on.
    """
    state = state.strip().lower()
    if state not in ("", "on", "off"):
        return {"error": "Shuffle takes 'on' or 'off'."}
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"shuffle": "unknown",
                "note": "No active Spotify device, so shuffle state is "
                        "unknown."}
    err = _spotify_error(res)
    if err:
        return err
    if not state:
        on = bool(res["json"].get("shuffle_state", False))
        return {"shuffle": "on" if on else "off",
                "note": f"Shuffle is {'on' if on else 'off'}."}
    put = await spotify.api("PUT", "/me/player/shuffle",
                            params={"state": state})
    err = _spotify_error(put)
    if err:
        return err
    return {"shuffle": state, "note": f"Shuffle turned {state}."}


@mcp.tool()
async def music_repeat(mode: str = "") -> dict:
    """Change the Spotify repeat mode, or report it.

    Args:
        mode: "off", "track" (repeat the current song), "context" (repeat the
            album/playlist), or "" to cycle to the next mode.
    """
    mode = mode.strip().lower()
    if mode and mode not in ("off", "track", "context"):
        return {"error": "Repeat takes 'off', 'track', or 'context'."}
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"repeat": "unknown",
                "note": "No active Spotify device, so repeat state is "
                        "unknown."}
    err = _spotify_error(res)
    if err:
        return err
    cur = res["json"].get("repeat_state", "off")
    if not mode:
        order = {"off": "context", "context": "track", "track": "off"}
        mode = order.get(cur, "context")
    put = await spotify.api("PUT", "/me/player/repeat", params={"state": mode})
    err = _spotify_error(put)
    if err:
        return err
    label = {"off": "off", "track": "on for the current song",
             "context": "on for the album or playlist"}[mode]
    return {"repeat": mode, "note": f"Repeat is now {label}."}


@mcp.tool()
async def music_now_playing() -> dict:
    """Report the track currently playing on Spotify."""
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {"playing": False, "note": "Nothing is playing right now."}
    err = _spotify_error(res)
    if err:
        return err
    item = res["json"].get("item") or {}
    if not item:
        return {"playing": False, "note": "Nothing is playing right now."}
    out = {"playing": _track_line(item),
           "progress_ms": res["json"].get("progress_ms"),
           "duration_ms": item.get("duration_ms"),
           "paused": not res["json"].get("is_playing", False)}
    dev_name = (res["json"].get("device") or {}).get("name", "")
    if dev_name:
        out["device"] = dev_name   # so the LLM can say WHERE it plays
        note = _device_hint(dev_name)
        if note:
            out["note"] = note
    # (Auto-queueing more by the same artist happens in music_play, where the
    # played URI is known - it cannot happen here.)
    return out


@mcp.tool()
async def music_queue(query: str) -> dict:
    """Add a song to the Spotify play queue.

    Args:
        query: The song to queue.
    """
    hit = await _resolve(query, "track")
    if "error" in hit:
        return hit
    res = await spotify.api("POST", "/me/player/queue",
                            params={"uri": hit["uri"]})
    err = _spotify_error(res)
    if err:
        return err
    return {"queued": hit["name"]}


@mcp.tool()
async def music_devices() -> dict:
    """List the Spotify devices available for playback."""
    res = await spotify.api("GET", "/me/player/devices")
    err = _spotify_error(res)
    if err:
        return err
    devs = [{"name": it["name"], "type": it["type"],
             "active": it.get("is_active", False)}
            for it in res["json"].get("devices", [])]
    return {"devices": devs}



if __name__ == "__main__":
    setup_logging("tools")
    logger.info(f"Tools MCP server on http://{HOST}:{PORT}/mcp")
    mcp.run("streamable-http", host=HOST, port=PORT)


