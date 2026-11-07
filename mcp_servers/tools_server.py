"""Real-world task tools as an MCP server.

Every tool here uses free services or the Pi itself, with no API keys:

  get_time        local date/time in any timezone
  get_weather     current weather + today's forecast (Open-Meteo)
  web_search      web search (DuckDuckGo via the `ddgs` package)
  wikipedia       short encyclopedia summary (Wikipedia REST API)
  set_timer etc.  countdown timers; the agent announces them when they finish
  notes           add / list / delete / clear a persistent to-do / notes list
  system_status   Pi CPU temperature, load, memory, uptime, IP
  set_lights      lights on/off + brightness + colour via Home Assistant
  set_thermostat  thermostat target temperature via Home Assistant
  get_temperature current room temperature (+ target) via Home Assistant
                  (all three need HASS_URL + HASS_TOKEN in .env)
  call_contact    reach a contact on Telegram with a call-back request
                  (bots can't initiate calls; the contact gets a Telegram
                  message asking them to call back)
  send_message    send a Telegram message (recipients are learned from
                  incoming bot messages; unknown ones go to the owner's
                  chat - needs TELEGRAM_BOT_TOKEN in .env)
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
import sys
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
from web.core.config import DEVICE_PROFILE

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

# (lat, lon) -> (monotonic timestamp, formatted result). The forecast itself
# is a second full round trip (~300-800 ms on the Pi) that "how's the weather?"
# repeats within seconds; weather doesn't change meaningfully in a minute, so
# serve repeats from cache. Failures are never cached, so a transient network
# error still retries on the next call.
WX_CACHE_TTL = 60.0
_wx_cache: dict[tuple[float, float], tuple[float, dict]] = {}


@asynccontextmanager
async def _lifespan(_server):
    _tg_start_poller()  # learn Telegram contacts from incoming messages
    yield {}
    _tg_stop_poller()
    if _http is not None and not _http.is_closed:
        await _http.aclose()
    await spotify.close()  # release the pooled Spotify client too


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
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
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
        wx_key = (lat, lon)
        hit_wx = _wx_cache.get(wx_key)
        if hit_wx is not None and time.monotonic() - hit_wx[0] < WX_CACHE_TTL:
            return hit_wx[1]
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
    result = {
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
    _wx_cache[(lat, lon)] = (time.monotonic(), result)
    return result


# search / knowledge


def _plain_text(value: str) -> str:
    """Strip markdown emphasis and stray HTML tags from search snippets.

    DuckDuckGo snippets routinely carry **bold** / *italic* markers and the
    occasional leftover HTML tag. Left in the tool output, the LLM echoes
    them into spoken replies. Clean at the source so the model never sees
    them.
    """
    if not value:
        return ""
    text = value
    # markdown bold: **text** / __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # markdown italic: *text* / _text_ (guarded so snake_case survives)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"\1", text)
    # leftover HTML tags, e.g. <b>, </span>
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Engines raced in parallel for web_search. No single engine is reliable:
# DuckDuckGo (the primary) intermittently times out or serves a bot-check page
# (HTTP 202) that parses to zero results, so we fan out and take the first
# engine that actually returns hits. These are the valid ddgs 9.x text
# backends - "bing" is NOT one of them (it silently falls back to "auto",
# which fans out to every engine including the encyclopedias).
_SEARCH_BACKENDS = ("duckduckgo", "brave", "mojeek", "startpage", "yahoo")
# Per-engine cap. Kept under the web bridge's 12 s tool timeout (with margin
# for MCP transport) so a hung engine can't blow the whole turn's budget.
_SEARCH_DEADLINE = 8.0


@mcp.tool()
async def web_search(query: str, max_results: int = 4) -> dict:
    """Search the web for current information (news, prices, facts, events).

    Races several search engines in parallel and returns the first that
    produces results, so one slow or challenged engine no longer kills the
    lookup.

    Args:
        query: What to search for.
        max_results: Number of results to return (1-8).
    """
    from ddgs import DDGS

    max_results = max(1, min(int(max_results or 4), 8))
    query = (query or "").strip()
    if not query:
        return {"error": "Give me something to search for."}

    def _search_one(backend: str) -> list:
        # timeout=4: ddgs's 5 s default would eat the whole shared budget on
        # the first engine alone, leaving the fallbacks no time to answer.
        return DDGS(timeout=4).text(query, max_results=max_results, backend=backend)

    # Sequential with a shared deadline: firing every engine at once hammers
    # them all (and ourselves) in parallel - the burst itself trips their
    # rate limiters, which is exactly the failure we are trying to dodge.
    # Try the primary first; only fall through when it yields nothing, and
    # stop as soon as the clock runs out.
    deadline = time.monotonic() + _SEARCH_DEADLINE
    for backend in _SEARCH_BACKENDS:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break
        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(_search_one, backend),
                timeout=remaining,
            )
        except Exception:
            # Timeout, "No results found.", a bot-check page, or a transport
            # error - all mean "nothing from this engine", so try the next.
            continue
        if hits:
            return {
                "query": query,
                "engine": backend,
                "results": [
                    {
                        "title": _plain_text(h.get("title") or ""),
                        "snippet": _plain_text(h.get("body") or ""),
                        "url": h.get("href"),
                    }
                    for h in hits
                    if h
                ],
            }
    return {
        "query": query,
        "results": [],
        "note": f"No web results for {query!r} - try different wording",
    }


@mcp.tool()
async def wikipedia(topic: str) -> dict:
    """Get a short encyclopedia summary of a person, place, thing or concept.

    Args:
        topic: The subject, e.g. "Jose Rizal" or "Raspberry Pi".
    """
    from urllib.parse import quote

    # Wikimedia's robot policy wants a descriptive UA with contact info; a bare
    # product string gets intermittently 403-throttled. This one complies.
    wiki_headers = {
        "User-Agent": "LavaVoiceAgent/1.0 (personal voice assistant; contact: charrieannenacor@gmail.com)"
    }
    topic = (topic or "").strip()
    if not topic:
        return {"error": "Give me a topic to look up."}

    http = get_http()
    search_json = None
    # Bursty throttling: a 403/429 here is usually transient, so retry with a
    # short backoff before giving up (two attempts total, ~1.5 s worst case).
    for attempt in range(2):
        try:
            search = await http.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 1,
                    "format": "json",
                },
                headers=wiki_headers,
            )
            if search.status_code in (403, 429):
                raise httpx.HTTPStatusError(
                    f"throttled ({search.status_code})",
                    request=search.request,
                    response=search,
                )
            search.raise_for_status()
            search_json = search.json()
            break
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
            if attempt == 0:
                await asyncio.sleep(1.5)
                continue
            return {"error": f"Wikipedia lookup failed: {str(e)[:160]}"}
    if search_json is None:
        return {"error": "Wikipedia lookup failed: no response"}

    if not search_json[1]:
        return {"error": f"No Wikipedia article found for {topic}"}
    title = search_json[1][0]
    # Quote the title: REST paths break on spaces/slashes/special chars.
    page_path = quote(title.replace(" ", "_"))
    try:
        page_resp = await http.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_path}",
            headers=wiki_headers,
        )
        page_resp.raise_for_status()
        page = page_resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
        # The opensearch description is a decent fallback if the summary dies.
        desc = (search_json[2] or [""])[0].strip()
        if desc:
            return {"title": title, "summary": _plain_text(desc)}
        return {"error": f"Wikipedia summary lookup failed: {str(e)[:160]}"}
    summary = (page.get("extract") or "").strip()
    if not summary:
        desc = (search_json[2] or [""])[0].strip()
        summary = desc or "Wikipedia has an article for this but no short summary."
    return {"title": page.get("title", title), "summary": _plain_text(summary)}


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
        seconds = max(1, int(seconds))  # "in 0 seconds" or negatives rot in place
    except (TypeError, ValueError):
        return {"error": f"Timer length {seconds!r} is not a number of seconds."}
    timer_id = uuid.uuid4().hex[:6]
    _timers[timer_id] = {
        "label": label,
        "seconds": seconds,
        "ends_at": time.time() + seconds,
    }
    _timers[timer_id]["task"] = asyncio.create_task(_run_timer(timer_id))
    return {"id": timer_id, "label": label, "seconds": seconds, "status": "started"}


@mcp.tool()
def list_timers() -> dict:
    """List running timers and how long each has left."""
    return {
        "timers": [
            {
                "id": k,
                "label": v["label"],
                "seconds_left": max(0, round(v["ends_at"] - time.time())),
            }
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
    # running: how many timers are still counting down. The bridge polls at 1 Hz
    # while this is > 0 and backs off to 5 s otherwise, so the common no-timer
    # state costs a round trip every 5 s instead of every second.
    return {"finished": done, "running": len(_timers)}


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
        return {
            "deleted": [],
            "remaining": notes,
            "error": f"No note matching {match!r}",
        }
    _save_notes(keep)
    return {"deleted": deleted, "remaining": keep}


@mcp.tool()
def clear_notes() -> dict:
    """Delete all saved notes."""
    n = len(_load_notes())
    _save_notes([])
    return {"deleted": n}


# smart home (Home Assistant: lights + thermostat)
#
# One self-hosted hub covers three command families (lights on/off, dim /
# colour, thermostat). Point HASS_URL at any Home Assistant instance (it can
# be the same Raspberry Pi) and mint a long-lived access token under
# Profile -> Security. With no HASS_URL/HASS_TOKEN in .env every tool below
# returns an honest "not connected" error instead of pretending.

HASS_URL = (os.getenv("HASS_URL") or "").strip().rstrip("/")
HASS_TOKEN = (os.getenv("HASS_TOKEN") or "").strip()
HASS_LIGHT_ENTITY = (os.getenv("HASS_LIGHT_ENTITY") or "light.living_room").strip()
HASS_CLIMATE_ENTITY = (os.getenv("HASS_CLIMATE_ENTITY") or "").strip()

# /api/states is a full-home dump (~tens of KB); entity discovery caches it
# for a short window so "dim the living room" + "turn it off" back to back
# costs one poll, not two.
STATES_CACHE_TTL = 15.0
_states_cache: tuple[float, list] | None = None

# Spoken white tones land on kelvin; named colours pass through to Home
# Assistant, which accepts any CSS colour name.
_KELVIN_BY_TONE = {
    "warm": 2700,
    "warm white": 2700,
    "soft white": 2700,
    "cool": 5000,
    "cool white": 5000,
    "daylight": 5600,
    "white": 4000,
}


def _hass_unconfigured() -> dict | None:
    if not HASS_URL or not HASS_TOKEN:
        return {
            "error": "No smart home hub connected - add HASS_URL and "
            "HASS_TOKEN to .env (Home Assistant long-lived access token) "
            "to enable lights and thermostat."
        }
    return None


async def _hass(method: str, path: str, **kw) -> tuple[int, object]:
    """Call the Home Assistant REST API; returns (status, parsed body)."""
    http = get_http()
    r = await http.request(
        method,
        f"{HASS_URL}{path}",
        headers={"Authorization": f"Bearer {HASS_TOKEN}"},
        **kw,
    )
    try:
        body = r.json()
    except ValueError:
        body = None
    return r.status_code, body


def _hass_status_error(status: int) -> dict | None:
    if status == 401:
        return {"error": "Home Assistant rejected the access token - check HASS_TOKEN."}
    if status == 404:
        return {"error": f"Home Assistant at {HASS_URL} did not find that endpoint - check HASS_URL."}
    if status >= 400:
        return {"error": f"Home Assistant said: {status}"}
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


async def _hass_entities(prefix: str, room: str) -> tuple[list[dict], str | None]:
    """Match entities of one domain (e.g. "light") against a room name.

    Returns (matches, error). A match needs the room word inside the entity
    id or its friendly name; an empty room matches everything in the domain.
    """
    global _states_cache
    now = time.monotonic()
    if _states_cache is None or now - _states_cache[0] > STATES_CACHE_TTL:
        status, states = await _hass("GET", "/api/states")
        if not isinstance(states, list):
            err = _hass_status_error(status) or {
                "error": f"Could not read Home Assistant's entity list (status {status})."
            }
            return [], err
        _states_cache = (now, states)
    want = _norm(room)
    matches = []
    for s in _states_cache[1]:
        eid = s.get("entity_id", "")
        if not eid.startswith(f"{prefix}."):
            continue
        friendly = (s.get("attributes") or {}).get("friendly_name", "")
        if not want or want in _norm(eid.split(".", 1)[1]) or want in _norm(friendly):
            matches.append({"entity_id": eid, "name": friendly or eid})
    return matches, None


@mcp.tool()
async def set_lights(
    room: str = "living room",
    on: bool = True,
    brightness_pct: int = 0,
    color: str = "",
) -> dict:
    """Turn lights on or off, optionally setting brightness and colour.

    Needs a Home Assistant hub (HASS_URL + HASS_TOKEN in .env).

    Args:
        room: Room or light name, e.g. "living room" or "desk lamp".
        on: True to turn the lights on, False to turn them off.
        brightness_pct: Brightness 1-100; 0 leaves it unchanged.
        color: Optional colour or white tone: "warm", "cool", "red", "blue", ...
    """
    unconfigured = _hass_unconfigured()
    if unconfigured:
        return unconfigured
    room = (room or "").strip() or "living room"
    if brightness_pct:
        brightness_pct = max(1, min(100, int(brightness_pct)))
    color = (color or "").strip().lower()
    matches, err = await _hass_entities("light", room)
    if err:
        return err
    if not matches:
        # No name hit: fall back to the configured default entity so "the
        # lights" still works when only one room is automated.
        if HASS_LIGHT_ENTITY:
            matches = [{"entity_id": HASS_LIGHT_ENTITY, "name": room}]
        else:
            return {"error": f"No lights matching {room!r} in Home Assistant."}
    ids = [m["entity_id"] for m in matches]
    label = " + ".join(dict.fromkeys(m["name"].title() for m in matches))
    service = "turn_off" if not on else "turn_on"
    data: dict = {"entity_id": ids}
    if on:
        if brightness_pct:
            data["brightness_pct"] = brightness_pct
        if color:
            kelvin = _KELVIN_BY_TONE.get(color)
            if kelvin:
                data["color_temp_kelvin"] = kelvin
            else:
                data["color_name"] = color
    status, _ = await _hass("POST", f"/api/services/light/{service}", json=data)
    err = _hass_status_error(status)
    if err:
        return err
    out: dict = {"lights": label, "on": on}
    if on and brightness_pct:
        out["brightness_pct"] = brightness_pct
    if on and color:
        out["color"] = color
    return out


@mcp.tool()
async def set_thermostat(temperature: float, room: str = "", unit: str = "C") -> dict:
    """Set the thermostat's target temperature.

    Needs a Home Assistant hub (HASS_URL + HASS_TOKEN in .env).

    Args:
        temperature: Target temperature, e.g. 22.
        room: Optional room / thermostat name; empty uses the only one.
        unit: "C" or "F" - the number is sent as given.
    """
    unconfigured = _hass_unconfigured()
    if unconfigured:
        return unconfigured
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return {"error": f"{temperature!r} is not a temperature."}
    if temperature.is_integer():
        temperature = int(temperature)  # 22.0 -> 22: "set to 22 °C", not "22.0"
    unit = (unit or "C").strip().upper()
    if unit not in ("C", "F"):
        unit = "C"
    if unit == "C" and not 5 <= temperature <= 35:
        return {"error": f"{temperature} C is outside a sane thermostat range (5-35)."}
    if unit == "F" and not 41 <= temperature <= 95:
        return {"error": f"{temperature} F is outside a sane thermostat range (41-95)."}
    matches, err = await _hass_entities("climate", room)
    if err:
        return err
    if not matches:
        if HASS_CLIMATE_ENTITY:
            matches = [{"entity_id": HASS_CLIMATE_ENTITY, "name": room or "Thermostat"}]
        else:
            return {"error": f"No thermostat matching {room or 'this home'!r} in Home Assistant."}
    if len(matches) > 1 and room:
        matches = matches[:1]  # a spoken room name targets one thermostat
    ids = [m["entity_id"] for m in matches]
    label = matches[0]["name"].title() if matches[0]["name"] else "Thermostat"
    status, _ = await _hass(
        "POST",
        "/api/services/climate/set_temperature",
        json={"entity_id": ids, "temperature": temperature},
    )
    err = _hass_status_error(status)
    if err:
        return err
    return {"thermostat": label, "temperature": temperature, "unit": unit}


@mcp.tool()
async def get_temperature(room: str = "") -> dict:
    """Read the current temperature from the thermostat.

    Reports the temperature the room is AT right now (and the target, if the
    thermostat exposes one). Use this for "what's the temperature?", "how
    hot is it in here?" - set_thermostat only changes the target.

    Needs a Home Assistant hub (HASS_URL + HASS_TOKEN in .env).

    Args:
        room: Optional room / thermostat name; empty uses the only one.
    """
    unconfigured = _hass_unconfigured()
    if unconfigured:
        return unconfigured
    matches, err = await _hass_entities("climate", room)
    if err:
        return err
    if not matches:
        if HASS_CLIMATE_ENTITY:
            matches = [{"entity_id": HASS_CLIMATE_ENTITY, "name": room or "Thermostat"}]
        else:
            return {"error": f"No thermostat matching {room or 'this home'!r} in Home Assistant."}
    if len(matches) > 1 and room:
        matches = matches[:1]  # a spoken room name targets one thermostat

    # Re-read just the matched entities (fresh, not from the 15 s discovery
    # cache - a temperature read should reflect the hub right now).
    status, states = await _hass("GET", "/api/states")
    if not isinstance(states, list):
        return _hass_status_error(status) or {
            "error": f"Could not read Home Assistant's entity list (status {status})."
        }
    by_id = {s.get("entity_id"): s for s in states if isinstance(s, dict)}

    # climate.state is the HVAC MODE ("heating", "off", ...) - the numbers are
    # in attributes: current_temperature = room temp now, temperature = target.
    temps, units, targets, modes, names = [], [], [], [], []
    for m in matches:
        st = by_id.get(m["entity_id"])
        if not st:
            continue
        attrs = st.get("attributes") or {}
        unit = attrs.get("temperature_unit", "°C")
        cur = attrs.get("current_temperature")
        tgt = attrs.get("temperature")
        if isinstance(cur, (int, float)):
            temps.append(float(cur))
            units.append(unit)
        if isinstance(tgt, (int, float)):
            targets.append(float(tgt))
        if isinstance(st.get("state"), str):
            modes.append(st["state"])
        names.append(m["name"] or m["entity_id"].split(".", 1)[1].replace("_", " "))
    if not temps:
        return {
            "error": f"{names[0] if names else 'The thermostat'} reports no temperature reading."
        }

    def _fmt(v: float) -> int | float:
        return int(v) if float(v).is_integer() else round(v, 1)

    out: dict = {
        "thermostat": names[0],
        "current": _fmt(sum(temps) / len(temps)),
        "unit": units[0],
    }
    if len(names) > 1:
        out["rooms"] = {n: _fmt(t) for n, t in zip(names, temps)}
    if targets:
        out["target"] = _fmt(targets[0])
    if modes:
        out["mode"] = modes[0]
    return out


# calls / messages - Telegram machinery (registry, poller, tools) lives
# below, after _resolve_contact.

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_PHONE_RE = re.compile(r"^\+?[\d\s()\-]{3,}$")


async def _resolve_contact(contact: str) -> dict:
    """Turn a name / number / address into {"display", "dial"} or {"error"}."""
    c = (contact or "").strip()
    if not c:
        return {"error": "Who should I call?"}
    if _EMAIL_RE.match(c):
        return {"display": c, "dial": c}
    if _PHONE_RE.match(c) and any(ch.isdigit() for ch in c):
        return {"display": c, "dial": re.sub(r"[^\d+]", "", c)}
    if sys.platform != "darwin":
        return {
            "error": f"I can only call from a Mac - {c!r} needs a number "
            "or an email address here."
        }
    # A name: ask Contacts.app for the first phone number (macOS prompts
    # for Contacts access on the first call).
    script = (
        'tell application "Contacts" to get value of phone 1 of '
        f'(first person whose name contains "{c.replace(chr(34), "")}")'
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, stderr = await asyncio.wait_for(proc.communicate(), 8)
    except Exception as e:
        return {"error": f"Contacts lookup failed: {e}"}
    if proc.returncode != 0:
        why = (stderr or b"").decode(errors="replace").strip()
        return {
            "error": f"I could not find a number for {c!r} in Contacts"
            + (f" ({why.splitlines()[0][:80]})" if why else "")
            + " - give me a phone number or email address."
        }
    digits = re.sub(r"[^\d+]", "", out.decode(errors="replace").strip())
    if not digits:
        return {"error": f"{c!r} has no phone number in Contacts."}
    return {"display": c, "dial": digits}


# --- Telegram ---------------------------------------------------------------
# Calls and messages both go out through the Telegram Bot API. Two platform
# limits shape the design (both verified against the Bot API docs):
#
#   1. Bots cannot initiate calls - calls are peer-to-peer and the Bot API
#      has no call method. "Call X" therefore sends X a call-BACK request.
#   2. A bot can only message a chat that has messaged IT first (or a group
#      it was added to). There is no phone-number -> chat_id lookup, so the
#      bot learns its contacts from incoming messages: a background
#      getUpdates poller records each sender (name, username, phone if
#      present) in a local registry, and the tools resolve names against
#      that registry first, then Contacts.app.
#
# TELEGRAM_BOT_TOKEN is required for either tool. TELEGRAM_CHAT_ID is the
# OWNER's chat id - the default destination when a name resolves nowhere
# else (single-user setups: everyone talks to the bot in the owner's chat).
# Without the token the tools return an honest "not configured" error.

TG_REGISTRY_FILE = ROOT / "logs" / "telegram_contacts.json"
TG_POLL_INTERVAL = 2.0  # seconds between getUpdates polls
TG_LONG_POLL = 25  # server-side long-poll seconds (max 50)

_tg_registry: dict[str, dict] = {}  # key -> {name, username, phone, chat_id}
_tg_poll_task: asyncio.Task | None = None


def _tg_config() -> dict | None:
    """Return {"token", "owner_chat_id"} from the environment, or None."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    return {
        "token": token,
        "owner_chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    }


def _tg_result(resp: httpx.Response) -> tuple[bool, dict]:
    """Parse a Bot API response into (ok, payload).

    Telegram returns 200 with {"ok": false, ...} for logical failures, so
    the status code alone is not enough - both are checked.
    """
    try:
        payload = resp.json()
    except Exception:
        return False, {"description": f"HTTP {resp.status_code}"}
    if resp.status_code >= 400 or not payload.get("ok", False):
        desc = payload.get("description") or f"HTTP {resp.status_code}"
        return False, {"description": str(desc)[:200]}
    return True, payload


def _tg_load_registry() -> None:
    """Load the contact registry from disk (corrupt file -> empty)."""
    global _tg_registry
    try:
        data = json.loads(TG_REGISTRY_FILE.read_text())
        if isinstance(data, dict):
            _tg_registry = data
    except (OSError, ValueError):
        _tg_registry = {}


def _tg_save_registry() -> None:
    try:
        TG_REGISTRY_FILE.write_text(json.dumps(_tg_registry, indent=2))
    except OSError as e:
        logger.warning("[tools] could not save Telegram registry: {}", e)


def _tg_register(chat: dict) -> None:
    """Record an incoming-message sender so later commands can reach them."""
    first = (chat.get("first_name") or "").strip()
    last = (chat.get("last_name") or "").strip()
    name = " ".join(x for x in (first, last) if x)
    phone = (chat.get("phone_number") or "").strip()
    key = str(chat.get("id"))
    entry = _tg_registry.get(key, {})
    if name:
        entry["name"] = name
    if chat.get("username"):
        entry["username"] = str(chat["username"])
    if phone:
        entry["phone"] = phone
    entry["chat_id"] = key
    _tg_registry[key] = entry
    _tg_save_registry()


async def _tg_poll_updates() -> None:
    """Background loop: consume getUpdates and register each sender.

    Runs for the life of the tools server when a bot token is configured.
    A dead network or a revoked token must never kill the server, so every
    failure just waits and retries.
    """
    cfg = _tg_config()
    if cfg is None:
        return
    _tg_load_registry()
    offset = 0
    http = get_http()
    while True:
        try:
            resp = await http.post(
                f"https://api.telegram.org/bot{cfg['token']}/getUpdates",
                data={
                    "offset": offset,
                    "timeout": TG_LONG_POLL,
                    "allowed_updates": json.dumps(["message"]),
                },
                timeout=TG_LONG_POLL + 10,
            )
            ok, payload = _tg_result(resp)
            if not ok:
                # 409 = another instance is polling (double start); 401 =
                # bad token. Both are config problems, not code bugs.
                logger.warning(
                    "[tools] Telegram getUpdates failed: {}",
                    payload.get("description"),
                )
                await asyncio.sleep(10)
                continue
            for upd in payload.get("result", []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or {}
                chat = msg.get("chat")
                if chat:
                    _tg_register(chat)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("[tools] Telegram poll hiccup: {}", e)
            await asyncio.sleep(TG_POLL_INTERVAL)


def _tg_start_poller() -> None:
    """Start the updates poller once (idempotent)."""
    global _tg_poll_task
    if _tg_poll_task is None or _tg_poll_task.done():
        _tg_poll_task = asyncio.create_task(_tg_poll_updates())


def _tg_stop_poller() -> None:
    global _tg_poll_task
    if _tg_poll_task is not None and not _tg_poll_task.done():
        _tg_poll_task.cancel()
    _tg_poll_task = None


def _tg_find(name: str) -> dict | None:
    """Match a spoken name against the registry (substring, case-insens.)."""
    n = (name or "").strip().lower()
    if not n:
        return None
    for entry in _tg_registry.values():
        hay = " ".join(
            x for x in (entry.get("name"), entry.get("username"), entry.get("phone")) if x
        ).lower()
        if n in hay:
            return entry
    return None


async def _tg_resolve(recipient: str) -> dict:
    """Resolve a recipient to a Telegram destination.

    Order: registry (learned from incoming messages) -> Contacts.app number
    (only useful when the person has also messaged the bot, since the bot
    needs a chat_id) -> the owner's own chat (TELEGRAM_CHAT_ID).
    Returns {"chat_id", "display"} or {"error"}.
    """
    raw = (recipient or "").strip()
    if not raw:
        return {"error": "Who should I message?"}
    cfg = _tg_config()
    if cfg is None:
        return {
            "error": (
                "Telegram isn't configured - add TELEGRAM_BOT_TOKEN to .env "
                "(talk to @BotFather for the token). People become reachable "
                "once they have messaged the bot at least once."
            )
        }
    # 1) Learned registry entry (by name, @username, or phone digits).
    hit = _tg_find(raw)
    if hit is None and _PHONE_RE.match(raw) and any(ch.isdigit() for ch in raw):
        digits = re.sub(r"[^\d+]", "", raw)
        for entry in _tg_registry.values():
            if entry.get("phone") and entry["phone"].lstrip("+") == digits.lstrip("+"):
                hit = entry
                break
    if hit:
        return {
            "chat_id": hit["chat_id"],
            "display": hit.get("name") or hit.get("username") or str(hit["chat_id"]),
        }
    # 2) Contacts.app: resolves a name to a NUMBER, but the bot still needs
    #    a chat_id, so this only helps if that number is in the registry
    #    (checked above) - otherwise fall through to the owner chat.
    if not _PHONE_RE.match(raw) or not any(ch.isdigit() for ch in raw):
        resolved = await _resolve_contact(raw)
        if "error" not in resolved:
            return {
                "chat_id": cfg["owner_chat_id"],
                "display": resolved["display"],
                "note": (
                    f"{resolved['display']} hasn't messaged the bot yet, so "
                    "the message goes to your own Telegram chat instead."
                ),
            }
    # 3) Default: the owner's chat.
    if not cfg["owner_chat_id"]:
        return {
            "error": (
                f"I don't have a Telegram chat for {raw} yet. Have them "
                "message the bot once (so I can learn their chat), or set "
                "TELEGRAM_CHAT_ID in .env as the default destination."
            )
        }
    return {"chat_id": cfg["owner_chat_id"], "display": raw,
            "note": f"No learned chat for {raw} - sent to your own Telegram chat."}


@mcp.tool()
async def call_contact(contact: str = "", service: str = "audio") -> dict:
    """Reach a contact on Telegram with a call-back request.

    Telegram bots cannot initiate calls themselves (calls are peer-to-peer
    and the Bot API has no call method), so this sends the contact a
    Telegram message asking them to call back - the closest a bot can get
    to placing a call.

    Args:
        contact: Who to call: a name like "Mom" or a phone number.
        service: "audio" for a voice call-back, "video" for a video one.
    """
    dest = await _tg_resolve(contact)
    if "error" in dest:
        return dest
    service = "video" if (service or "").strip().lower() == "video" else "audio"
    kind = "video" if service == "video" else "voice"
    cfg = _tg_config()
    text = f"Lava wants to give you a {kind} call - please call me back."
    try:
        resp = await get_http().post(
            f"https://api.telegram.org/bot{cfg['token']}/sendMessage",
            data={"chat_id": dest["chat_id"], "text": text},
        )
        ok, payload = _tg_result(resp)
    except Exception as e:
        return {"error": f"The Telegram call request failed: {e}"}
    if not ok:
        return {
            "error": (
                f"Telegram rejected the message to {dest['display']}"
                + (f" ({payload.get('description')})" if payload.get("description") else "")
                + "."
            )
        }
    return {
        "calling": dest["display"],
        "via": f"Telegram {kind} call request",
        # Speakable note: the LLM relays this almost verbatim, so keep one
        # clause per sentence (a period, not a dash, gives a natural pause).
        "note": (
            f"I asked {dest['display']} to call me back on Telegram. "
            "They will get a message from the bot."
        ),
    }


@mcp.tool()
async def send_message(recipient: str = "", message: str = "") -> dict:
    """Send a Telegram message to a contact.

    Sends the message immediately through the Telegram Bot API - it lands
    in the recipient's Telegram app, not in iMessage. The recipient is
    matched against the bot's learned contacts (people who have messaged
    the bot), then Contacts.app; unknown recipients go to the owner's own
    chat (TELEGRAM_CHAT_ID) with an honest note about it.

    Args:
        recipient: Who to message: a name like "Mom", or a phone number.
        message: The message text to send.
    """
    text = " ".join((message or "").split())
    if not text:
        return {"error": "What should the message say?"}
    dest = await _tg_resolve(recipient)
    if "error" in dest:
        return dest
    cfg = _tg_config()
    try:
        resp = await get_http().post(
            f"https://api.telegram.org/bot{cfg['token']}/sendMessage",
            data={"chat_id": dest["chat_id"], "text": text},
        )
        ok, payload = _tg_result(resp)
    except Exception as e:
        return {"error": f"Sending the Telegram message failed: {e}"}
    if not ok:
        return {
            "error": (
                f"I could not send the message to {dest['display']!r}"
                + (f" ({payload.get('description')})" if payload.get("description") else "")
            )
        }
    note = f"Sent to {dest['display']} on Telegram."
    if dest.get("note"):
        note += " " + dest["note"]
    return {"to": dest["display"], "via": "Telegram", "note": note}




# system


@mcp.tool()
def system_status() -> dict:
    """Report the Raspberry Pi's health: CPU temperature, load, memory, uptime and IP address."""
    out: dict = {}
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            out["cpu_temp_c"] = round(int(f.read()) / 1000, 1)
    except OSError:
        out["cpu_temp_c"] = None
    out["load_1m"] = round(os.getloadavg()[0], 2)
    try:
        with open("/proc/meminfo") as f:
            mem = dict(line.split(":", 1) for line in f)
        total = int(mem["MemTotal"].split()[0]) / 1024
        avail = int(mem["MemAvailable"].split()[0]) / 1024
        out["memory_used_pct"] = round(100 * (1 - avail / total))
        with open("/proc/uptime") as f:
            out["uptime_hours"] = round(float(f.read().split()[0]) / 3600, 1)
    except (OSError, KeyError):
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            out["ip"] = s.getsockname()[0]
    except OSError:
        out["ip"] = None
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
        return {
            "error": "I couldn't reach Spotify just now - it may be a "
            "network blip; try again in a moment."
        }
    if status == 0:
        return {
            "error": "Spotify is not linked. Open "
            f"http://127.0.0.1:{WEB_PORT}/spotify/auth in a "
            "browser to connect it."
        }
    if status == 401:
        return {
            "error": "Spotify authorization expired. Re-link at "
            f"http://127.0.0.1:{WEB_PORT}/spotify/auth."
        }
    if status == 429:
        return {
            "error": "Spotify is rate-limiting this account right now - "
            "wait a few seconds and try again."
        }
    if status == 403:
        msg = (res.get("json", {}).get("error", {}) or {}).get("message", "") or ""
        low = msg.lower()
        if "volume" in low or "restrict" in low:
            # e.g. the web player: Spotify accepts the account but the device
            # has no remotely controllable volume.
            return {
                "error": "This Spotify device does not support remote volume "
                "control - the web player is the usual case. Use the "
                "volume keys on the device itself."
            }
        return {"error": "Spotify Premium is required to control playback."}
    if status == 404:
        return {
            "error": "No active Spotify device found. Open Spotify on "
            "your phone or computer and press play once, then "
            "try again."
        }
    if status >= 400:
        msg = res.get("json", {}).get("error", {}).get("message", "")
        return {"error": f"Spotify said: {msg or status}"}
    return None


def _track_line(t: dict) -> str:
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    return f"{t['name']} — {artists}"


def _is_browser_device(name: str) -> bool:
    """True for web-player/browser devices.

    The free web player pauses whenever its tab loses focus - and the user
    is talking to Lava in a browser tab, so playback there dies seconds in."""
    low = name.lower()
    return "web player" in low or any(
        b in low for b in ("safari", "chrome", "firefox", "edge")
    )


def _on_raspberry_pi() -> bool:
    """True when Lava runs on a Pi (shared profile - see web/core/config)."""
    return DEVICE_PROFILE == "pi"


def _device_hint(name: str, devices: list | None = None) -> str:
    """Voice-friendly hint when playback lands somewhere users overlook.

    `devices` is the full /me/player/devices list when the caller has it:
    it decides whether the remedy is "switch to the other device" (a
    non-browser device exists) or "there is no better device here" - and on
    a Pi the latter means installing a Connect receiver, not opening a
    desktop app a headless Pi has no screen for."""
    if not _is_browser_device(name):
        return ""
    base = (
        "Playing in the Spotify WEB PLAYER browser tab - it pauses "
        "whenever that tab loses focus, so the music stops on its own "
        "as soon as the user switches away. "
    )
    others = [
        d["name"]
        for d in (devices or [])
        if d.get("name") and not _is_browser_device(d["name"])
    ]
    if others:
        return base + (
            f"A non-browser device is available ({', '.join(others)}) - "
            "suggest moving playback there, or opening the Spotify desktop "
            "app; also check the tab's mute button."
        )
    if _on_raspberry_pi():
        return base + (
            "No other device is connected. On this Raspberry Pi the fix is "
            "to install a Spotify Connect receiver (Spotify Soloist) so the "
            "Pi itself becomes a device - see the README's Raspberry Pi "
            "section. Opening a desktop app is not possible on a headless Pi."
        )
    return base + (
        "No other device is connected - suggest opening the Spotify desktop "
        "app on this computer; also check the tab's mute button."
    )


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
    _playback["checked_at"] = time.monotonic()  # claim the slot even on failure
    try:
        res = await spotify.api("GET", "/me/player")
        if res["status"] == 200:
            _playback["is_playing"] = bool(res["json"].get("is_playing"))
        elif res["status"] == 204:
            _playback["is_playing"] = False
    except Exception as e:
        # Keep the last known state; the next poll retries.
        logger.debug(f"playback poll failed, keeping cached state: {e}")
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
    res = await spotify.api(
        "GET",
        "/search",
        require_user=False,
        params={"q": query, "type": kind, "limit": 5},
    )
    err = _spotify_error(res)
    if err:
        return err
    items = (res["json"].get(f"{kind}s", {}) or {}).get("items", [])
    if not items:
        return {"results": [], "note": f"No {kind} matching {query!r}"}
    out = []
    for it in items:
        if not it:  # Spotify occasionally returns null entries
            continue
        row = {"name": it["name"], "uri": it["uri"]}
        if kind == "track" or kind in ("album",):
            row["artist"] = ", ".join(a["name"] for a in it.get("artists", []))
        elif kind == "playlist":
            row["owner"] = it.get("owner", {}).get("name", "")
        out.append(row)
    return {"results": out}


async def _resolve(query: str, kind: str) -> dict:
    """Search and return the best match's uri and display name."""
    res = await spotify.api(
        "GET",
        "/search",
        require_user=False,
        params={"q": query, "type": kind, "limit": 1},
    )
    items = [
        it for it in (res["json"].get(f"{kind}s", {}) or {}).get("items", []) if it
    ]  # Spotify occasionally returns null entries
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
        match = next(
            (
                it
                for it in d.get("json", {}).get("devices", [])
                if device.lower() in it["name"].lower()
            ),
            None,
        )
        if not match:
            return {"error": f"No Spotify device named {device!r}"}
        qs = f"?device_id={match['id']}"
    else:
        # No explicit device: if playback would land in a browser web player
        # and a non-browser device exists, move it there. The free web player
        # pauses whenever its tab loses focus - and the user is talking to
        # Lava in a browser tab - so music there dies seconds in.
        cur = await spotify.api("GET", "/me/player")
        cur_dev = ((cur.get("json") or {}).get("device") or {}).get("name", "")
        if _is_browser_device(cur_dev):
            d = await spotify.api("GET", "/me/player/devices")
            pref = next(
                (
                    it
                    for it in d.get("json", {}).get("devices", [])
                    if not _is_browser_device(it["name"])
                ),
                None,
            )
            if pref:
                qs = f"?device_id={pref['id']}"
    res = await spotify.api("PUT", f"/me/player/play{qs}", json=body)
    if res["status"] == 404 and not device:
        # nothing active: if exactly one device exists, move playback there
        d = await spotify.api("GET", "/me/player/devices")
        devs = d.get("json", {}).get("devices", [])
        if len(devs) == 1:
            res = await spotify.api(
                "PUT", f"/me/player/play?device_id={devs[0]['id']}", json=body
            )
        elif devs:
            # Several devices: prefer a non-browser one (see above); only
            # ask the user when every option is a web player.
            pref = next((it for it in devs if not _is_browser_device(it["name"])), None)
            if pref:
                res = await spotify.api(
                    "PUT", f"/me/player/play?device_id={pref['id']}", json=body
                )
            else:
                names = ", ".join(it["name"] for it in devs)
                return {
                    "error": f"Several Spotify devices are available ({names}) "
                    "- say which one to use."
                }
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
                confirmed = True  # the requested track is on air
                break
    if dev_name:
        out["device"] = dev_name  # so the LLM can say WHERE it plays
        note = _device_hint(dev_name)
        if not confirmed and not note:
            name = fallback_name or "your request"
            note = (
                f"Spotify accepted the command on {dev_name} but has not "
                f"confirmed {name} yet - check that the app on that "
                "device is open, unlocked, and not muted."
            )
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
    body = {"uris": [hit["uri"]]} if kind == "track" else {"context_uri": hit["uri"]}
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
    _set_playing(False)  # the mic gates react to music the instant it stops
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
    # 2xx response. Probe at 0.2 / 0.5 / 0.8 s (cumulative): Spotify
    # registers a skip within ~0.3-0.9 s, so a real skip lands on probe 1 or
    # 2 (0.2-0.5 s instead of the old 0.6-0.9 s), and an empty queue falls
    # through after 0.8 s and 3 read-backs instead of 1.2 s and 4.
    waited = 0.0
    for probe_at in (0.2, 0.5, 0.8):
        await asyncio.sleep(probe_at - waited)
        waited = probe_at
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
        return {
            "restarted": _track_line(cur),
            "note": "No previous track in the queue - restarted the current song.",
        }
    # Try each credited artist in turn (featured artists often have no
    # catalog of their own), then fall back to track-name search. The old
    # queries were malformed: `track:` takes a track NAME, so searching
    # `track:<artist>` returned nothing half the time.
    items = []
    for a in cur.get("artists") or []:
        name = a.get("name", "")
        if not name:
            continue
        res2 = await spotify.api(
            "GET",
            "/search",
            require_user=False,
            params={"q": f'artist:"{name}"', "type": "track", "limit": 10},
        )
        items = [
            it
            for it in ((res2["json"].get("tracks", {}) or {}).get("items", []))
            if it and it["uri"] != cur_uri
        ]
        if items:
            break
    if not items:
        res2 = await spotify.api(
            "GET",
            "/search",
            require_user=False,
            params={"q": f"{cur.get('name', '')}", "type": "track", "limit": 6},
        )
        items = [
            it
            for it in ((res2["json"].get("tracks", {}) or {}).get("items", []))
            if it and it["uri"] != cur_uri
        ]
    if not items:
        return {
            "skipped": direction,
            "note": "No other tracks found by this artist on Spotify - "
            "ask me to play a different song.",
        }
    nxt = items[0]
    play = await spotify.api("PUT", "/me/player/play", json={"uris": [nxt["uri"]]})
    err = _spotify_error(play)
    if err:
        return err
    _set_playing(True)
    out = {
        "playing": _track_line(nxt),
        "note": "Queue was empty, so played the next song by the same artist.",
    }
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
            return (
                "",
                None,
                f"Volume {val} percent is out of range - the maximum is 100.",
            )
        if "%" in raw or "percent" in raw:
            return "", val, ""  # "40 percent" is always absolute
        if re.search(r"\+", raw):
            return "+", val, ""
        if re.search(r"[-]", raw) or down and not up:
            return "-", val, ""
        if up and not down:
            return "+", val, ""
        return "", val, ""
    if up and not down:
        return "+", VOLUME_STEP, ""  # "turn it up" — no number given
    if down and not up:
        return "-", VOLUME_STEP, ""
    if up and down:
        return "", None, "I got both 'up' and 'down' in there - say one direction."
    return "", None, ""  # fall through: report current volume


async def _spotify_current_volume() -> dict:
    """GET /me/player -> {"error"} | {"volume_pct": int|None}."""
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {
            "volume_pct": None,
            "note": "No active Spotify device, so there is no volume to read.",
        }
    err = _spotify_error(res)
    if err:
        return err
    pct = (res["json"].get("device") or {}).get("volume_percent")
    return {
        "volume_pct": pct,
        "note": "That is the device's own volume, as reported by Spotify.",
    }


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
        return {
            "error": err_msg
            or f"I could not understand the volume '{level}'. Say a "
            "percentage like '40', an adjustment like 'up' or 'down 5', "
            "'max', or 'mute'."
        }
    res = await spotify.api("GET", "/me/player")
    if res["status"] == 204:
        return {
            "error": "No active Spotify device to change the volume on. "
            "Open Spotify somewhere and press play once, then "
            "try again."
        }
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
    put = await spotify.api(
        "PUT", "/me/player/volume", params={"volume_percent": target}
    )
    err = _spotify_error(put)
    if err:
        return err
    verb = "raised" if sign == "+" else ("lowered" if sign == "-" else "set")
    out = {
        "volume_pct": target,
        "action": verb,
        "note": f"Volume {verb} to {target} percent.",
    }
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
        return {
            "shuffle": "unknown",
            "note": "No active Spotify device, so shuffle state is unknown.",
        }
    err = _spotify_error(res)
    if err:
        return err
    if not state:
        on = bool(res["json"].get("shuffle_state", False))
        return {
            "shuffle": "on" if on else "off",
            "note": f"Shuffle is {'on' if on else 'off'}.",
        }
    put = await spotify.api("PUT", "/me/player/shuffle", params={"state": state})
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
        return {
            "repeat": "unknown",
            "note": "No active Spotify device, so repeat state is unknown.",
        }
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
    label = {
        "off": "off",
        "track": "on for the current song",
        "context": "on for the album or playlist",
    }[mode]
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
    out = {
        "playing": _track_line(item),
        "progress_ms": res["json"].get("progress_ms"),
        "duration_ms": item.get("duration_ms"),
        "paused": not res["json"].get("is_playing", False),
    }
    dev_name = (res["json"].get("device") or {}).get("name", "")
    if dev_name:
        out["device"] = dev_name  # so the LLM can say WHERE it plays
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
    res = await spotify.api("POST", "/me/player/queue", params={"uri": hit["uri"]})
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
    devs = [
        {"name": it["name"], "type": it["type"], "active": it.get("is_active", False)}
        for it in res["json"].get("devices", [])
    ]
    return {"devices": devs}


if __name__ == "__main__":
    setup_logging("tools")
    logger.info(f"Tools MCP server on http://{HOST}:{PORT}/mcp")
    mcp.run("streamable-http", host=HOST, port=PORT)
