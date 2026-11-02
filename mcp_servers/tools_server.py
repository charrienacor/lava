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

Run:  python -m mcp_servers.tools_server          (port 8003 by default)
"""

import asyncio
import json
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from loguru import logger
from mcp.server.mcpserver import MCPServer

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
        geo_resp = await http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en"},
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()
        if not geo.get("results"):
            return {"error": f"Could not find a place called {location}"}
        place = geo["results"][0]
        wx_resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
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
        "place": f"{place['name']}, {place.get('country', '')}".strip(", "),
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

    def _search():
        return DDGS().text(query, max_results=max(1, min(max_results, 8)))

    try:
        hits = await asyncio.to_thread(_search)
    except Exception as e:
        return {"error": f"search failed: {e}"}
    return {
        "query": query,
        "results": [{"title": h.get("title"), "snippet": h.get("body"), "url": h.get("href")} for h in hits],
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
        if label_or_id in (k, v["label"]):
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


if __name__ == "__main__":
    setup_logging("tools")
    logger.info(f"Tools MCP server on http://{HOST}:{PORT}/mcp")
    mcp.run("streamable-http", host=HOST, port=PORT)
