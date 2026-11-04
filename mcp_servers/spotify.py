"""Shared Spotify Web API client for the Lava tools server and web bridge.

Auth model:
  - The account owner links Lava once by visiting /spotify/auth on the bridge;
    Spotify redirects to /callback, the code is exchanged, and the refresh
    token is stored in logs/spotify_token.json.
  - With a refresh token, all tools run with the user's account.
  - Without one, catalog search still works via client credentials, but
    playback tools return a "not linked" error.
"""

import asyncio
import base64
import json
import os
import time
from urllib.parse import urlencode

import httpx
from loguru import logger

from mcp_servers.common import LOGS_DIR

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = (
    "user-read-playback-state user-modify-playback-state "
    "user-read-currently-playing playlist-read-private "
    "user-library-read"
)
TOKEN_FILE = LOGS_DIR / "spotify_token.json"

_state = {"access": None, "expires": 0.0}
# In-memory copy of the stored refresh token: linked() runs on every API call
# and on the bridge's 1 Hz playback poll, and re-reading the token file from
# disk each time is pointless I/O.
_refresh_cache = {"token": None}
# Why the last token fetch failed ("unlinked" | "network"), so callers can
# say "Spotify isn't linked" and "can't reach Spotify" apart.
_token_fail = {"kind": ""}
# One pooled client for every Spotify call (accounts + api). A fresh client
# per call paid a full TLS handshake (~100-300 ms) on EVERY request; a play
# command makes 4-8 requests, so pooling alone cuts music latency by a second
# or more. Same lesson the weather tool already learned.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=15)
    return _client


async def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _basic() -> str:
    pair = (
        f"{os.getenv('SPOTIFY_CLIENT_ID', '')}:{os.getenv('SPOTIFY_CLIENT_SECRET', '')}"
    )
    return "Basic " + base64.b64encode(pair.encode()).decode()


def configured() -> bool:
    return bool(os.getenv("SPOTIFY_CLIENT_ID")) and bool(
        os.getenv("SPOTIFY_CLIENT_SECRET")
    )


def load_refresh() -> str:
    if _refresh_cache["token"] is not None:
        return _refresh_cache["token"]
    try:
        token = json.loads(TOKEN_FILE.read_text()).get("refresh_token", "")
    except Exception:
        token = ""
    _refresh_cache["token"] = token
    return token


def save_refresh(refresh_token: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"refresh_token": refresh_token}))
    _refresh_cache["token"] = refresh_token


def linked() -> bool:
    return bool(load_refresh())


def authorize_url(redirect_uri: str, state: str | None = None) -> str:
    params = {
        "client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
    }
    if state:
        params["state"] = state
    return "https://accounts.spotify.com/authorize?" + urlencode(params)


async def exchange_code(code: str, redirect_uri: str) -> bool:
    """One-time: exchange the authorization code and store the refresh token."""
    try:
        r = await get_client().post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Authorization": _basic()},
        )
        if r.status_code != 200:
            return False
        j = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"spotify token exchange failed: {e}")
        return False
    if j.get("refresh_token"):
        save_refresh(j["refresh_token"])
    _state["access"] = j["access_token"]
    _state["expires"] = time.time() + j.get("expires_in", 3600) - 60
    _token_fail["kind"] = ""
    return True


async def access_token(require_user: bool) -> str | None:
    """Bearer token for API calls; user token if linked, else app token.

    Returns None when no token exists (sets _token_fail["kind"] to "unlinked")
    or when the network is down (kind "network")."""
    _token_fail["kind"] = ""
    if _state["access"] and time.time() < _state["expires"]:
        return _state["access"]
    refresh = load_refresh()
    if refresh:
        data = {"grant_type": "refresh_token", "refresh_token": refresh}
    elif not require_user:
        data = {"grant_type": "client_credentials"}
    else:
        _token_fail["kind"] = "unlinked"
        return None
    try:
        r = await get_client().post(
            TOKEN_URL, data=data, headers={"Authorization": _basic()}
        )
        if r.status_code != 200:
            _token_fail["kind"] = "unlinked"
            return None
        j = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"spotify token refresh failed: {e}")
        _token_fail["kind"] = "network"
        return None
    _state["access"] = j.get("access_token")
    _state["expires"] = time.time() + j.get("expires_in", 3600) - 60
    if j.get("refresh_token"):  # Spotify may rotate it
        save_refresh(j["refresh_token"])
    return _state["access"]


async def api(method: str, path: str, require_user: bool = True, **kwargs) -> dict:
    """Call the Web API; refresh the token and retry once on 401.

    Never raises: transport/token failures come back as
    {"status": -1} (network) or {"status": 0} (not linked). That matters
    because an exception raised inside an MCP tool handler tears the whole
    server down (anyio's TaskGroup kills the session) - which is how every
    "stop the music" once became "tools MCP server not connected".
    """
    for attempt in range(2):
        token = await access_token(require_user)
        if not token:
            return {"status": -1 if _token_fail["kind"] == "network" else 0, "json": {}}
        try:
            r = await get_client().request(
                method,
                API + path,
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
        except httpx.HTTPError as e:
            # One quick retry in case the failure was transient; a play is
            # worth the 200 ms it costs the rare false alarm.
            if attempt == 0:
                _state["access"] = None
                await asyncio.sleep(0.2)
                continue
            logger.warning(f"spotify {method} {path} failed: {e}")
            return {"status": -1, "json": {}}
        if r.status_code == 401 and attempt == 0:
            _state["access"] = None  # force a refresh, then retry
            continue
        try:
            return {"status": r.status_code, "json": r.json()}
        except ValueError:
            return {"status": r.status_code, "json": {}}
    return {"status": -1, "json": {}}
