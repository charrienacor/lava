"""Seed the Home Assistant long-lived access token for Lava.

Runs once (idempotent - start_ha.sh skips it when .env already has
HASS_TOKEN). Targets the HA 2026.x API:

  1. Complete first-boot onboarding if pending:
       GET  /api/onboarding                      -> step statuses
       POST /api/onboarding/users   (no auth)    -> creates the OWNER/admin
                                                    user, returns auth_code
       POST /auth/token  (form)                  -> exchange code for an
                                                    access token
       POST /api/onboarding/core_config (auth)
       POST /api/onboarding/analytics   (auth)
       POST /api/onboarding/integration (auth)
  2. Mint the LONG-LIVED token over the WebSocket (the only way since
     HA 2024.x; the REST "long-lived" endpoint is gone):
       WS   /api/websocket  (Authorization: Bearer <access_token>)
       ->   {"type": "auth/long_lived_access_token", "lifespan": 3650,
             "client_name": "Lava voice agent"}
  3. Verify the token against /api/ and write HASS_URL/HASS_TOKEN to .env.

Usage:  venv/bin/python ha/seed_token.py [--port 8123]
Onboarding password: 'lava' (local-only demo instance; the live instance's
password was later changed to 'lava090201' - this constant only matters on a
fresh first boot; change it in the
HA UI afterwards if you care).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CLIENT_ID = "https://home-assistant.io/lava"  # same domain as redirect -> no IndieAuth fetch
REDIRECT_URI = "https://home-assistant.io/lava/callback"
ONBOARD_PW = "lava"

BASE = "http://127.0.0.1:{port}"


def http(method: str, path: str, body=None, token: str | None = None, form: bool = False):
    url = BASE + path
    if form:
        data = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in body.items()).encode()
        ctype = "application/x-www-form-urlencoded"
    else:
        data = json.dumps(body).encode() if body is not None else None
        ctype = "application/json"
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", ctype)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:300]}


def ws_long_lived_token(access_token: str, lifespan_days: int = 3650) -> str | None:
    """Mint a long-lived access token over the HA WebSocket."""
    import asyncio

    import websockets

    async def _mint() -> str | None:
        ws_base = BASE.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_base}/api/websocket"
        async with websockets.connect(uri) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") == "auth_required":
                await ws.send(json.dumps({"type": "auth", "access_token": access_token}))
                hello = json.loads(await ws.recv())
            if hello.get("type") != "auth_ok":
                print(f"  WS auth failed: {hello}", file=sys.stderr)
                return None
            await ws.send(json.dumps({
                "id": 1,
                "type": "auth/long_lived_access_token",
                "lifespan": lifespan_days,
                "client_name": "Lava voice agent",
            }))
            deadline = time.time() + 20
            while time.time() < deadline:
                msg = json.loads(await ws.recv())
                if msg.get("id") == 1:
                    if msg.get("type") == "result":
                        return msg.get("result")
                    print(f"  WS token command failed: {msg}", file=sys.stderr)
                    return None
            return None

    return asyncio.run(_mint())


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    BASE = BASE.format(port=args.port)

    # 1. onboarding ----------------------------------------------------------
    status, steps = http("GET", "/api/onboarding")
    if status != 200:
        print(f"  onboarding state unreadable (HTTP {status}): {steps}", file=sys.stderr)
        return 1
    done = {s["step"] for s in steps if s.get("done")}
    pending = [s["step"] for s in steps if not s.get("done")]
    print(f"  onboarding: done={sorted(done) or '-'} pending={pending or '-'}")

    access_token = None
    if "user" not in done:
        status, out = http("POST", "/api/onboarding/users", {
            "name": "Lava",
            "username": "lava",
            "password": ONBOARD_PW,
            "client_id": CLIENT_ID,
            "language": "en",
        })
        if status not in (200, 201):
            print(f"  onboarding/users failed (HTTP {status}): {out}", file=sys.stderr)
            return 1
        print("  admin user 'lava' created (password 'lava')")
        status, out = http("POST", "/auth/token", {
            "grant_type": "authorization_code",
            "code": out.get("auth_code"),
            "client_id": CLIENT_ID,
        }, form=True)
        if status not in (200, 201) or not out.get("access_token"):
            print(f"  auth/token exchange failed (HTTP {status}): {out}", file=sys.stderr)
            return 1
        access_token = out["access_token"]

    if not access_token:
        # The user already exists (created in an earlier partial run, or
        # onboarding finished previously): log in through the password flow.
        status, flow = http("POST", "/auth/login_flow", {
            "client_id": CLIENT_ID,
            "handler": ["homeassistant", None],
            "redirect_uri": REDIRECT_URI,
        })
        if status not in (200, 201):
            print(f"  login_flow start failed (HTTP {status}): {flow}", file=sys.stderr)
            return 1
        flow_id = flow.get("flow_id") or flow.get("id")
        status, out = http("POST", f"/auth/login_flow/{flow_id}", {
            "client_id": CLIENT_ID,
            "username": "lava",
            "password": ONBOARD_PW,
        })
        if status not in (200, 201):
            print(f"  login_flow password step failed (HTTP {status}): {out}", file=sys.stderr)
            return 1
        # The completed flow returns an authorization CODE in "result" (or an
        # access_token directly on some versions) - exchange the code for a
        # real token.
        code = out.get("result") or out.get("auth_code")
        if out.get("access_token"):
            access_token = out["access_token"]
        elif code:
            status, out = http("POST", "/auth/token", {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": CLIENT_ID,
            }, form=True)
            if status not in (200, 201) or not out.get("access_token"):
                print(f"  auth/token exchange failed (HTTP {status}): {out}", file=sys.stderr)
                return 1
            access_token = out["access_token"]
        else:
            print(f"  no token or code in login result: {out}", file=sys.stderr)
            return 1

    # remaining onboarding steps (all authenticated)
    for step, body in (
        ("core_config", {}),
        ("analytics", {}),
        ("integration", {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI}),
    ):
        if step not in done:
            status, out = http("POST", f"/api/onboarding/{step}", body, token=access_token)
            if status not in (200, 201):
                print(f"  onboarding/{step} failed (HTTP {status}): {out}", file=sys.stderr)
                return 1
    print("  onboarding complete")

    # 2. mint the long-lived token over the WebSocket -------------------------
    token = ws_long_lived_token(access_token)
    if not token:
        print("  long-lived token mint failed", file=sys.stderr)
        return 1

    # 3. verify + persist ------------------------------------------------------
    status, _ = http("GET", "/api/", token=token)
    if status != 200:
        print(f"  token rejected by /api/ (HTTP {status})", file=sys.stderr)
        return 1

    env_path = __file__.rsplit("/ha/", 1)[0] + "/.env"
    with open(env_path) as f:
        env = f.read()
    if "HASS_URL=" not in env:
        env += f"\n# smart home (long-lived token minted by ha/seed_token.py)\nHASS_URL={BASE}\n"
    if "HASS_TOKEN=" not in env:
        env += f"HASS_TOKEN={token}\n"
    with open(env_path, "w") as f:
        f.write(env)
    print(f"  long-lived token minted and verified ({token[:8]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
