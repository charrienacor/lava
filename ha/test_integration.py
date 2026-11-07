#!/usr/bin/env python3
"""End-to-end test: Lava's smart-home tools against the live Home Assistant.

Drives the SAME pipeline a spoken command uses - the bridge's /api/run
endpoint (exec_tool -> tools MCP -> HA REST API) - and then verifies, via
HA's own /api/states, that the hub's state really changed. No faking: every
step is asserted, and failures are printed with the raw payloads.

Prereqs: Home Assistant up (ha/start_ha.sh), tools MCP + bridge up
(deploy/start_all.sh + web.server), HASS_URL/HASS_TOKEN in .env.

Usage:  .venv/bin/python ha/test_integration.py
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import os  # noqa: E402

HASS_URL = os.getenv("HASS_URL", "").rstrip("/")
HASS_TOKEN = os.getenv("HASS_TOKEN", "")

BRIDGE = "http://127.0.0.1:8080"
HDRS = {"Authorization": f"Bearer {HASS_TOKEN}"}

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    return cond


async def hass_states(client: httpx.Client) -> dict:
    r = await client.get(f"{HASS_URL}/api/states", headers=HDRS)
    r.raise_for_status()
    return {s["entity_id"]: s for s in r.json()}


async def run_tool(client: httpx.Client, tool: str, args: dict) -> dict:
    """One tool call through the bridge's /api/run (the voice pipeline)."""
    r = await client.post(f"{BRIDGE}/api/run", json={"tool": tool, "args": args})
    body = r.json()
    if not body.get("ok"):
        return {"error": f"bridge rejected: {body.get('error')}"}
    # /api/run returns the tool result under "result"
    res = body.get("result")
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except ValueError:
            pass
    return res if isinstance(res, dict) else {"raw": body}


async def main() -> int:
    if not HASS_URL or not HASS_TOKEN:
        print("ABORT: HASS_URL/HASS_TOKEN missing from .env")
        return 2

    async with httpx.AsyncClient(timeout=30) as client:
        # 0. prerequisites
        print("\n[0] prerequisites")
        r = await client.get(f"{HASS_URL}/api/", headers=HDRS)
        check("Home Assistant reachable", r.status_code == 200, f"HTTP {r.status_code}")
        if r.status_code != 200:
            return 1
        r = await client.get(f"{BRIDGE}/api/status")
        check("Lava bridge reachable", r.status_code == 200, f"HTTP {r.status_code}")
        if r.status_code != 200:
            return 1
        status = r.json()
        tools = status.get("tools") or []
        check(
            "smart-home tools loaded",
            {"set_lights", "set_thermostat", "get_temperature"} <= set(tools),
            f"tools={tools}",
        )

        states = await hass_states(client)
        # Discover the virtual entities (don't hardcode ids - the climate
        # entity id derives from its name, e.g. climate.living_room_thermostat).
        light = next((eid for eid in states if eid.startswith("light.living_room")), None)
        climate = next((eid for eid in states if eid.startswith("climate.living_room")), None)
        check("virtual light exists in HA", light is not None, ", ".join(sorted(states))[:120])
        check("virtual thermostat exists in HA", climate is not None)
        if not light or not climate:
            return 1

        # 1. "Turn on the living room lights."
        print('\n[1] "Turn on the living room lights."')
        res = await run_tool(client, "set_lights", {"room": "living room", "on": True})
        check("tool call ok", "error" not in res, json.dumps(res)[:160])
        await asyncio.sleep(0.5)
        states = await hass_states(client)
        check("HA light state == on", states[light]["state"] == "on", states[light]["state"])

        # 2. "Dim the lights to 30%."
        print('\n[2] "Dim the lights to 30%. "')
        res = await run_tool(client, "set_lights", {"room": "living room", "on": True, "brightness_pct": 30})
        check("tool call ok", "error" not in res, json.dumps(res)[:160])
        await asyncio.sleep(0.5)
        states = await hass_states(client)
        bright = states[light].get("attributes", {}).get("brightness")
        # 30% of 255 = 76.5 -> HA rounds to 76 or 77
        check("HA brightness ~30%", bright is not None and 74 <= bright <= 78, f"brightness={bright}")

        # 3. "Set the temperature to 22 degrees."
        print('\n[3] "Set the temperature to 22 degrees."')
        res = await run_tool(client, "set_thermostat", {"temperature": 22})
        check("tool call ok", "error" not in res, json.dumps(res)[:160])
        await asyncio.sleep(0.5)
        states = await hass_states(client)
        tgt = states[climate].get("attributes", {}).get("temperature")
        check("HA thermostat target == 22", tgt == 22, f"target={tgt}")

        # 4. "What's the current temperature?"
        print('\n[4] "What\'s the current temperature?"')
        res = await run_tool(client, "get_temperature", {})
        check("tool call ok", "error" not in res, json.dumps(res)[:160])
        check("reports a current temperature", isinstance(res.get("current"), (int, float)), json.dumps(res)[:160])
        check("reports the target too", res.get("target") == 22, json.dumps(res)[:160])

        # 5. turn off (cleanup + off-path)
        print('\n[5] "Turn off the lights."')
        res = await run_tool(client, "set_lights", {"room": "living room", "on": False})
        check("tool call ok", "error" not in res, json.dumps(res)[:160])
        await asyncio.sleep(0.5)
        states = await hass_states(client)
        check("HA light state == off", states[light]["state"] == "off", states[light]["state"])

    print(f"\n{'=' * 52}\nRESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed:", *FAILED, sep="\n  - ")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
