"""Unit checks for the get_temperature tool + its notification banners.

No network: the HA REST layer is stubbed. Run:  .venv/bin/python ha/test_unit.py
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mcp_servers.tools_server as ts  # noqa: E402
from web.core.notifications import describe  # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))


FAKE_STATES = [
    {
        "entity_id": "climate.living_room",
        "state": "heat",
        "attributes": {
            "friendly_name": "Living Room Thermostat",
            "current_temperature": 23.0,
            "temperature": 22.0,
            "temperature_unit": "°C",
        },
    },
    {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {"friendly_name": "Living Room Light", "brightness": 76},
    },
    {
        "entity_id": "climate.kitchen",
        "state": "off",
        "attributes": {"friendly_name": "Kitchen Thermostat", "current_temperature": 24.5, "temperature": 21.0},
    },
]


async def fake_hass(method, path, **kw):
    assert method == "GET" and path == "/api/states", f"unexpected call {method} {path}"
    return 200, FAKE_STATES


async def main():
    ts.HASS_URL = "http://fake"
    ts.HASS_TOKEN = "fake"
    ts._hass = fake_hass
    ts._states_cache = None

    print("[get_temperature]")
    res = await ts.get_temperature(room="living room")
    print("   ", json.dumps(res))
    check("current from attributes (not state)", res.get("current") == 23, json.dumps(res))
    check("target included", res.get("target") == 22, json.dumps(res))
    check("mode passthrough", res.get("mode") == "heat", json.dumps(res))
    check("no error", "error" not in res)

    res = await ts.get_temperature()  # empty room = whole home
    print("   ", json.dumps(res))
    check("whole-home average", res.get("current") == 23.8 and "rooms" in res, json.dumps(res))

    res = await ts.get_temperature(room="kitchen")
    print("   ", json.dumps(res))
    check("room match works", res.get("current") == 24.5, json.dumps(res))
    check("kitchen target", res.get("target") == 21, json.dumps(res))

    res = await ts.get_temperature(room="nonexistent")
    print("   ", json.dumps(res))
    check("no match -> honest error", "error" in res and "No thermostat" in res["error"])

    # climate without current_temperature attribute
    async def fake_hass_weird(method, path, **kw):
        return 200, [
            {"entity_id": "climate.weird", "state": "off", "attributes": {"friendly_name": "Weird"}}
        ]

    ts._hass = fake_hass_weird
    res = await ts.get_temperature()
    print("   ", json.dumps(res))
    check("missing reading -> honest error", "error" in res and "no temperature" in res["error"])

    print("\n[notifications: describe()]")
    cases = [
        (
            "get_temperature",
            {},
            {"thermostat": "Living Room Thermostat", "current": 23, "unit": "°C", "target": 22, "mode": "heat"},
            ("thermostat", "Temperature", "Living Room Thermostat · 23 °C · set to 22"),
        ),
        (
            "get_temperature",
            {},
            {"thermostat": "Living Room Thermostat", "current": 23.5, "unit": "°C", "target": 23.5},
            ("thermostat", "Temperature", "Living Room Thermostat · 23.5 °C"),  # target==current -> omitted
        ),
        (
            "get_temperature",
            {},
            {"error": "No thermostat matching 'this home' in Home Assistant."},
            ("thermostat", "Temperature", None),  # err flag set
        ),
        (
            "set_thermostat",
            {"temperature": 22},
            {"thermostat": "Living Room Thermostat", "temperature": 22, "unit": "C"},
            ("thermostat", "Thermostat", "Living Room Thermostat · set to 22 °C"),
        ),
        (
            "set_lights",
            {"room": "living room", "on": True, "brightness_pct": 30},
            {"lights": "Living Room Light", "on": True, "brightness_pct": 30},
            ("dim", "Dim / color", "Living Room Light · 30%"),
        ),
        (
            "set_lights",
            {"room": "living room", "on": True},
            {"lights": "Living Room Light", "on": True},
            ("lights", "Lights", "Living Room Light · on"),
        ),
    ]
    for name, args, res, (want_cat, want_title, want_sub) in cases:
        d = describe(name, args, json.dumps(res))
        got = (d["cat"], d["title"], d["sub"])
        label = f"{name} -> {got[1]} / {got[2]}"
        if want_sub is None:
            check(label, d["cat"] == want_cat and d["err"] is not None, json.dumps(d))
        else:
            check(label, got == (want_cat, want_title, want_sub), json.dumps(d))

    print(f"\n{'=' * 52}\nRESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed:", *FAILED, sep="\n  - ")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
