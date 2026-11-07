#!/usr/bin/env bash
#
# start_ha.sh - start the virtual Home Assistant hub that backs Lava's
# smart-home tools (lights + thermostat). No physical hardware involved:
# the entities in ha/configuration.yaml are virtual, and the HA dashboard
# (http://127.0.0.1:8123) shows them updating live.
#
# First run seeds the long-lived access token that Lava's tools use
# (ha/seed_token.py, written to .env as HASS_TOKEN).
#
# Usage:  ./ha/start_ha.sh          (from the repo root or anywhere)
#
# On the Raspberry Pi prefer the Docker setup (deploy/home-assistant/)
# instead - this script is for a dev machine or a Pi without Docker.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

PY="$SCRIPT_DIR/venv/bin/python"
CONFIG="$SCRIPT_DIR/config"      # HA's config dir (runtime; git-ignored)
DATA="$SCRIPT_DIR/data"
SRC_CONFIG="$SCRIPT_DIR/configuration.yaml"   # tracked source of truth
PORT=8123

port_open() {
    python3 - "$1" <<'PYEOF' >/dev/null 2>&1
import socket, sys
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except Exception:
    sys.exit(1)
PYEOF
}

# If the hub is already up (e.g. the Docker setup on a Pi, see
# deploy/home-assistant/), no local venv is needed at all - skip straight
# to making sure .env points at it.
if port_open "$PORT"; then
    echo "  Home Assistant (:$PORT) - already running, skipping"
else
    if [ ! -x "$PY" ]; then
        echo "ERROR: Home Assistant venv not found at $PY" >&2
        echo "       create it (needs Python 3.13):" >&2
        echo "         python3.13 -m venv $SCRIPT_DIR/venv" >&2
        echo "         $SCRIPT_DIR/venv/bin/pip install homeassistant websockets" >&2
        exit 1
    fi
    # seed_token.py mints the long-lived token over a WebSocket; make sure
    # the client lib is present (HA itself doesn't need it).
    "$PY" -c "import websockets" 2>/dev/null || "$PY" -m pip install -q websockets
    mkdir -p "$DATA" "$CONFIG"
    # HA's config dir must contain configuration.yaml; copy the tracked
    # source in so edits to ha/configuration.yaml take effect on restart.
    cp "$SRC_CONFIG" "$CONFIG/configuration.yaml"
    echo "  Home Assistant (:$PORT) - starting (first boot takes ~20-60 s)..."
    nohup "$PY" -m homeassistant -c "$CONFIG" \
        >"$ROOT/logs/ha.log" 2>&1 &
    echo $! > "$ROOT/logs/ha.pid"
fi

# Wait for the HTTP API to accept connections.
deadline=$(( SECONDS + 120 ))
while [ $SECONDS -lt $deadline ]; do
    if port_open "$PORT"; then
        echo "  OK - Home Assistant is up at http://127.0.0.1:$PORT"
        break
    fi
    sleep 1
done
if ! port_open "$PORT"; then
    echo "  FAIL - Home Assistant did not come up in 120 s - check logs/ha.log" >&2
    exit 1
fi

# Seed the long-lived token (idempotent: reuses the existing one in .env).
if ! grep -q '^HASS_TOKEN=' "$ROOT/.env" 2>/dev/null; then
    echo "  seeding long-lived access token..."
    if "$PY" "$SCRIPT_DIR/seed_token.py" --port "$PORT"; then
        echo "  OK - HASS_TOKEN written to .env"
    else
        echo "  WARN - token seeding failed; mint one manually:" >&2
        echo "         HA UI -> Profile -> Security -> Long-Lived Access Tokens" >&2
        echo "         then add HASS_TOKEN=<token> to .env" >&2
    fi
else
    echo "  HASS_TOKEN already in .env - reusing"
fi

# Point .env at this instance if not set already.
if ! grep -q '^HASS_URL=' "$ROOT/.env" 2>/dev/null; then
    printf '\n# smart home (virtual Home Assistant hub, see ha/)\nHASS_URL=http://127.0.0.1:%s\n' "$PORT" >> "$ROOT/.env"
    echo "  HASS_URL written to .env"
fi
