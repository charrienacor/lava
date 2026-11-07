#!/usr/bin/env bash
#
# start_all.sh — bring up the Lava MCP servers:
#
#   asr MCP   :8001   (Groq Whisper)
#   tts MCP   :8002   (Piper)
#   tools MCP :8003
#   HA hub    :8123   (virtual Home Assistant - lights + thermostat)
#
# The web bridge (web/server.py, :8080) is deliberately NOT started here —
# run it separately, after the MCP servers are up:
#   .venv/bin/python -m web.server        # then open http://127.0.0.1:8080
# (the bridge auto-reconnects to a dead MCP server every 2 s, but the
# servers must be up for its first connection, so start them first)
#
# Usage:  ./start_all.sh
#
# Each service is detached with nohup, so it keeps running after this
# terminal closes. PIDs are recorded in logs/<name>.pid; stop_all.sh
# tears them down. Logs land in logs/<name>.log (rotated 10 MB,
# kept 3 days).

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"

if [ ! -x "$PY" ]; then
    echo "ERROR: virtualenv not found at $PY" >&2
    echo "       create it at the project root ($ROOT):" >&2
    echo "       python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

mkdir -p "$LOGS"

# --- helpers ---------------------------------------------------------------

# Readiness = a plain TCP connect (the same nc check stop_all.sh uses).
# Deliberately NOT an HTTP POST /mcp: each such probe makes the server
# fabricate and terminate an MCP transport ("Created new transport ...
# 400 Bad Request / Terminating session" churn — the same behavior the
# bridge's supervisor comment in web/core/mcp_client.py warns about).
# "Port up but app not ready" is fine: the bridge's MCPConn
# auto-reconnects every 2 s until the handshake succeeds.
#
# Port check is a plain Python socket connect, NOT `nc -z -G`: the -G flag is
# BSD/macOS netcat only, and Raspberry Pi OS ships netcat-traditional, which
# rejects it (so the check would always fail and the servers would be reported
# as "not coming up" even when they are). Python is guaranteed present (it is
# the runtime), so this is portable across both hosts.
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

wait_mcp_ready() {  # <port> <name> <seconds>
    local deadline=$(( SECONDS + $3 ))
    while [ $SECONDS -lt $deadline ]; do
        if port_open "$1"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

start_one() {  # <name> <port> <module>
    if port_open "$2"; then
        echo "  $1 (:$2) — already running, skipping"
        return 0
    fi
    nohup "$PY" -m "$3" >/dev/null 2>&1 &
    echo $! > "$LOGS/$1.pid"
    echo "  $1 (:$2) — starting (pid $(cat "$LOGS/$1.pid"))"
}

# --- start -----------------------------------------------------------------

echo "Starting Lava MCP servers…"
start_one asr   8001 mcp_servers.asr_server
start_one tts   8002 mcp_servers.tts_server
start_one tools 8003 mcp_servers.tools_server

echo "  waiting for MCP server ports to accept connections…"
ok=1
for pair in "8001 asr" "8002 tts" "8003 tools"; do
    set -- $pair
    if wait_mcp_ready "$1" "$2" 30; then
        echo "  ✓ $2 MCP listening (:$1)"
    else
        echo "  ✗ $2 MCP (:$1) not accepting connections after 30 s — check logs/$2.log"
        ok=0
    fi
done

echo
if [ "$ok" -eq 1 ]; then
    echo "All MCP servers are up."
else
    echo "Some MCP servers did not come up — see logs/ above." >&2
    exit 1
fi

# --- virtual Home Assistant hub (:8123) ------------------------------------
# Backs the smart-home tools (set_lights / set_thermostat / get_temperature).
# start_ha.sh handles its own readiness wait (120 s) and token seeding, and
# no-ops if the hub is already running.
echo
"$ROOT/ha/start_ha.sh"
