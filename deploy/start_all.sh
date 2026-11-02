#!/usr/bin/env bash
#
# start_all.sh — bring up the Lava MCP servers:
#
#   asr MCP   :8001   (Groq Whisper)
#   tts MCP   :8002   (Piper)
#   tools MCP :8003
#
# The web bridge (web/server.py, :8080) is deliberately NOT started here —
# run it separately, after the MCP servers are up:
#   .venv/bin/python -m web.server        # then open http://127.0.0.1:8080
# (the bridge connects to the MCP servers once at startup and does not
# retry, so it must start AFTER them)
#
# Usage:  ./start_all.sh
#
# Each service is detached with nohup, so it keeps running after this
# terminal closes. PIDs are recorded in logs/<name>.pid; stop_all.sh
# tears them down. Logs land in logs/<name>.log (rotated 10 MB,
# kept 3 days).
#
set -u

# This script lives in deploy/, but the project root (where .venv/ and
# logs/ actually live) is the PARENT directory — resolve it, don't assume CWD.
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

port_open() {
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/"
}

wait_mcp_ready() {  # <port> <name> <seconds>
    local deadline=$(( SECONDS + $3 ))
    while [ $SECONDS -lt $deadline ]; do
        if curl -s -o /dev/null --max-time 2 -X POST "http://127.0.0.1:$1/mcp" \
                -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream"; then
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

echo "  waiting for MCP servers to accept connections…"
ok=1
for pair in "8001 asr" "8002 tts" "8003 tools"; do
    set -- $pair
    if wait_mcp_ready "$1" "$2" 30; then
        echo "  ✓ $2 MCP ready (:$1)"
    else
        echo "  ✗ $2 MCP (:$1) not ready after 30 s — check logs/$2.log"
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
