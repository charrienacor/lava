#!/usr/bin/env bash
#
# stop_all.sh — stop the Lava MCP servers (asr, tts, tools).
#
# The web bridge (web/server.py, :8080) is not managed here — stop it
# yourself (Ctrl-C in its terminal, or:  pkill -f "web\.server").
#
# Usage:  ./stop_all.sh
#
# Prefers the PID files written by start_all.sh (logs/<name>.pid); falls
# back to a pgrep on the module names, so it also cleans up servers that
# were started by hand. Sends SIGTERM first, escalates to SIGKILL after
# 5 s, and verifies the ports are actually free.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
LOGS="$ROOT/logs"

kill_pid() {  # <pid> <label>
    local pid="$1" label="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  $label — pid $pid not running (stale), removing pid file"
        rm -f "$LOGS/$label.pid"
        return 0
    fi
    kill "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "  $label — pid $pid ignored SIGTERM, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
        sleep 0.5
    fi
    echo "  $label — stopped (pid $pid)"
    rm -f "$LOGS/$label.pid"
}

echo "Stopping Lava MCP servers…"

# --- 1) PID files from start_all.sh ----------------------------------------

for name in tools tts asr; do
    pidfile="$LOGS/$name.pid"
    if [ -f "$pidfile" ]; then
        kill_pid "$(cat "$pidfile")" "$name"
    fi
done

# --- 2) Fallback: anything started by hand ----------------------------------

for mod in "mcp_servers\.tools_server" "mcp_servers\.tts_server" "mcp_servers\.asr_server"; do
    for pid in $(pgrep -f "$mod" 2>/dev/null); do
        # Skip our own shell if the pattern happens to match it.
        [ "$pid" = "$$" ] && continue
        kill_pid "$pid" "stray($mod)"
    done
done

# --- 3) Verify ports are free -----------------------------------------------
# Uses nc (no lsof dependency). A port is "free" when a TCP connect is refused.

# Python socket connect, not `nc -z -G` (the -G flag is BSD/macOS netcat only;
# Raspberry Pi OS's netcat-traditional rejects it). See start_all.sh.
port_listening() {
    python3 - "$1" <<'PYEOF' >/dev/null 2>&1
import socket, sys
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except Exception:
    sys.exit(1)
PYEOF
}

sleep 1
clean=1
for p in 8003 8002 8001; do
    if port_listening "$p"; then
        echo "  ✗ :$p still has a listener — find it with:  lsof -nP -iTCP:$p -sTCP:LISTEN"
        clean=0
    else
        echo "  ✓ :$p free"
    fi
done

echo
if [ "$clean" -eq 1 ]; then
    echo "MCP servers stopped. Logs preserved in logs/."
else
    echo "Something is still holding a port — see above." >&2
    exit 1
fi