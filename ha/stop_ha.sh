#!/usr/bin/env bash
# stop_ha.sh - stop the virtual Home Assistant hub started by start_ha.sh.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
ROOT="$(dirname "$SCRIPT_DIR")"
PIDFILE="$ROOT/logs/ha.pid"

if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "  Home Assistant (pid $PID) stopping..."
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
        echo "  Home Assistant stopped"
    else
        echo "  Home Assistant not running (stale pidfile)"
    fi
    rm -f "$PIDFILE"
else
    echo "  no pidfile at $PIDFILE - nothing to stop"
fi
