#!/usr/bin/env bash
#
# run.sh — one-shot launcher: creates .venv if needed, installs dependencies,
# starts the three MCP servers, then runs the web bridge in the foreground.
#
#   ./run.sh
#
# Steps:
#   1. .venv missing?  -> python3 -m venv .venv && pip install -r requirements.txt
#   2. .env missing / no GROQ_API_KEY? -> abort with instructions
#   3. deploy/start_all.sh -> asr :8001, tts :8002, tools :8003 (waits until ready)
#   4. python -m web.server -> :8080, in the FOREGROUND (Ctrl-C stops only the
#      bridge; the MCP servers keep running until ./deploy/stop_all.sh)
#
set -u

# Works from any CWD: this script lives at the project root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
ROOT="$SCRIPT_DIR"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"

# --- 1) virtualenv -----------------------------------------------------------

if [ ! -x "$PY" ]; then
    echo "Creating virtualenv at $ROOT/.venv ..."
    python3 -m venv .venv || { echo "ERROR: could not create .venv" >&2; exit 1; }
    echo "Installing dependencies (first run can take a few minutes)..."
    .venv/bin/pip install --quiet --upgrade pip || { echo "ERROR: pip upgrade failed" >&2; exit 1; }
    .venv/bin/pip install --quiet -r requirements.txt || { echo "ERROR: pip install -r requirements.txt failed" >&2; exit 1; }
    echo "  ✓ .venv ready"
else
    echo "  ✓ .venv already present"
fi

# --- 2) .env ------------------------------------------------------------------

if [ ! -f "$ROOT/.env" ]; then
    echo "ERROR: no .env at $ROOT/.env" >&2
    echo "       create it with at least:  GROQ_API_KEY=***   (https://console.groq.com/keys)" >&2
    exit 1
fi
if ! grep -q '^GROQ_API_KEY=.' "$ROOT/.env"; then
    echo "ERROR: GROQ_API_KEY is not set in $ROOT/.env — the LLM and ASR need it." >&2
    exit 1
fi

# --- 3) MCP servers ------------------------------------------------------------

"$ROOT/deploy/start_all.sh" || exit 1

# --- 4) web bridge (foreground) -------------------------------------------------

# Free port 8080 if a previous bridge is still holding it (e.g. a run.sh whose
# terminal was closed, or a hand-started server). Without this the bridge dies
# with: OSError [Errno 48] address already in use.
if pgrep -f "web\.server" >/dev/null 2>&1; then
    echo "  ! stale web bridge still running — stopping it first"
    pkill -f "web\.server" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -f "web\.server" >/dev/null 2>&1 || break
        sleep 1
    done
    if pgrep -f "web\.server" >/dev/null 2>&1; then
        pkill -9 -f "web\.server" 2>/dev/null
        sleep 1
    fi
fi

echo
echo "Starting web bridge -> http://127.0.0.1:8080"
echo "  Ctrl-C stops only the bridge; stop the MCP servers with:  ./deploy/stop_all.sh"
echo
exec "$PY" -m web.server
