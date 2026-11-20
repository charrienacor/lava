#!/usr/bin/env bash
# Lava on the Pi's own mic and speaker — ONE command, everything else follows:
#
#   bash pi_run.sh          # server (background) + pi_client (foreground)
#   bash pi_run.sh stop     # kill a running setup
#
# Ctrl+C stops both. The laptop just opens the viewer:
#   http://<pi-ip>:8000/?view=1
#
# Spotify is remote now: the worker at spotify-mcp.charrienacor.workers.dev
# takes SPOTIFY_API_KEY from .env — no OAuth grant, no browser login, no
# token file to copy.
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"

log()  { printf '\033[1;32mlava-pi\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mlava-pi\033[0m %s\n' "$*" >&2; }

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1; }

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "ui/pi_client.py" 2>/dev/null && log "stopped pi_client" || true
  pkill -f "ui/server.py"   2>/dev/null && log "stopped server"   || true
  exit 0
fi

# 1) venv — same policy as run.sh (first run builds it) ---------------------
if [[ ! -x "$PY" ]]; then
  log "creating .venv (first run — a few minutes)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
    uv pip install --python "$PY" -r requirements.txt
  else
    python3 -m venv .venv
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements.txt
  fi
fi

# websockets: the voice link to the viewer (once; not in requirements.txt)
if ! "$PY" -c "import websockets" >/dev/null 2>&1; then
  log "installing websockets (once, needs internet)…"
  "$PY" -m pip install --quiet websockets
fi

# 2) env + new files present -------------------------------------------------
[[ -f .env ]] || { fail "no .env — Lava needs her keys (GROQ_API_KEY, …)"; exit 1; }
[[ -f ui/pi_client.py ]] || { fail "ui/pi_client.py missing — rsync ui/ agent/ mcp/ from the Mac again"; exit 1; }

# A partial rsync used to surface only after the warm-up wait: the server
# dies at import when a module is missing (agent/turn_clock.py joined the
# set recently). One cheap import here fails in seconds instead.
if ! "$PY" -c "import sys; sys.path.insert(0, 'agent'); import pipeline" >/dev/null 2>&1; then
  fail "agent/ failed to import — a module is missing or stale."
  fail "rsync agent/ (and mcp/) from the Mac again"
  exit 1
fi

# 3) API keys — boot-critical ones block, optional ones warn -------------------
# Names only, never values.
env_has() { grep -qE "^[[:space:]]*$1=" .env; }
for k in GROQ_API_KEY GROQ_LLM_MODEL LLM_FALLBACK_MODEL ASR_MODEL; do
  env_has "$k" || { fail ".env is missing $k — Lava cannot boot without it"; exit 1; }
done
for k in GITHUB_PAT TAVILY_API_KEY SPOTIFY_API_KEY; do
  env_has "$k" || log "note: no $k in .env — that MCP server is skipped, its tools stay off"
done

# 4) already running? ---------------------------------------------------------
if port_open 8000; then
  fail "something already listens on :8000 — is Lava up? stop first: bash pi_run.sh stop"
  exit 1
fi

# 5) server in the background, wait for the warm log line --------------------
mkdir -p logs
# Fresh log each run: the wait loop greps for THIS run's warm line — an
# appended old one would match instantly and hand pi_client a cold server.
: > logs/pi_server.log
log "starting the server (MCP + Piper warm-up — up to ~3 min on the Pi)…"
nohup "$PY" ui/server.py >> logs/pi_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

WARM="Lava web UI"
DEGRADED="warmup failed"
warm=""
for _ in $(seq 1 180); do
  if grep -q "$WARM" logs/pi_server.log 2>/dev/null; then warm="ok"; break; fi
  # partial warmup (a server didn't connect): the UI still works, per tap
  if grep -q "$DEGRADED" logs/pi_server.log 2>/dev/null; then warm="degraded"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "server died — last lines of logs/pi_server.log:"
    tail -5 logs/pi_server.log >&2
    exit 1
  fi
  sleep 1
done
case "$warm" in
  ok)
    log "server warm (pid $SERVER_PID · log: logs/pi_server.log)" ;;
  degraded)
    log "server is UP but warmup failed (some MCP server didn't connect) —"
    log "sessions will connect per tap: slower first turn. See logs/pi_server.log" ;;
  "")
    fail "server not warm after 3 min — see logs/pi_server.log"
    exit 1 ;;
esac

# 6) the voice — this terminal owns mic + speaker until Ctrl+C ---------------
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
log "laptop viewer:  http://${IP:-<pi-ip>}:8000/?view=1"
log "talk to Lava — Ctrl+C stops the voice AND the server"
"$PY" ui/pi_client.py

log "stopped"
