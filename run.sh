#!/usr/bin/env bash
# Lava launcher — one command from clean checkout to a warm voice agent:
#
#   1. virtual environment   created on first run (uv if available, else venv)
#   2. .env check            the keys (GROQ_API_KEY, …) live there
#   3. server                ui/server.py — its own warmup connects the MCP
#                            servers, loads the TTS voice, synthesizes the
#                            greeting, and only then prints the URL
#
# Extra args pass through:  ./run.sh --port 8000
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"

log()  { printf '\033[1;33mlava\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mlava\033[0m %s\n' "$*" >&2; }

# 1) virtual environment ---------------------------------------------------
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

# 2) environment -------------------------------------------------------------
if [[ ! -f .env ]]; then
  fail "no .env found — Lava needs GROQ_API_KEY (and friends) in it to run"
  exit 1
fi

# 3) server -------------------------------------------------------------------
# exec: the shell process is replaced, so Ctrl-C and pass-through args
# (--port …) behave exactly like running ui/server.py directly.
exec "$PY" ui/server.py "$@"
