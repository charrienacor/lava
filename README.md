# Lava

Full-duplex voice agent for a Raspberry Pi 4: speaks and listens at the same
time, runs its tools through MCP, and treats its own token budget, its
speaker's identity, and destructive actions as first-class constraints.

```
voice in → RNNoise → Silero VAD → ASR (MCP: Groq Whisper large-v3-turbo, speaker-gated)
        → LLM (Groq gpt-oss-120b + MCP tools via gateways) → TTS (MCP: Piper)
        → voice out        (barge-in cancels TTS/LLM at any moment)
```

## Architecture

| Piece | File | Role |
|---|---|---|
| Orchestrator | `agent/pipeline.py` | entry point, transports, `.env` load (single source), audio rates |
| MCP layer | `agent/mcp_client.py` | one client per server (`mcp/mcp_servers.json`), allowlists, tool-call audit |
| Tool policy | `agent/tools/*.toml` | `enabled` per tool + `confirm = true` for protected actions — the single source of truth |
| Tool gateways | `agent/tool_gateway.py` | progressive disclosure (1 generic tool per heavy server, TOML-generated index) + enforced confirm gate + result framing |
| Voice bridges | `agent/asr.py`, `agent/tts.py` | drive the asr/tts MCP servers directly (excluded from the LLM schema) |
| Acoustics | `agent/vad.py` + RNNoise filter | 16 kHz in, RNNoise denoise, tuned Silero VAD + **SmartTurn v3.2 semantic end-of-turn** (bundled 8.7 MB ONNX, ~40 ms/verdict) — a mid-thought pause no longer commits a turn; `SMART_TURN=0` falls back to VAD timers only |
| Context | `agent/context.py`, `agent/prompt.py` | system prompt personalized from `data/profile.yaml`; auto context summarization |
| Identity | `mcp/asr_server.py`, `agent/enroll_voice.py`, `agent/record_voice.py` | voiceprint at `data/voiceprint.json` (0600); gate fails open until enrolled |
| Deploy | `deploy/lava.service`, `deploy/install_rpi.sh` | systemd supervision (`Restart=always`), one-command RPi install |

## Token economy (Groq free tier: 8,000 tokens/minute)

Full tool schemas cost ~8,500 tokens per request — over the whole budget.
Every capability server therefore hides behind **one** gateway tool whose
definition is a compact TOML-generated index: **~1,216 tokens total**
(github 504, spotify 367, tavily 174, time 171). The model calls
`github(action="merge_pull_request", arguments={...})`; the gateway validates
the action against the TOML and forwards over the real MCP session. The OpenAI
client retries 429/5xx with backoff (`llm.py`), and long conversations
self-compress (context summarization at 4,000 tokens / 12 messages).

## Safety model

- **Allowlist**: only TOML-enabled tools are callable, at the MCP client and
  again at the gateway. Disabled tools are unreachable even via the gateway.
- **Confirm gate**: TOML-flagged actions cannot run from one call. First call
  → `CONFIRMATION_REQUIRED` (the model must ask out loud). The re-call needs
  `confirmed=true` *and* a real user turn in the context since the request —
  the model cannot confirm for the user. Approval binds to the action's
  **target** (`_IDENTITY_KEYS`): a "yes, but title it X" rides along; a
  different PR/file/playlist needs fresh approval.
- **Injection**: tool results are delimited `<tool_result>` and the system
  prompt declares them untrusted data.
- **Speaker gate**: ASR transcribes only the enrolled owner (voiceprint
  cosine ≥ 0.75, tunable via `ASR_SPEAKER_MIN_SIM`); non-owner speech yields
  no transcript. Known limits: threshold untested against a real second
  human; replayed recordings of the owner pass — no liveness detection.
- **Audit**: tool calls, retries, and gate denials are visible live in the
  web UI (tray toasts) and in `logs/web.log`.

## Run

```bash
# macOS dev: voice web client on :7860
.venv/bin/python agent/pipeline.py --transport websocket

# CLI: health check (exit 0/1 — cron-able)
.venv/bin/python agent/pipeline.py --check-mcp

# tests
.venv/bin/python -m pytest
```

## Spotify connection

No OAuth helper: the bearer lives in `.env` as `SPOTIFY_ACCESS_TOKEN` and is
sent verbatim (`mcp_servers.json` expands `${SPOTIFY_ACCESS_TOKEN}` into the
Authorization header). Renew it yourself when it expires — e.g. with
`npx @modelcontextprotocol/inspector` against the spotify server — then
restart Lava: the bearer is frozen into the client at connect time. An
expired token just makes the spotify tools fail with a 401; everything else
keeps working.

## Raspberry Pi deploy

```bash
sudo apt install -y libportaudio2 build-essential
./deploy/install_rpi.sh          # deps + venv + systemd enable
./deploy/setup_aec_rpi.sh        # hands-free duplex: system-level echo cancellation
systemctl status lava            # Restart=always: crashes heal in 5 s
journalctl -u lava -f            # operational log
```

### Hands-free duplex (AEC)

With a speaker and a mic on the same box the bot hears its own voice and can
interrupt itself — RNNoise is a denoiser, not an echo canceller.
`deploy/setup_aec_rpi.sh` fixes this at the audio-stack level, with **no
client code changes**: it loads PipeWire/PulseAudio's `module-echo-cancel`
(webrtc backend, speex fallback) as a virtual microphone `lava_aec`, makes it
the default capture source, and persists the config. Any client capturing
`default` — or the source named "LavaAEC" — sends echo-cancelled audio.
Verify with the loopback test it prints at the end. A browser-based client
doesn't need this (Chrome applies AEC itself); headphones also bypass the
echo path entirely.

## Enroll the owner's voice

```bash
.venv/bin/python agent/record_voice.py --samples 3   # follow the printed command
```

Deleting `data/voiceprint.json` returns the gate to fail-open.

## Known limits

- Groq free tier: 1,000 requests/day — fine for one user, not multi-tenant.
- The confirm gate governs gateway tools; the asr/tts bridges are voice-loop
  driven and never model-callable by construction.
- Secrets live in `.env` (0600). No liveness detection on the speaker gate.
