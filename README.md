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
| Context | `agent/context.py`, `agent/prompt.py` | system prompt personalized from `data/profile.yaml`; auto context summarization — no memory/RAG: one conversation at a time, history summarized when it grows |
| Identity | `mcp/asr_server.py`, `agent/enroll_voice.py`, `agent/record_voice.py` | voiceprint at `data/voiceprint.json` (0600); gate fails open until enrolled; speaker encoder is a 6 MB ONNX LSTM (`models/`, no torch) |
| Web UI | `ui/server.py`, `ui/static/` | voice UI + one pipeline per browser tab on one origin; warm MCP clients and a pre-synthesized greeting shared by every session |
| Launcher | `run.sh` | creates `.venv` on first run, checks `.env`, starts `ui/server.py` |

## Our MCP servers (ASR + TTS)

Both are stdio MCP servers built on the MCP SDK 2.x (`MCPServer`), launched
by the agent from `mcp/mcp_servers.json`:

| Server | Tools | Model |
|---|---|---|
| `mcp/asr_server.py` | `transcribe(audio_b64, fmt)`, `enroll(audio_b64, fmt)` | Groq Whisper large-v3-turbo, gated by an on-device speaker encoder (`mcp/voice_encoder.py`) |
| `mcp/tts_server.py` | `speak(text)` → Base64 WAV | Piper `en_US-hfc_female-medium` (ONNX, on-device), warmed at load |

A tool that fails server-side comes back as an MCP `is_error` result; the
client raises on it (`McpToolError`), so an error message can never become a
transcript or a spoken reply. These tools drive the voice loop directly and
are never exposed to the LLM.

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
- **Speaker gate**: only the enrolled owner's speech becomes a turn
  (voiceprint cosine ≥ `ASR_SPEAKER_MIN_SIM`, default **0.50** in code);
  non-owner speech yields no transcript and a "Not your voice" toast. With
  `ASR_SPECULATIVE=1` (default) the audio is uploaded to Groq while the gate
  runs and a rejected segment's transcript is discarded — set it to `0` if
  non-owner audio must never leave the device. Known limits: threshold
  untested against a real second human; replayed recordings of the owner
  pass — no liveness detection.
- **Audit**: tool calls, retries, and gate denials are visible live in the
  web UI (tray toasts) and in `logs/web.log`.

## Run

```bash
# voice web UI on :8000 (first run builds .venv) — the normal way
./run.sh                    # or: ./run.sh --port 8080

# pipecat dev runner on :7860 (Protobuf websocket client)
.venv/bin/python agent/pipeline.py --transport websocket

# CLI: health check (exit 0/1 — cron-able)
.venv/bin/python agent/pipeline.py --check-mcp

# tests
.venv/bin/python -m pytest
```

## Latency

What the voice loop does to keep time-to-first-audio low:

- **Warm everything at startup**: MCP servers connect once and are shared by
  all sessions; Piper runs a throwaway synthesis at load; the greeting is
  pre-synthesized and streamed from cache.
- **Hot connections**: the ASR server's Groq client never expires idle
  connections and pings while a conversation is active; each session warms
  its LLM connection while the greeting plays (a cold Groq request measured
  ~590 ms vs ~250 ms warm).
- **Gate in parallel**: the Whisper upload starts while the speaker gate
  computes its embedding.
- **Streaming TTS**: each sentence is split at punctuation; the next chunk is
  synthesized while the current one plays, with no silent lead-in.
- **Short first sentence**: the prompt asks the model to lead with a short
  sentence, since speech starts only once it is complete.
- **Fast end-of-turn**: SmartTurn lets the VAD stop window stay at 0.2 s.

### Knobs (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `RNNOISE` | `1` | `0` skips server-side RNNoise — the browser already denoises; saves Pi CPU |
| `SMART_TURN` | `1` | `0` = VAD timers only (no SmartTurn inference per pause) |
| `LLM_REASONING_EFFORT` | `low` | GPT-OSS think time before the first token |
| `LLM_KEEPALIVE_SECS` | `45` | keeps the Groq LLM connection open per session (`0` = warm once) |
| `ASR_SPECULATIVE` | `1` | Whisper upload runs in parallel with the speaker gate; rejected voices still reach Groq (`0` = gate first) |
| `ASR_KEEPALIVE_SECS` | `45` | keeps the ASR server's Groq connection open while in use (`0` = warm once) |
| `ASR_MODEL` | `whisper-large-v3-turbo` | the ASR server's model (turbo is the fast one) |

`ASR_*` / `TTS_*` variables are forwarded to their stdio MCP servers
automatically (`mcp_client.py`). Every turn logs its real
time-to-first-audio split (`TTFA … endpointing+ASR …, LLM+TTS …`).

## Spotify connection

The remote Spotify MCP server (`https://spotify-mcp.charrienacor.workers.dev/mcp`
in `mcp_servers.json`) is [spotify-mcp-cloudflare](https://github.com/charrienacor/spotify-mcp-cloudflare)
— deployed from that repo to Cloudflare Workers.

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
./run.sh                          # first run builds .venv, then serves :8000
```

To keep it running, a systemd unit (adjust `User` and paths), saved as
`/etc/systemd/system/lava.service`:

```ini
[Unit]
Description=Lava voice agent
After=network-online.target sound.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/lava
ExecStart=/home/pi/lava/.venv/bin/python ui/server.py
Restart=always
RestartSec=5
Nice=-10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now lava
journalctl -u lava -f             # operational log
```

### Pi 4 performance tuning

- **OS**: 64-bit Raspberry Pi OS Lite (no desktop) — onnxruntime's aarch64
  build is faster than 32-bit.
- **Power and cooling first**: official 5.1 V / 3 A supply plus heatsink and
  fan. `vcgencmd get_throttled` must print `0x0`; anything else means
  under-voltage or thermal throttling.
- **CPU governor**: `performance` removes the ramp-up delay after idle:
  `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
  (persist it with a oneshot systemd unit or `cpufrequtils`).
- **Overclock** (`/boot/firmware/config.txt`): `arm_freq=2000`,
  `over_voltage=6`. Stress-test while watching `vcgencmd measure_temp` and
  stay under ~75 °C. Avoid `force_turbo=1` — combined with over-voltage it
  can permanently set the warranty bit.
- **Headless trims**: `gpu_mem=16`, `dtoverlay=disable-bt`, disable unused
  services.
- **Network**: wired Ethernet — every turn makes ~3 round trips to Groq, so
  jitter is latency. On Wi-Fi: `sudo iw wlan0 set power_save off`.
- **Storage**: boot from a USB 3 SSD instead of the SD card; use zram swap.
- **Browser off the Pi**: Chromium competes with Silero, SmartTurn and Piper
  for four cores. Open the UI from a phone or laptop and set `RNNOISE=0`.
- **Read the TTFA log**: high `endpointing+ASR` → try `SMART_TURN=0`; Piper's
  real-time factor above ~0.5 → a lighter voice; slowest first token → try
  `GROQ_LLM_MODEL=openai/gpt-oss-20b` (some loss in quality).

### Hands-free duplex (AEC)

With a speaker and a mic on the same box the bot hears its own voice and can
interrupt itself — RNNoise is a denoiser, not an echo canceller. A browser
client doesn't need anything (Chrome applies AEC itself), and headphones
bypass the echo path entirely. For a raw local mic, load the audio stack's
echo canceller as the default source (PipeWire-Pulse / PulseAudio):

```bash
pactl load-module module-echo-cancel aec_method=webrtc \
  source_name=lava_aec sink_name=lava_aec_sink
pactl set-default-source lava_aec
pactl set-default-sink lava_aec_sink
```

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
