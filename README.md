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
| Tool policy | `agent/tools/*.toml` | `enabled` per tool, `confirm = true` for protected actions, `identity = [...]` binding each spoken approval to its target — the single source of truth |
| Tool gateways | `agent/tool_gateway.py` | progressive disclosure (1 generic tool per heavy server, TOML-generated index) + enforced confirm gate + live-enum hints + rejected-argument stripping (`PARAM_INCOMPATIBILITIES`) + per-turn call caps + result framing |
| Voice bridges | `agent/asr.py`, `agent/tts.py` | drive the asr/tts MCP servers directly (excluded from the LLM schema) |
| Acoustics | `agent/vad.py` + RNNoise filter | 16 kHz in, RNNoise denoise, tuned Silero VAD + **SmartTurn v3.2 semantic end-of-turn** (bundled 8.7 MB ONNX, ~40 ms/verdict) — a mid-thought pause no longer commits a turn; `SMART_TURN=0` falls back to VAD timers only |
| Context | `agent/context.py`, `agent/prompt.py` | system prompt personalized from `data/profile.yaml`; auto context summarization — no memory/RAG: one conversation at a time, history summarized when it grows |
| Identity | `mcp/asr_server.py`, `voice/enroll_voice.py`, `voice/record_voice.py` | voiceprint at `data/voiceprint.json` (0600); gate fails open until enrolled; speaker encoder is a 6 MB ONNX LSTM (`voice/models/`, no torch) |
| Web UI | `ui/server.py`, `ui/static/` | voice UI + one pipeline per browser tab on one origin; warm MCP clients and a pre-synthesized greeting shared by every session |
| Latency meter | `agent/turn_clock.py` | honest per-turn TTFA milestones: the clock re-arms at every VAD stop, so the `TTFA …` log split measures the real wait |
| Launcher | `run.sh` / `pi_run.sh` | laptop flow: `run.sh` creates `.venv` on first run, checks `.env`, starts `ui/server.py`; Pi flow: `pi_run.sh` preflights keys + imports, starts the server in the background and runs `ui/pi_client.py` on the Pi's own mic |

## Our MCP servers (ASR + TTS)

Both are stdio MCP servers built on the MCP SDK 2.x (`MCPServer`), launched
by the agent from `mcp/mcp_servers.json`:

| Server | Tools | Model |
|---|---|---|
| `mcp/asr_server.py` | `transcribe(audio_b64, fmt)`, `enroll(audio_b64, fmt)` | Groq Whisper large-v3-turbo, gated by an on-device speaker encoder (`voice/voice_encoder.py`) |
| `mcp/tts_server.py` | `speak(text)` → Base64 WAV | Piper (ONNX, on-device), voice via `TTS_VOICE` — default `en_US-hfc_female-medium`; the Pi runs `en_US-amy-low` (~30% less compute: hfc_medium measured RTF 0.73–1.16 on a Pi 4, barely realtime), warmed at load |

A tool that fails server-side comes back as an MCP `is_error` result; the
client raises on it (`McpToolError`), so an error message can never become a
transcript or a spoken reply. These tools drive the voice loop directly and
are never exposed to the LLM.

## Token economy (Groq free tier: 8,000 tokens/minute)

Every LLM request re-sends its fixed overhead, so size is both latency and
quota. Each capability server therefore hides behind **one** gateway tool
whose definition is a compact TOML-generated index instead of its full
schemas:

| Server | Full schemas (never sent) | Gateway index (sent) | Saved |
|---|---|---|---|
| github — 10 actions | ~2,540 | ~800 | ~1,740 |
| spotify — 10 actions | ~1,100 | ~530 | ~570 |
| tavily — 2 actions | ~1,610 | ~370 | ~1,240 |
| time — 2 actions | ~290 | ~310 | −20 \* |
| **Total** | **~5,540** | **~2,020** | **~3,520** |

\* the one server whose index is bigger than its raw schemas — they are
tiny already; it stays behind the gateway for the uniform call shape, not
the size.

The rest of each request:

| Piece | Tokens |
|---|---|
| System prompt (identity, date, profile, style) | ~380 |
| Conversation context (auto-summarized past 4,000 tokens / 12 messages) | ≤ ~4,000 |
| **Fixed overhead per request** | **~2,400** |

At full-schema size, one tool-call turn (two requests) would spend
~11,000 tokens — over a whole minute's budget before a single word of
context. At the index's ~2,400 the same turn leaves most of the 8k for
the conversation.

Per action the index carries a one-line operating summary — the TOML
description's first sentence, ≤170 chars — plus the exact argument names
captured live from the server, and the **values** of enum-valued params
(`time_range(day|week|month|year)`): with names only, the model invents
values and burns a turn on the rejection. `include_*` display toggles are
deliberately left off the wire — naming `include_raw_content` would
invite multi-k raw dumps into an 8k-TPM context. Arguments the server
rejects in practice are stripped before the wire and never advertised
(`PARAM_INCOMPATIBILITIES` — tavily's `country` failed in two separate
ways in one evening; each 400 cost a call plus an LLM round trip, and
the model re-learned the lesson every session). Every gateway call
logs its wall time (`tool tavily.tavily_search: 930 ms`) — the ground
truth for "is it the tool or the LLM around it". The model calls
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
  **target** (the TOMLs' `identity = [...]` keys, per action): a "yes, but
  title it X" rides along; a different PR/file/playlist needs fresh approval.
- **Injection**: tool results are delimited `<tool_result>` and the system
  prompt declares them untrusted data.
- **Speaker gate**: only the enrolled owner's speech becomes a turn
  (voiceprint cosine ≥ `ASR_SPEAKER_MIN_SIM`, default **0.50** in code; the
  Pi runs **0.65** — her own Piper voice measured 0.46–0.56 against the
  print, and at the default threshold she transcribed her own echo and
  answered herself); non-owner speech yields no transcript and a "Not your
  voice" toast. With
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
- **Honest TTFA logging**: the per-turn
  `TTFA … — endpointing+ASR …, LLM+TTS …` line measures from the stop that
  **committed** the turn (`agent/turn_clock.py` re-arms at every VAD stop;
  typed tray chips anchor at the typed request). The old clock anchored at
  the first stop of a session segment-chain, so one noise-triggered stop
  produced fake 46–120 s "endpointing+ASR" splits — tune from the new line.
- **One denoiser per client**: browser tabs ask for WebRTC noise
  suppression in `getUserMedia` and connect with `?dn=0`, so those
  sessions skip server-side RNNoise — duplicate denoising on the Pi's
  four cores, frame after frame. Raw-mic clients (`ui/pi_client.py`) keep
  the full front end; the Pi itself runs `RNNOISE=0` since its webrtc AEC
  already suppresses noise at the source (that resampler was burning
  ~70% of a core at idle — CPU Piper needed).
- **No learning 400s**: arguments the server rejects are stripped before
  the wire (measured: tavily's `country` cost two wasted calls plus two
  extra LLM requests per searched question — 4–6 s — before the strip).

### Knobs (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `RNNOISE` | `1` | server-side RNNoise for raw mics; browser tabs auto-skip it via `?dn=0` (they denoise in-page), `0` forces it off everywhere |
| `SMART_TURN` | `1` | `0` = VAD timers only (no SmartTurn inference per pause) |
| `LLM_REASONING_EFFORT` | `low` | GPT-OSS think time before the first token |
| `LLM_KEEPALIVE_SECS` | `45` | keeps the Groq LLM connection open per session (`0` = warm once) |
| `ASR_SPECULATIVE` | `1` | Whisper upload runs in parallel with the speaker gate; rejected voices still reach Groq (`0` = gate first) |
| `ASR_KEEPALIVE_SECS` | `45` | keeps the ASR server's Groq connection open while in use (`0` = warm once) |
| `ASR_MODEL` | `whisper-large-v3-turbo` | the ASR server's model (turbo is the fast one) |
| `ASR_SPEAKER_MIN_SIM` | `0.50` | gate threshold; the Pi runs `0.65` — above her synthetic voice's 0.46–0.56 scores |
| `TTS_VOICE` | `en_US-hfc_female-medium` | the tts server's Piper voice; the Pi runs `en_US-amy-low` (hfc_medium measured RTF 0.73–1.16 on a Pi 4 — barely realtime) |
| `ECHO_SUPPRESS` | `1` | pi_client's echo hold with loud-voice break-in (`0` = off; see AEC) |
| `ECHO_BREAKIN_LEVEL` | `0.08` | peak that opens the mic mid-reply (measured: her echo peaks 0.025 post-AEC, your speech 0.1–1.0) |
| `ECHO_HOLD_SECS` | `2.0` | the rolling window one loud syllable opens — your utterance flows uninterrupted |
| `ECHO_SUPPRESS_TAIL_SECS` | `0.8` | mic stays dark after her last chunk (Bluetooth air time) |

`ASR_*` / `TTS_*` variables are forwarded to their stdio MCP servers
automatically (`mcp_client.py`). Every turn logs its real
time-to-first-audio split (`TTFA … endpointing+ASR …, LLM+TTS …`).

## Spotify connection

The remote Spotify MCP server (`https://spotify-mcp.charrienacor.workers.dev/mcp`
in `mcp_servers.json`) is [spotify-mcp-cloudflare](https://github.com/charrienacor/spotify-mcp-cloudflare)
— deployed from that repo to Cloudflare Workers.

Machine auth by API key: `.env` holds `SPOTIFY_API_KEY`, sent verbatim
(`mcp_servers.json` expands it into the Authorization header). The worker
accepts this key from the owner's agent before its OAuth layer — no browser,
no token expiry, no Inspector dance. Rotate it with
`npx wrangler secret put SPOTIFY_API_KEY` in `spotify-mcp-cloudflare/spotify-mcp`
and update `.env` to match. Browser clients (Claude, Cursor, the Inspector)
are unaffected — they still use the worker's OAuth sign-in flow.

## Raspberry Pi deploy

One command — the Pi's own mic and speaker, everything else follows:

```bash
sudo apt install -y libportaudio2 build-essential   # once (portaudio)
bash pi_run.sh                                      # server + the Pi's voice
```

First run builds the `.venv` (uv if available, else pip) and installs
`websockets`; every run then preflights before starting anything:

- **Boot-critical keys** (`GROQ_API_KEY`, `GROQ_LLM_MODEL`, `LLM_FALLBACK_MODEL`,
  `ASR_MODEL`) fail fast **by name** if missing from `.env`; optional ones
  (`GITHUB_PAT`, `TAVILY_API_KEY`, `SPOTIFY_API_KEY`) warn and that server
  is skipped.
- **Import smoke test** — a partial rsync (a missing agent module) fails in
  seconds instead of surfacing as a server death after the warm-up.
- **Warm wait** — up to ~3 min (MCP connect + Piper voice load), then it
  prints the viewer URL. A partial warmup doesn't kill the run: it says
  which didn't connect and the UI connects per tap (slower, still works).

```bash
bash pi_run.sh stop     # kill a running setup (Ctrl+C stops both too)
```

The laptop or phone just opens the viewer — `http://<pi-ip>:8000/?view=1` —
and mirrors the Pi's conversation; viewers own no mic. Console log:
`logs/pi_server.log` (truncated each run so the warm check stays honest);
the operational log is `logs/web.log`.

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
  for four cores. Open the UI from a phone or laptop — browser tabs skip the
  server-side RNNoise on their own (`?dn=0`), so nothing else needs flipping.
- **Read the TTFA log**: high `endpointing+ASR` → try `SMART_TURN=0`; Piper's
  real-time factor above ~0.5 → `TTS_VOICE=en_US-amy-low` (measured: hfc_medium
  RTF 0.73–1.16 on a Pi 4, amy-low ~30% less compute) and pair it with
  `RNNOISE=0` — the webrtc AEC already suppresses noise at the source, and
  the 48 kHz resampler was burning ~70% of a core at idle; slowest first
  token → try `GROQ_LLM_MODEL=openai/gpt-oss-20b` (some loss in quality).

### Hands-free duplex (AEC)

With a speaker and a mic on the same box the bot hears its own voice and can
interrupt itself — RNNoise is a denoiser, not an echo canceller. A browser
client doesn't need anything (Chrome applies AEC itself), and headphones
bypass the echo path entirely. `pi_run.sh` handles the raw-mic case for you:
it loads the audio stack's echo canceller (`module-echo-cancel`, webrtc) as
`lava_aec` and points the default source/sink at the cancelled twins —
pi_client's raw streams then route through it.

Two measured realities on top of that:

- **Bluetooth defeats the canceller** — the reference signal leads the
  sound by ~200 ms and BT clock drift keeps the echo path moving, so the
  adaptive filter never converges (measured: 5 dB). Wired output (the Pi's
  3.5 mm jack or HDMI) converges properly (expect 20+ dB).
- So `pi_client.py` adds an energy-gated hold (`ECHO_SUPPRESS=1`, the
  default, + `ECHO_SUPPRESS_TAIL_SECS` 0.8 after the last chunk): her echo
  never reaches the VAD — and the path stays **full-duplex**, with the line
  drawn from measurements rather than guesswork (her echo peaks ~0.025 at
  the mic post-AEC, your speech 0.1–1.0): one loud syllable (peak ≥
  `ECHO_BREAKIN_LEVEL`, default 0.08) opens the mic for a rolling
  `ECHO_HOLD_SECS` window so the whole utterance flows continuously and
  barge-in fires mid-reply. `ECHO_SUPPRESS=0` once her output is wired and
  the AEC converges.

The manual AEC equivalent, if you ever need it (PipeWire-Pulse / PulseAudio):

```bash
pactl load-module module-echo-cancel aec_method=webrtc \
  source_name=lava_aec sink_name=lava_aec_sink
pactl set-default-source lava_aec
pactl set-default-sink lava_aec_sink
```

## Enroll the owner's voice

```bash
.venv/bin/python voice/record_voice.py --samples 3   # follow the printed command
```

Enroll through the mic chain you'll actually use — the print matches what
the gate *hears*. Changing the front end invalidates it: a raw-mic
enrollment sagged to ~0.5 similarity once the webrtc AEC entered the path,
the same range her own synthetic voice scores, and the gate could no
longer tell you apart. Re-enroll after any AEC/mic change (the clips are
recorded through the current default source, so the flow above just
works). Deleting `data/voiceprint.json` returns the gate to fail-open.

## Known limits

- Groq free tier: 1,000 requests/day — fine for one user, not multi-tenant.
- The confirm gate governs gateway tools; the asr/tts bridges are voice-loop
  driven and never model-callable by construction.
- Secrets live in `.env` (0600). No liveness detection on the speaker gate.
