# Lava — full-duplex voice agent

Lava is a voice assistant you talk to from a browser tab. No wake word: tap
once to open the mic, then just talk. It answers out loud while still
listening — so you can interrupt it mid-sentence by simply speaking. When you
go quiet, it asks if you're still there before turning the mic off.

Everything runs on free services: speech recognition and the brain are on
Groq's free tier, the voice (Piper) is fully local. Developed on a laptop
(macOS), built for a Raspberry Pi. Python ≥ 3.12.

```
browser mic ──▶ Silero VAD ──▶ [ASR :8001] ──▶ Groq LLM (+ [tools :8003]) ──▶ [TTS :8002] ──▶ browser speaker
     ▲                                                                                 │
     └──────────── you talk while it answers → it stops and listens (barge-in) ────────┘
```

**Contents:** [Quick start](#quick-start) · [Talking to it](#talking-to-it) ·
[How it works](#how-it-works) · [Configuration](#configuration) ·
[Hosted voice](#optional-hosted-voice-groq-orpheus) ·
[Raspberry Pi](#raspberry-pi) · [Latency benchmark](#latency-benchmark) ·
[Troubleshooting](#troubleshooting) · [Project layout](#project-layout)

---

## Quick start

You need **Python ≥ 3.12** and a free **Groq API key**
(https://console.groq.com/keys — the free tier is enough).

**1. Configure** — create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

**2. Launch** — one command from the project root:

```bash
./run.sh
```

The first run creates a virtualenv and installs dependencies (a few minutes),
then starts the three MCP servers and the web bridge. When you see
`Starting web bridge -> http://127.0.0.1:8080`, open that URL.

**3. Talk** — allow the microphone, then tap the orb.

`GET /api/status` shows the live model, loaded tools, city, and session count.

---

## Talking to it

- **Just talk.** No wake word. Ask for the time, weather, a web search,
  Wikipedia; set timers; save reminders — it answers out loud in a sentence
  or two.
- **Interrupt it** by talking while it's answering: it stops immediately and
  listens to what you say next (barge-in).
- **Going quiet:** after ~25 s of silence it asks *"are you still there?"*
  (rotating wordings); after ~45 s of total silence the mic turns off. Tap
  the orb to resume — the conversation is remembered per browser tab for
  24 h.
- **End the session** by tapping the orb while it's live. Tap again to start
  a new one.

### Stopping and restarting

- `Ctrl-C` in the `run.sh` terminal stops only the web bridge.
- The MCP servers keep running in the background — stop them with:

```bash
./deploy/stop_all.sh
```

Re-running `./run.sh` is safe: it clears a stale bridge on `:8080` and skips
MCP servers that are already up. Logs for every process are in
`logs/<name>.log` (rotated at 10 MB).

Manual mode (one terminal per process), if you prefer:

```bash
.venv/bin/python -m mcp_servers.asr_server      # :8001
.venv/bin/python -m mcp_servers.tts_server      # :8002
.venv/bin/python -m mcp_servers.tools_server    # :8003
.venv/bin/python -m web.server                  # :8080
```

---

## How it works

### Audio transport

The browser captures the mic with echo cancellation and noise suppression on,
converts to 16-bit PCM inside an AudioWorklet thread, and streams 32 ms
binary frames over a WebSocket — no base64, no JSON per audio frame. Reply
audio streams back the same way: a 2-byte sample-rate header followed by
16-bit PCM chunks, which the browser schedules gaplessly.

### Turn detection (VAD)

The server runs Silero VAD plus a loudness gate on every 32 ms frame.
A "span" opens on speech and closes after 0.2 s of silence; the span's audio
(plus 120 ms of lead-in, so first words survive) goes to Whisper. Two extra
guards keep room noise from becoming turns: spans that are mostly
non-speech are still transcribed but Whisper's own quality gate drops
degraded audio, and the LLM is instructed to silently `SKIP` speech that
isn't addressed to it (TV dialogue, ads, background voices).

### Barge-in

200 ms (`VAD_BARGE_MS`) of detected speech cuts the bot instantly — audio
stops client-side, the in-flight turn is cancelled, and your sentence becomes
the next turn. Browser echo cancellation keeps the bot's own voice from
triggering it.

### Response pipeline

Whisper transcribes → the LLM streams a reply, calling tools (time, weather,
search, Wikipedia, timers, notes, light, Pi health) whenever they can answer
→ sentences are split for speech and synthesized while the LLM keeps
streaming (a `Speaker` pipeline: sentence N+1 is synthesized while N plays)
→ the first clause goes out before the full sentence exists (a comma after
28 chars is a split point). Fixed lines — greeting, apologies, idle
check-ins — are pre-synthesized at startup and replay from cache in ~10 ms.

### Idle handling

`idle_watch` tracks real voice, not raw frames. After `IDLE_PROMPT_SECS`
(25 s) of quiet it speaks a check-in ("Hi, are you still there?" — rotating
wordings); talking over the check-in barges it like any other speech. After
`IDLE_CHECK_SECS` (45 s) of total silence the mic turns off; the next orb tap
resumes the same conversation.

### Reliability

The bridge holds persistent connections to the three MCP servers, probes
each every 10 s, and reconnects automatically if one dies and comes back.
TTS has a fallback chain (`TTS_FALLBACK`) so a quota or network failure on
the primary engine degrades to Piper instead of going silent. LLM quota
errors walk a model failover chain.

---

## Configuration

All configuration lives in `.env` (see `.env` keys reference below). Defaults
are sensible — the keys you're most likely to touch first:

| Key | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Free key from https://console.groq.com/keys |
| `GROQ_LLM_MODEL` | `openai/gpt-oss-20b` | Primary LLM |
| `VAD_MIN_VOLUME` | `0.7` | Loudness floor — raise if TV/room noise triggers turns |
| `VAD_BARGE_MS` | `0.2 s` | Speech burst that cuts the bot mid-reply (barge-in speed vs TV false cuts) |
| `IDLE_PROMPT_SECS` / `IDLE_CHECK_SECS` | `25` / `45` | Silence before the spoken "still there?" check-in / before the mic turns off |
| `TTS_BACKEND` | `piper` | `piper` (local) or `groq` (hosted, English-only — see below) |
| `DEFAULT_CITY` / `DEFAULT_TIMEZONE` | `Quezon City` / `Asia/Manila` | Used by time, weather, and identity |
| `WEB_PORT` | `8080` | Bridge port (`ASR_PORT` / `TTS_PORT` / `TOOLS_PORT` for the MCPs) |

### Full reference

**Bridge & LLM** (`web/server.py`)

| Key | Default | Meaning |
|---|---|---|
| `GROQ_LLM_MODEL` | `openai/gpt-oss-20b` | Primary LLM |
| `GROQ_LLM_MODELS` | — | Comma-separated failover chain tried in order on quota errors (429) |
| `LLM_TEMPERATURE` | `0.8` | Sampling consistency for voice replies |
| `LLM_MAX_TOKENS` | `768` | Bounds a runaway generation |
| `TURN_DEADLINE_SECS` | `45` | Hard cap on one turn (ASR → tools → speech) |
| `TOOL_TIMEOUT_SECS` | `12` | Per-tool-call timeout |
| `TTS_FIRST_CHUNK_CHARS` | `28` | Earliest comma split for the first spoken chunk |
| `ASR_MCP_URL` / `TTS_MCP_URL` / `TOOLS_MCP_URL` | `http://127.0.0.1:800x/mcp` | Backend endpoints |

**VAD & turn detection**

| Key | Default | Meaning |
|---|---|---|
| `VAD_CONFIDENCE` | `0.7` | Silero speech-likelihood threshold |
| `VAD_MIN_VOLUME` | `0.7` | BS.1770 loudness floor (rejects far-away speech) |
| `VAD_START_SECS` / `VAD_STOP_SECS` | `0.2` / `0.2` | Span open/close silence windows |
| `VAD_MIN_VOICED_MS` | `450` | Minimum voiced length for an utterance |
| `VAD_MIN_VOICED_RATIO` | `0.30` | Span voiced fraction below which it's treated as noise |
| `VAD_BARGE_MS` | `0.2 s` | Speech burst that cuts the bot mid-reply |
| `VAD_PRE_ROLL_MS` | `120` | Lead-in audio included before first voiced frame (first-word capture) |
| `VAD_ATTACK` / `VAD_RELEASE` | `0.2` / `0.2` | Loudness smoothing (keep symmetric in noisy rooms) |
| `VAD_PROCESS_NOISY_SPANS` | `true` | Transcribe spans the ratio gate would discard — recovers commands spoken over TV |

**ASR** (`mcp_servers/asr_server.py`)

| Key | Default | Meaning |
|---|---|---|
| `ASR_BACKEND` | `groq` | Speech-to-text engine |
| `GROQ_ASR_MODEL` | `whisper-large-v3-turbo` | Groq Whisper model |
| `ASR_LANGUAGE` | `en` | ISO-639-1 hint (`en`, `tl`, …) |
| `ASR_NO_SPEECH_MAX` / `ASR_LOGPROB_MIN` | `0.8` / `-1.3` | Quality gate: drop segments Whisper itself flags as unintelligible |

**TTS** (`mcp_servers/tts_server.py`)

| Key | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `piper` | `piper` (local) or `groq` (hosted) |
| `PIPER_VOICE` | `en_US-hfc_female-medium` | Voice id (auto-downloads to `models/piper/`) |
| `PIPER_LENGTH_SCALE` | `1.0` | `< 1` speaks faster (try `0.9` on a Pi) |
| `TTS_FALLBACK` | `piper` when primary isn't piper | Engine used when the primary fails |
| `GROQ_TTS_MODEL` | `canopylabs/orpheus-v1-english` | Hosted model (with `TTS_BACKEND=groq`) |
| `GROQ_TTS_VOICE` | `hannah` | `autumn` / `diana` / `hannah` / `austin` / `daniel` / `troy` |
| `GROQ_TTS_MAX_CHARS` | `180` | Hosted voice input cap; longer text is split and concatenated |

**Idle & infrastructure**

| Key | Default | Meaning |
|---|---|---|
| `IDLE_PROMPT_SECS` / `IDLE_CHECK_SECS` | `25` / `45` | Check-in after this much quiet / mic off after this much total silence |
| `DEFAULT_CITY` / `DEFAULT_TIMEZONE` | `Quezon City` / `Asia/Manila` | Time, weather, and identity |
| `MCP_HOST` | `127.0.0.1` | Bind host for the three MCP servers |
| `ASR_PORT` / `TTS_PORT` / `TOOLS_PORT` | `8001` / `8002` / `8003` | MCP server ports |
| `WEB_PORT` | `8080` | Bridge port |
| `MODELS_DIR` | `./models` | Where the Piper voice is stored |

---

## Optional hosted voice (Groq Orpheus)

Piper is fully offline; if you'd rather have a hosted neural voice, Groq
offers Orpheus with the same free key. One-time setup: open
https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english and
accept the model terms, then set in `.env`:

```bash
TTS_BACKEND=groq
GROQ_TTS_VOICE=hannah     # autumn/diana/hannah/austin/daniel/troy
```

English only. Longer sentences are split at the 180-character API cap and
stitched automatically. If the hosted voice hits a quota error or the network
drops, `TTS_FALLBACK=piper` covers the gap so speech never dies mid-sentence.

---

## Raspberry Pi

- Copy the project folder **without** `.venv/` (virtualenvs don't survive a
  copy); `./run.sh` rebuilds it on the Pi.
- **Piper is the bottleneck, not the VAD.** `-medium` voices have RTF ≈ 0.5–1
  on a Pi 4. The pipelined Speaker hides most of it behind playback, but for
  snappier first audio set `PIPER_LENGTH_SCALE=0.9`, or switch `PIPER_VOICE`
  to a `-low` quality model.
- `TTS_BACKEND=groq` moves synthesis off the Pi entirely (see above) — often
  the biggest single win on Pi hardware.
- The browser does the int16 conversion and batches to 32 ms frames inside
  the AudioWorklet thread, so the Pi's main thread only forwards
  ArrayBuffers.
- Watch `ttfa_ms` in the turn stats (also in the logs) — on a Pi expect
  ~1.2–2 s with a tool call, ~0.9–1.4 s without.

---

## Latency benchmark

`tests/bench_latency.py` measures the interaction loop the same way every
time. It is designed for classroom use: no per-student setup, no
modifications, and identical stimulus audio for everyone.

```bash
./run.sh                                        # terminal 1: the stack
.venv/bin/python tests/bench_latency.py --runs 3 --label your_name
```

How it stays standardized:

- Stimulus phrases are pre-rendered WAVs in `tests/assets/` (committed to
  the repo). The benchmark loads them with the stdlib only — no TTS at
  runtime. If the folder is missing it regenerates once via
  `tests/generate_assets.py` (needs Piper, which `./run.sh` installs).
- Every run opens a **fresh session** with a unique id, so results never
  depend on conversation state.
- Audio is streamed at real-time pace as binary 32 ms frames — exactly what
  the browser does — so human speaking variance is removed and runs are
  comparable across machines and days.
- Results are written to `tests/results/latency_<label>_<timestamp>.json`
  with per-turn raw numbers, a summary (median / p95 / min / max), the
  host name, and the stimuli used — submit that file.

Scenarios: `no_tool` (pure LLM reply), `tool_time` (get_time),
`tool_weather` (get_weather — the slowest common tool), and `barge_cut`
(interrupting the greeting and measuring the cut). Each runs `--runs`
times and also reports the bridge's own component breakdown per turn.

Metric definitions:

- **client_ttfa** — last mic-speech byte sent → first reply-audio byte
  received. What a user perceives: VAD end-of-speech detection (~0.2 s) +
  ASR + first clause + first TTS chunk + transport.
- **client_total** — speech end → the whole reply streamed.
- **server ttfa_ms** — the bridge's own measure (turn start → first audio
  streamed). Expect `client_ttfa ≈ server ttfa_ms + 0.2–0.3 s`.
- **barge_cut** — first byte of interrupting speech → the bot's audio is
  cut (~0.8 s measured; the VAD must first accumulate `VAD_BARGE_MS` of
  gated speech over the bot's own audio).

### Reading the output

```text
=== summary (ms; median / p95) ===
scenario        n         client_ttfa        client_total           asr           llm         tts      ttfa_srv
no_tool         3       1440/   2217       1563/   2291           489           891           168          1409
tool_time       3       1825/   1936       1826/   1936           412          1381             0          1794
tool_weather    3       3995/   4384       4103/   4501           581          3635           157          4110
barge_cut       3        788/    788          -/      -             -             -             -             -
```

- **n** successful runs. **median/p95** = typical run / worst run — a p95 far
  above the median is jitter (weak WiFi or a throttled free tier), not a
  code problem.
- **client_ttfa** is the number a user feels: finished speaking → first
  reply audio. **ttfa_srv** measures the same event server-side; the small
  gap between them (~30–300 ms) is the VAD stop-detection plus transport —
  their agreement is the sanity check that both clocks are correct.
- **asr / llm / tts** attribute the time: `asr` isolates the network leg
  (upload + round trips), `llm` covers Groq generation *and tool waits*
  (weather turns are slow because the weather API is slow, not the LLM),
  `tts` is local Piper CPU — expect it to rise sharply on a Pi.
- `barge_cut` has no component columns: a barge is not a turn, just VAD
  detection + the cut.

Reference numbers (Mac laptop, wired WiFi, free tier): no_tool ≈ 1.4 s,
tool_time ≈ 1.8 s, tool_weather ≈ 4 s, barge_cut ≈ 0.8 s. Compare rows, not
machines: `client_ttfa` for overall feel, `asr` for network quality, `tts`
for local CPU.

Uses a few Groq calls per run — keep `--runs` modest on the free tier.

## Troubleshooting

- **`Address already in use` on :8080** — a stale bridge holds the port.
  `./run.sh` clears it automatically; otherwise
  `lsof -iTCP:8080 -sTCP:LISTEN` (or `pkill -f "web.server"`), then relaunch.
- **The agent stopped responding entirely** — an MCP server probably died.
  The bridge reconnects on its own once the backend is back; bring backends
  up with `./deploy/start_all.sh` (or just re-run `./run.sh`). Check
  `logs/<name>.log` for the culprit.
- **It answers TV dialogue / room chatter** — raise `VAD_MIN_VOLUME` first
  (soft voices start getting eaten above ~0.76), then `VAD_MIN_VOICED_MS`;
  or set `VAD_PROCESS_NOISY_SPANS=false` to stop transcribing noisy spans.
- **It ignores you when the TV is loud** — your speech is gating sparsely;
  lower `VAD_MIN_VOLUME` slightly, or speak a beat closer to the mic.
- **LLM quota errors** — Groq's free-tier quota is per model; the failover
  chain (`GROQ_LLM_MODELS`) moves on automatically. Check `/api/status` for
  the live model.
- **Safari plays no audio** — browsers block autoplay until a gesture; tap
  the orb once to unlock.
- **Bot cuts itself off (echo barge)** — raise `VAD_BARGE_MS`. If barge-in
  feels slow instead, lower it.

---

## Project layout

```
lava/
├── run.sh                  # one-shot launcher (recommended)
├── deploy/
│   ├── start_all.sh        # starts the three MCP servers only
│   └── stop_all.sh         # stops them
├── mcp_servers/            # asr_server, tts_server, tools_server, common
├── web/                    # server.py (bridge + VAD + orchestration), index.html (orb UI)
├── models/piper/           # Piper voice, auto-downloaded on first use
├── logs/                   # <name>.log (rotated), <name>.pid, notes.json
└── .env                    # all configuration
```
