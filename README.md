# Lava — full-duplex voice agent

Lava is a voice assistant you talk to from a browser tab. No wake word: tap
once to open the mic, then just talk. It answers out loud **while still
listening** — so you can interrupt it mid-sentence by simply speaking. When
you go quiet, it asks if you're still there before turning the mic off.

Everything runs on free services: speech recognition and the brain are on
Groq's free tier, the voice (Piper) is fully local. Developed on a laptop
(macOS), built for a Raspberry Pi. Python ≥ 3.12.

```
browser mic ──▶ Silero VAD ──▶ [ASR :8001] ──▶ Groq LLM (+ [tools :8003]) ──▶ [TTS :8002] ──▶ browser speaker
     ▲                                                                                          │
     └──────────────────────── you talk while it answers → it stops and listens (barge-in) ─────┘
```

**Contents:** [Features](#features) · [Quick start](#quick-start) ·
[Talking to it](#talking-to-it) · [Tools](#tools) ·
[How it works](#how-it-works) · [Configuration](#configuration) ·
[Spotify music](#spotify-music) · [Raspberry Pi](#raspberry-pi) ·
[Troubleshooting](#troubleshooting) · [Project layout](#project-layout) ·
[Security notes](#security-notes)

---

## Features

- **Full-duplex** — it listens while it talks. Interrupt it by speaking; it
  yields in well under a second and acts on what you say.
- **Smart about overlaps** — distinguishes a real interruption from a
  "mm-hmm" backchannel or you talking to someone else, and behaves correctly
  for each (see [Overlap handling](#overlap-handling-full-duplex-bench-v15)).
- **Real tools** — time, weather, web search, Wikipedia, timers, persistent
  notes, a light switch, Pi health checks, and full Spotify playback control.
- **Natural voice** — local Piper TTS streams the first clause before the
  sentence is finished, so replies start fast.
- **Robust** — auto-reconnects to its backends, walks an LLM failover chain
  on quota limits, and degrades TTS to a fallback instead of going silent.
- **Self-measuring** — live per-turn latency metrics (TTFA, ASR/LLM/TTS
  breakdown) in `/api/status`.

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

> **Want Spotify?** It needs two extra steps (a client id/secret and a one-time
> account link) — see [Spotify music](#spotify-music). Everything else works
> out of the box.

---

## Talking to it

- **Just talk.** No wake word. Ask for the time, weather, a web search,
  Wikipedia; set timers; save reminders; play Spotify music — it answers out
  loud in a sentence or two.
- **Interrupt it** by talking while it's answering: it stops immediately and
  listens to what you say next (barge-in).
- **Going quiet:** after ~50 s of silence it asks *"are you still there?"*
  (rotating wordings); after ~70 s of total silence the mic turns off. Tap the
  orb to resume — the conversation is remembered per browser tab for 24 h.
- **End the session** by tapping the orb while it's live. Tap again to start a
  new one.

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

## Tools

Twenty-seven tools are exposed to the LLM over the tools MCP server
(`:8003`). Twenty-five are things you can ask for by voice; two
(`pop_finished_timers`, `music_state`) are internal — the bridge polls them,
they aren't commands.

| Area | Tool | What it does | Try saying |
|---|---|---|---|
| **Time** | `get_time` | Current local date/time | "what time is it?" |
| **Weather** | `get_weather` | Forecast for a place (defaults to `DEFAULT_CITY`) | "will it rain in Manila?" |
| **Web** | `web_search` | DuckDuckGo news/web results | "search for …" |
| **Web** | `wikipedia` | First paragraph of an article | "who is …?" |
| **Timers** | `set_timer` | Countdown; announced when it fires | "set a 10 minute timer for pasta" |
| | `list_timers` | Show running timers + time left | "what timers are running?" |
| | `cancel_timer` | Cancel by label or id | "cancel the pasta timer" |
| **Notes** | `add_note` | Save a note/to-do (persists across restarts) | "remember to buy eggs" |
| | `list_notes` | Read back saved notes | "what's on my list?" |
| | `delete_note` | Remove notes matching a phrase | "delete the reminder about eggs" |
| | `clear_notes` | Delete all notes | "clear my notes" |
| **System** | `set_light` | Toggle the light (simulated — wire a relay for real) | "turn the light on" |
| | `system_status` | Pi CPU temp, load, memory, uptime, IP | "how's the Pi doing?" |
| **Music** | `music_search` | Find tracks/albums/artists/playlists | "search for lo-fi playlists" |
| | `music_play` | Play a track/album/playlist (optionally on a device) | "play Bohemian Rhapsody" |
| | `music_pause` / `music_resume` | Pause / continue | "pause" / "go ahead" |
| | `music_restart` | Restart the current track | "play it again" |
| | `music_skip` | Next / previous (with empty-queue fallback) | "next song" |
| | `music_volume` | Absolute, relative, "max", "mute", or read | "set the volume to 40" |
| | `music_shuffle` | Toggle shuffle | "shuffle on" |
| | `music_repeat` | Off / album / song | "repeat the song" |
| | `music_now_playing` | Current track + device | "what's playing?" |
| | `music_queue` | Add a track to the queue | "queue this song" |
| | `music_devices` | List available playback devices | "which devices do I have?" |
| *(internal)* | `pop_finished_timers` | Bridge polls for timers that fired | — |
| *(internal)* | `music_state` | Bridge polls Spotify playback state | — |

Music tool behavior in detail (including the empty-queue skip fallback and the
fast path that answers music commands in under a second) is covered in
[Spotify music](#spotify-music) and [Music mode](#music-mode).

---

## How it works

> **Skip ahead if you just want to use it.** Everything below explains the
> internals; none of it is needed to run or talk to Lava. Jump straight to
> [Configuration](#configuration) or [Spotify music](#spotify-music).

### Architecture

Four cooperating processes, connected over MCP (Model Context Protocol) and a
WebSocket:

| Process | Port | Role |
|---|---|---|
| `web.server` (bridge) | `8080` | The orchestrator. Runs VAD, owns the conversation, drives the LLM, coordinates barge-in, and streams audio to/from the browser over a WebSocket. |
| `mcp_servers.asr_server` | `8001` | Speech-to-text (Groq Whisper). |
| `mcp_servers.tts_server` | `8002` | Text-to-speech (local Piper, fully offline). |
| `mcp_servers.tools_server` | `8003` | All 27 tools, including Spotify (via `mcp_servers/spotify.py`). |

The bridge keeps a persistent connection to each MCP server, probes every 10 s,
and reconnects automatically if one dies.

### Audio transport

The browser captures the mic with echo cancellation and noise suppression on,
converts to 16-bit PCM inside an AudioWorklet thread, and streams 32 ms binary
frames over the WebSocket — no base64, no JSON per audio frame. Reply audio
streams back the same way: a 2-byte sample-rate header followed by 16-bit PCM
chunks, which the browser schedules gaplessly.

Wire protocol (binary audio, low overhead):

- **client → server:** raw 16-bit LE mono PCM at 16 kHz in 512-sample (32 ms)
  frames; JSON text frames carry control only (`{"type":"ready"}`).
- **server → client:** uint16 LE sample rate + 16-bit LE mono PCM, one TTS
  chunk per frame; JSON text frames carry control (`text`, `transcript`,
  `barge`, `tool`, `done`, `error`, `silence`, `bye`).

### Turn detection (VAD)

The bridge runs Silero VAD plus a loudness gate on every 32 ms frame. A "span"
opens on speech and closes after 0.2 s of silence; the span's audio (plus 120
ms of lead-in, so first words survive) goes to Whisper. Two extra guards keep
room noise from becoming turns: spans that are mostly non-speech are still
transcribed but Whisper's own quality gate drops degraded audio, and the LLM is
instructed to silently `SKIP` speech that isn't addressed to it (TV dialogue,
ads, background voices).

### Overlap handling (Full-Duplex-Bench v1.5)

200 ms (`VAD_BARGE_MS`) of detected speech over the bot starts *deferred
barge-in* (technique from Full-Duplex-Bench, arXiv:2507.23159): the agent does
not cut yet. The overlapping span is transcribed once it closes and classified
into the paper's four overlap scenarios — each with its desired behavior:

| Scenario | Example | Behavior |
|---|---|---|
| **User interruption** | "stop, play something else" | cut now, respond to the new request |
| **User backchannel** | "mm-hmm", "yeah", "got it" (≤ `BACKCHANNEL_MAX_WORDS` words) | keep talking exactly where it was |
| **Talking to other** | "Maria, are you coming?", "you guys, let's go" | hold the floor — no cut, no trace in the UI |
| **Background speech** | far-field TV / third-party talk | held by the VAD loudness gate; the music-mode lyric guard drops what survives |

The talk-to-other detector is deliberately conservative — it only fires on a
proper-noun name followed by a comma ("Maria, …"), a vocative + name ("hey
John, …"), or an explicit not-you cue ("you guys", "talk to you") — so a real
interruption is never swallowed (the agent's own name "Lava" is excluded). Set
`FD_HOLD_FLOOR=false` to disable it. A watchdog keeps responsiveness bounded:
a span that stays open longer than `BACKCHANNEL_CUT_MAX_SECS` is treated as
real speech and cuts immediately, so talking over the bot for more than a
second always wins the floor.

The bridge also measures the paper's two overlap latencies and reports them in
`/api/status` under `fd_metrics` (rolling windows): **t_stop** (barge onset →
bot audio cut) and **t_resp** (cut → first audio of the response turn), plus
per-scenario behavior counters.

### Response pipeline

Whisper transcribes → the LLM streams a reply, calling tools whenever they can
answer → sentences are split for speech and synthesized while the LLM keeps
streaming (a `Speaker` pipeline: sentence N+1 is synthesized while N plays) →
the first clause goes out before the full sentence exists (a comma after 28
chars is a split point). Fixed lines — greeting, apologies, idle check-ins —
are pre-synthesized at startup and replayed from cache in ~10 ms.

### Transcript correction

Groq's Whisper endpoint does not accept an `initial_prompt`, so the model
cannot be biased toward your vocabulary — proper nouns and unusual words get
snapped to phonetically-similar common words ("Clara Benin" → "Sarah Benin",
"benign", "Sony Brand"). To repair this, a fast LLM **proofread pass** runs
right after ASR and before the transcript is shown or handed to the main LLM:
it fixes only obvious recognition errors (names, places, brands, spelled-out
words) using the recent conversation as context. It never changes meaning,
returns the transcript unchanged when it already looks right, and falls back
to the raw text on any error or timeout — so it can only ever help. Short turns
(< `ASR_CORRECT_MIN_WORDS`) skip the pass entirely. By default it uses the
first non-reasoning model in the chain (a trivial task should not pay the
reasoning tax or burn the heavy primary's daily quota).

### Music mode

The usual playback device is a laptop speaker sitting next to the mic, so the
song the agent is playing flows right back in: lyrics become phantom voice
turns, and the music itself barge-cuts the bot's replies. While a `music_*`
tool is playing, the bridge therefore tightens its mic gates (louder volume
floor, higher voiced-ratio, bigger loudness-jump requirement — `MUSIC_MODE_*`
keys below), drops noisy spans whose transcript looks like the lyrics of the
current track, suppresses the "are you still there?" idle check-ins, keeps the
session warm, and skips the transcript-correction pass (it only adds latency
there). It also injects a per-turn note telling the main LLM the music is
playing through the speakers, so it does not repeat lyric-like fragments back.

While music plays, well-formed commands ("stop", "next", "previous", "louder",
"set the volume to 40", "play Bohemian Rhapsody"…) skip the LLM entirely and
act directly on the tools server — sub-second acknowledgment instead of a 2–4 s
LLM round trip. Anything the parser is not sure about falls through to the
normal LLM path, and volume numbers inside a music command ("turn it up by 40")
are relative, while "up to 40" is absolute.

### Idle handling

`idle_watch` tracks real voice, not raw frames. After `IDLE_PROMPT_SECS`
(50 s) of quiet it speaks a check-in ("Hi, are you still there?" — rotating
wordings); talking over the check-in barges it like any other speech. After
`IDLE_CHECK_SECS` (70 s) of total silence the mic turns off; the next orb tap
resumes the same conversation.

### Reliability

The bridge holds persistent connections to the three MCP servers, probes each
every 10 s with a plain TCP check, and reconnects automatically if one dies
and comes back (a bare HTTP POST once confused restarting servers into a 400
loop). A call that hits a mid-session drop is retried once on a fresh
connection. Every tool handler on the tools server settles its own Spotify and
network errors into spoken messages — an unhandled exception there used to kill
the whole server session, which is what reliably ate "stop the music". The
bridge polls Spotify's playing state each second while music runs (and once per
`MUSIC_POLL_IDLE_MAX_SECS` when idle) so its mic gates track reality rather
than the last command. LLM quota errors walk a model failover chain. Whisper audio first gets
its leading/trailing silence trimmed, which both detects the all-silence span
early (saved upload) and cuts ASR round-trip size for breathy padding.

---

## Configuration

All configuration lives in `.env`. Defaults are sensible — the keys you're
most likely to touch first:

| Key | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Free key from https://console.groq.com/keys |
| `GROQ_LLM_MODEL` | `openai/gpt-oss-20b` | Primary LLM |
| `VAD_MIN_VOLUME` | `0.7` | Loudness floor — raise if TV/room noise triggers turns |
| `VAD_BARGE_MS` | `200` | Speech burst (ms) that cuts the bot mid-reply (barge-in speed vs TV false cuts) |
| `IDLE_PROMPT_SECS` / `IDLE_CHECK_SECS` | `50` / `70` | Silence before the spoken "still there?" check-in / before the mic turns off |
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
| `VAD_BARGE_MS` | `200` | Speech burst (ms) that cuts the bot mid-reply |
| `VAD_BARGE_ORIGIN_MIN_VOICED_MS` | `150` | Minimum voiced length before a barge-in is considered at all |
| `BACKCHANNEL_CUT_MAX_SECS` | `1.0` | Watchdog: defer the barge-in cut at most this long |
| `BACKCHANNEL_MAX_WORDS` | `4` | Longest transcript still classifiable as a backchannel |
| `FD_HOLD_FLOOR` | `true` | Hold the floor on talk-to-other overlap ("Maria, …", "you guys, …") — see Overlap handling |
| `VAD_PRE_ROLL_MS` | `120` | Lead-in audio included before first voiced frame (first-word capture) |
| `VAD_ATTACK` / `VAD_RELEASE` | `0.2` / `0.2` | Loudness smoothing (keep symmetric in noisy rooms) |
| `VAD_PROCESS_NOISY_SPANS` | `true` | Transcribe spans the ratio gate would discard — recovers commands spoken over TV |
| `FAST_MUSIC_COMMANDS` | `true` | While music plays, fielded commands bypass the LLM entirely |
| `MUSIC_MODE_GRACE_SECS` | `20` | After a music command, run tightened mic gates for at least this long |
| `MUSIC_MODE_MIN_RATIO` | `0.45` | Voiced-ratio floor while music plays (vs `VAD_MIN_VOICED_RATIO`) |
| `MUSIC_MODE_VOLUME_DELTA` | `0.04` | Extra loudness-jump needed before music counts as speech |
| `MUSIC_POLL_IDLE_MAX_SECS` | `300` | When nothing plays, poll the Spotify state at most this often |

**ASR** (`mcp_servers/asr_server.py`)

| Key | Default | Meaning |
|---|---|---|
| `ASR_BACKEND` | `groq` | Speech-to-text engine |
| `GROQ_ASR_MODEL` | `whisper-large-v3-turbo` | Groq Whisper model |
| `ASR_LANGUAGE` | `en` | ISO-639-1 hint (`en`, `tl`, …) |
| `ASR_NO_SPEECH_MAX` / `ASR_LOGPROB_MIN` | `0.8` / `-1.3` | Quality gate: drop segments Whisper itself flags as unintelligible |

**Transcript correction** (`web/server.py`, post-ASR LLM proofread)

| Key | Default | Meaning |
|---|---|---|
| `ASR_CORRECT` | `true` | Enable the context-aware mis-hearing repair pass |
| `ASR_CORRECT_MODEL` | first non-reasoning model in chain | Model that does the proofread (override to force one) |
| `ASR_CORRECT_MIN_WORDS` | `3` | Skip the pass for turns shorter than this (no API call) |
| `ASR_CORRECT_CTX` | `6` | Recent messages of context handed to the corrector |
| `ASR_CORRECT_TIMEOUT` | `2.5 s` | Hard cap; on timeout the raw transcript is used |
| `ASR_CORRECT_GRACE_SECS` | `0.9` | Grace window before a throttled corrector falls back to the raw text |

**TTS** (`mcp_servers/tts_server.py`)

| Key | Default | Meaning |
|---|---|---|
| `PIPER_VOICE` | `en_US-hfc_female-medium` | Voice id (auto-downloads to `models/piper/`) |
| `PIPER_LENGTH_SCALE` | `1.0` | `< 1` speaks faster (try `0.9` on a Pi) |

**Tools & Spotify** (`mcp_servers/tools_server.py`, `mcp_servers/spotify.py`)

| Key | Default | Meaning |
|---|---|---|
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | — | From a Spotify app (developer.spotify.com). Required by every music tool; search additionally works before the one-time account link |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8080/callback` | Must match the app's redirect URI exactly |
| `MUSIC_VOLUME_STEP` | `10` | Steps for direction-only volume words ("louder" = +10, "quieter" = −10) |

**Idle & infrastructure**

| Key | Default | Meaning |
|---|---|---|
| `IDLE_PROMPT_SECS` / `IDLE_CHECK_SECS` | `50` / `70` | Check-in after this much quiet / mic off after this much total silence |
| `DEFAULT_CITY` / `DEFAULT_TIMEZONE` | `Quezon City` / `Asia/Manila` | Time, weather, and identity |
| `MCP_HOST` | `127.0.0.1` | Bind host for the three MCP servers |
| `ASR_PORT` / `TTS_PORT` / `TOOLS_PORT` | `8001` / `8002` / `8003` | MCP server ports |
| `WEB_PORT` | `8080` | Bridge port |
| `MODELS_DIR` | `./models` | Where the Piper voice is stored |

---

## Spotify music

Twelve `music_*` tools: search, play, pause, resume, restart, skip, queue,
volume, shuffle, repeat, now-playing and device listing (plus an internal
`music_state` mirror the bridge polls — not a command).

- **Play** — "play Bohemian Rhapsody", "play the album X", "play the playlist
  Y". The reply names the track *and the device it landed on*, and warns you
  when that device is a browser tab (the web player plays inside a tab — bring
  it forward and check its mute button).
- **Stop / pause / resume** — "stop the music", "pause", "go ahead".
- **Skip** — "next song" / "previous" / "last one". Spotify silently ignores
  next/prev when the queue is empty (always, right after a single track), so
  the tool verifies the player actually changed and, if not, plays the next
  song by the same artist. "Previous" with an empty queue restarts the song.
- **Restart** — "play it again", "restart the song", "start over", "one more
  time", "rewind".
- **Volume** — "set the volume to 40", "turn it up to 40" (absolute), "turn it
  up 10" / "down 5" (relative), "louder"/"quieter" (± `MUSIC_VOLUME_STEP`, 10
  by default), "max", "mute"; "what's the volume?" reports the current level.
  The web player often refuses remote volume — the tool says so instead of
  pretending.
- **Shuffle / repeat** — "shuffle on", "repeat the song", "repeat the album";
  just "repeat" cycles off → album → song.
- **Now playing / queue / devices** — "what's playing?", "queue this song",
  "which devices do I have?".

While music is playing these commands are answered in well under a second by
the browser's fast path — they never reach the LLM (see *Music mode* above);
mic gates and lyric-reads keep the song itself from creating fake turns.

### Setup

1. Create an app at https://developer.spotify.com/dashboard (no approval
   needed), add `http://127.0.0.1:8080/callback` as a redirect URI, and put the
   client id/secret in `.env`.
2. Link your account once: open `http://127.0.0.1:8080/spotify/auth` in a
   browser, approve, and the refresh token is stored in
   `logs/spotify_token.json` (auto-refreshed thereafter).

Search and catalog lookups run on the app's client-credentials token, so they
work even before linking. Playback controls need the linked account **and a
Spotify Premium subscription** — free accounts get a clear spoken explanation
instead. If no device is active, Lava auto-transfers to a lone device, or asks
which of several to use.

---

## Raspberry Pi

- Copy the project folder **without** `.venv/` (virtualenvs don't survive a
  copy); `./run.sh` rebuilds it on the Pi.
- **Piper is the bottleneck, not the VAD.** `-medium` voices have RTF ≈ 0.5–1
  on a Pi 4. The pipelined Speaker hides most of it behind playback, but for
  snappier first audio set `PIPER_LENGTH_SCALE=0.9`, or switch `PIPER_VOICE` to
  a `-low` quality model.
- The browser does the int16 conversion and batches to 32 ms frames inside the
  AudioWorklet thread, so the Pi's main thread only forwards ArrayBuffers.
- Watch `ttfa_ms` in the turn stats (also in the logs) — on a Pi expect
  ~1.2–2 s with a tool call, ~0.9–1.4 s without.
- **Spotify on the Pi** — the account link is per machine: on the Pi, open
  `http://127.0.0.1:8080/spotify/auth` in a browser on the Pi itself (or over
  an `ssh -L 8080:localhost:8080` tunnel) so the callback lands on the Pi; the
  client id/secret in `.env` stay the same.

---

## Troubleshooting

- **`Address already in use` on :8080** — a stale bridge holds the port.
  `./run.sh` clears it automatically; otherwise `lsof -iTCP:8080 -sTCP:LISTEN`
  (or `pkill -f "web.server"`), then relaunch.
- **The agent stopped responding entirely** — an MCP server probably died. The
  bridge reconnects on its own once the backend is back; bring backends up with
  `./deploy/start_all.sh` (or just re-run `./run.sh`). Check `logs/<name>.log`
  for the culprit.
- **It answers TV dialogue / room chatter** — raise `VAD_MIN_VOLUME` first
  (soft voices start getting eaten above ~0.76), then `VAD_MIN_VOICED_MS`; or
  set `VAD_PROCESS_NOISY_SPANS=false` to stop transcribing noisy spans.
- **It ignores you when the TV is loud** — your speech is gating sparsely;
  lower `VAD_MIN_VOLUME` slightly, or speak a beat closer to the mic.
- **LLM quota errors** — Groq's free-tier quota is per model; the failover
  chain (`GROQ_LLM_MODELS`) moves on automatically. Check `/api/status` for the
  live model.
- **Safari plays no audio** — browsers block autoplay until a gesture; tap the
  orb once to unlock.
- **Music plays but you hear nothing** — it is playing on a device you didn't
  mean: the Spotify web player (a background browser tab, possibly muted) or
  another phone/laptop. Ask "which device is playing?", then re-play naming the
  right one: "play X on my Mac".
- **"Spotify is not linked" / "authorization expired"** — re-open
  `http://127.0.0.1:8080/spotify/auth` and approve again; the new token
  replaces the old one in `logs/spotify_token.json`.
- **"Premium is required"** — playback control needs a Premium account; search
  and "what's playing" still work on a free one.
- **Bot cuts itself off (echo barge)** — raise `VAD_BARGE_MS`. If barge-in
  feels slow instead, lower it.

---

## Project layout

```
lava/
├── run.sh                  # one-shot launcher (recommended)
├── requirements.txt        # Python dependencies
├── .env                    # all configuration (git-ignored — holds API keys)
├── deploy/
│   ├── start_all.sh        # starts the three MCP servers only
│   └── stop_all.sh         # stops them
├── mcp_servers/            # asr_server, tts_server, tools_server, spotify, common
├── web/                    # server.py (bridge + VAD + orchestration), index.html (orb UI)
├── models/piper/           # Piper voice, auto-downloaded on first use
└── logs/                   # <name>.log (rotated), <name>.pid, notes.json, spotify_token.json
```

HTTP/WebSocket endpoints served by the bridge (`:8080`):

| Endpoint | Purpose |
|---|---|
| `GET /` | The orb UI (`index.html`) |
| `GET /ws` | Audio + control WebSocket (the conversation) |
| `GET /api/status` | Live model, loaded tools, city, session count, `fd_metrics` |
| `GET /spotify/auth` | One-time Spotify account link (redirects to Spotify) |
| `GET /callback` | Spotify OAuth callback (stores the refresh token) |
| `GET /favicon.png` | Icon |

---

## Security notes

- **`.env` holds your secrets** (Groq key, Spotify client secret) and is
  git-ignored — never commit it.
- **`logs/spotify_token.json`** holds a Spotify refresh token (a real
  credential — whoever has it can act as your account) and is git-ignored
  (`logs/*`, with only `notes.json` re-included). Treat the file like a
  password.
- **`logs/notes.json`** is the one tracked log file (your saved notes).
- Everything binds to `127.0.0.1` by default; nothing is exposed to the
  network unless you change `MCP_HOST` / the ports.
- `set_light` is a **simulated** toggle — wire a relay to real GPIO before
  trusting it for anything physical.
