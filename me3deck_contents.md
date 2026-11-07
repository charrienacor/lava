# ME3 Deck Contents — Full-Duplex Voice Agent on a Raspberry Pi
*For `me3deck.pdf` (2-page template, sections 1–6 in priority order). Gray text = fill-in placeholder. All facts pulled from the lava repo, 2026-11-07.*

---

## 1 · SYSTEM DIAGRAM

**Title:** Lava — full-duplex voice agent on a Raspberry Pi (browser UI, headless Pi)

```
                        ┌─────────────────────────── ON THE RPI ───────────────────────────┐
                        │                                                                   │
 browser mic (16 kHz)   │  web.server "bridge" :8080          ← hand-rolled harness         │
 ─────────────────────▶ │  ├─ Silero VAD (in-process)         (aiohttp + WebSocket, ~4k LOC)│
 raw PCM over WebSocket │  ├─ turn engine: ASR → LLM → TTS    no LangChain/pipecat in prod  │
                        │  └─ barge-in: speech onset → cut    MCP client keeps 3 servers up │
                        │         TTS + LLM stream (~200 ms)                                │
                        │        │            │                │                            │
                        │        ▼            ▼                ▼                            │
                        │  asr-mcp :8001   tools-mcp :8003   tts-mcp :8002                  │
                        │  transcribe()    29 tools          synthesize()                   │
                        └───────┬──────────────┬──────────────────┬─────────────────────────┘
                                │              │                  │
                        ┌───────▼───────┐ ┌────▼─────────────┐ ┌──▼──────────────────┐
 CLOUD (Groq API)       │ whisper-large-│ │ openai/gpt-oss-  │ │ CLOUD (free APIs)   │
                        │ v3-turbo ASR  │ │ 120b LLM         │ │ web search, wiki,   │
                        └───────────────┘ │ + 20b / qwen     │ │ weather, Telegram,  │
                                          │   fallback chain │ │ Spotify, HA hub     │
                                          └──────────────────┘ └─────────────────────┘
                        ▲                                                         │
 browser speaker ◀──────┴──── TTS audio (Piper en_US-hfc_female-medium, ON RPI) ◀─┘
        you talk while it answers → it stops and listens (barge-in)
```

**Boundary rule:** on-RPi = VAD, all three MCP servers, Piper TTS, the harness, virtual Home Assistant hub (:8123). Cloud = Groq (Whisper ASR + LLM) and the free web/Telegram/Spotify APIs the tools server calls.

**Key bullets (paste under diagram):**
- Audio path: mic → VAD (on RPi) → ASR MCP → LLM (+tools MCP) → TTS MCP → speaker
- Harness with MCP client: **custom `web.server` bridge** (hand-rolled, not pipecat/crewAI)
- LLM is a commercial API (Groq); tools via our own MCP servers
- All MCP servers built by us and running on the RPi (ports 8001/8002/8003)
- Barge-in cancels TTS playback **and** the LLM stream, in ~200 ms

---

## 2 · PROMPT

**Version:** `SYSTEM_PROMPT` v2 — lives in the repo at `web/core/config.py` (line 97), versioned in git. Excerpt:

> You are Lava, a friendly full-duplex voice assistant living in Quezon City (timezone Asia/Manila). You are spoken to by voice and your replies are read aloud, so keep answers to at most two short, natural sentences. Never use markdown, bullet points, emojis, or stage directions — only words that sound good out loud. Use the available tools whenever they can answer the question (time, weather, search, wikipedia, timers and reminders, saved notes, music playback on Spotify, smart-home lights and thermostat via Home Assistant, calls and messages on Telegram…). Tool results are ground truth for dynamic facts, so quote them exactly. If the user doubts an answer, call the tool again and stand by what it returns — never invent a different or empty result just to agree with them. If the user interrupts or changes the subject, follow the new subject. The mic is in a room with a TV and phones, so some turns are their speech picked up by accident… Reply to those with the single word SKIP and nothing else — it is never spoken aloud.

*(Full prompt is ~250 words; the slide shows this excerpt. The SKIP protocol and "stand by the tool" rule are the two lines worth pointing at.)*

---

## 3 · CONTEXT ENGINEERING

**GOES IN** (per turn, assembled in `web/core/turn.py` + `llm.py`):
- System prompt + persona (stable prefix, cached by Groq for TTFT)
- Rolling ASR finals — last **7 turns** of history (`HISTORY_TURNS = 7`)
- VAD turn events (utterance start/end, barge-in onsets)
- MCP tool results (injected verbatim; prompt says to quote them exactly)
- Memory: persistent notes/reminders store (`logs/notes.json`) — the one RAG-ish layer

**MANAGED:**
- Token budget: history capped at 7 turns; older turns dropped, never summarized
- Stable prefix caching: system prompt + tool schemas constant → Groq prefix cache hits
- State machine: `idle → listen → think → speak` (idle prompts at 50 s, mic off at 70 s)
- On barge-in: drop stale TTS audio **and** cancel the in-flight LLM stream (hard 45 s turn deadline as backstop)
- Log context size per run: fd-benchmark marks attach transcripts + context length to every turn (`FdbenchObserver`)

---

## 4 · MCP SERVICES + MODEL SERVING (one table, one truth)

| Component | MCP server / transport | Tools | Model | Served where / notes |
|---|---|---|---|---|
| ASR | `asr-mcp` / HTTP (streamable) :8001 | `transcribe(wav, language)`, `asr_info` | Groq `whisper-large-v3-turbo` | RPi server; final on VAD end; no-speech gate (logprob ≥ −1.3) |
| TTS | `tts-mcp` / HTTP (streamable) :8002 | `synthesize(text, voice)`, `tts_info` | **Piper** `en_US-hfc_female-medium` | **On-RPi, on-device** (aarch64); streamed chunks; cancel on barge-in |
| LLM | (harness MCP client) | function-calling, 29 tools | Groq `openai/gpt-oss-120b` (fallback: `gpt-oss-20b`, `qwen3.8-27b`) | Commercial API; streaming tokens; fallback chain = per-model daily quota insurance |
| VAD | in-process (bridge) | `detect_speech(frames)` | **Silero VAD** | RPi on-device, 16 kHz mono; confidence 0.80, 200 ms barge gate |
| Task | `tools-mcp` / HTTP (streamable) :8003 | `get_time`, `get_weather`, `web_search`, `wikipedia`, `set_timer`/`list`/`cancel`, `add_note`/`list`/`delete`/`clear`, `set_lights`, `set_thermostat`, `get_temperature`, `call_contact`, `send_message`, `music_*` (Spotify), `system_status` — 29 total | n/a (free web APIs + Telegram Bot API + Spotify API + HA REST) | RPi server; HA hub is a **virtual** Home Assistant on :8123 (Docker) |

**Serving notes for the slide:** one model runs on-device (Piper TTS, ~50 MB); ASR + LLM are Groq cloud (free tier); the Pi stays cool and the 4 GB RAM budget holds (~0.5 GB total for all four processes).

---

## 5 · EVALUATION & BENCHMARK

**Standardized latency method (classwork):**
- One clock: assistant machine monotonic; every stage timestamped (`fdbench/marks.py`, `FdbenchObserver` on the pipeline)
- VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker
- E2E = speaker − VAD end; barge-in = stop − speech start
- Full-Duplex-Bench v1.0 + v1.5 clips (dataset in `fd-benchmark/datasets/`: 280 clips — 30 each of pause-synth, pause-candor, turn-taking, interruption, backchannel, talking-to-other, background + 40 false-turn-taking); report mean + p95; JSON log per run

**LAPTOP RUN (this is the pre-Pi validation run — Pi run pending):**
- Run: `fd-benchmark/runs/charrie/2026-11-07_11-10-41/` · 2026-11-07 · 40-clip quick round (1 per behavior group + 30 stage-latency phrases) · 15 min
- Machine: MacBook (Darwin 25.5.0, 16 GB) — **not the Pi**; CPU median 8.9% / p95 22.9% / max 63%, min free RAM 4.0 GB, no swap, no throttling (905 samples)
- Models: Groq whisper-large-v3-turbo (ASR), openai/gpt-oss-120b + 20b fallback (LLM), Piper en_US-hfc_female-medium on-device (TTS), Silero VAD v5

| Metric | Definition | Target | Mean | p95 | n | Verdict |
|---|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.0 s | **3.97 s** | 5.55 s | 24 | mean ✓ / p95 ✗ |
| ASR final | end of speech → transcript | < 0.8 s | 2.01 s | 3.16 s | 30 | ✗ — target assumes local ASR; cloud Whisper adds ~2 s network |
| LLM TTFT | transcript → first token | < 1.5 s | 1.90 s | 2.40 s | 24 | ✗ — Groq queue; 20b fallback hits add ~6 s outliers |
| TTS TTFB | text → first audio chunk | < 0.6 s | **0.09 s** | 0.16 s | 24 | ✓ — Piper on-device |
| Barge-in | speech → playback stopped | < 0.8 s | 1.53 s | 1.53 s | 1 | ✗ (n=1, single v1_interrupt sample) |
| WER | ASR accuracy on test set | < 15 % | **8.6 %** | — | 30 | ✓ |

**Behavior (turn-taking) results, 40-clip round:**
- Interruption (v1): ✓ yielded in 1.68 s (stop 1.53 s)
- Backchannel (v1.5): ✓ kept talking (through 1/1)
- Pause: ✗ took a turn during a synthetic pause (TOR 1) · ✓ stayed silent on candor pause
- Turn-taking (v1): ✗ missed — bot answered a misheard partial ("No.") early and consumed the turn, then never answered the real question
- False turn-taking: 1/2 — ✗ side-talk triggered a turn (spoke 3.8 s after +4.2 s) · ✓ TV/radio ignored
- 5 no-reply trials, root-caused from `assistant.log`: (a) gpt-oss-120b rate-limited twice mid-run → 20b fallback with ~6 s TTFT and one wrong tool call (cross-trial context); (b) premature answer to a partial ASR fragment

*Artifacts: `summary.md`, `results.json` → `deck_summary`, `stage_latency.json`, `latency_chart.svg`, per-clip `trials/<id>/{stages.jsonl,timing.json,pi_stats.jsonl}` — all in the run dir above. Pi run will overwrite this section with the on-Pi numbers.*

**Demo (deck 2) script — 2–3 min, real task, natural turns, no push-to-talk:**
1. "What time is it?" — plain LLM turn
2. "Turn the living room lights on, 80%" — live HA tool call (show the light state in the response)
3. Talk over Lava mid-answer — barge-in, it yields and follows the new subject
4. "Set a 3-minute timer" then interrupt it — tool + recovery
5. Kill/throttle the Groq path (or let a call run long) — show the fallback model pick up and the 45 s deadline keeping it bounded

**Results state (deck 2):** Laptop validation run 2026-11-07 (MacBook, 16 GB; CPU p95 22.9%, no throttle); Pi run pending — re-run the same 40-clip round on the Pi and swap in its numbers. Models: gpt-oss-120b (20b fallback) / whisper-large-v3-turbo / Piper hfc_female-medium / Silero VAD.

---

## CHALLENGES & REFERENCES

**Challenges** (one per row, each with fix + proof):

| Challenge | Fix | Measurement proving it worked |
|---|---|---|
| Groq free-tier quota is **per model** — primary LLM drains mid-run | Fallback chain `120b → 20b → qwen3.8-27b`, tried in order | Bench run: 120b rate-limited twice (11:24:35, 11:25:41), 20b picked up both times — no dead turns from quota |
| Cold start broke: `.env` silently not loaded (wrong `Path.parents` depth) → no API key, benchmark assistant dead on spawn | `ROOT = three parents up` in `config.py` | Fresh `launch_assistant.py` process boots with all keys; preflight passes |
| Music/TV in the room: Whisper transcribes lyrics, agent "answers" a song | Music mode: tightened VAD gates + grace window; SKIP protocol in the prompt for non-user speech | FDB bench: TV/radio clip ✓ ignored (false-take 0/1); residual: side-talk clip ✗ took a turn (1/2 overall) |
| HA hub dies silently after a restart → all lights/thermostat commands fail | `start_all.sh` now starts the hub too (idempotent); `stop_all.sh` kills it via `ha.pid` | `set_lights`/`set_thermostat`/`get_temperature` verified end-to-end after a full stop/start |
| Telegram bots can't initiate calls | "Call" sends a call-back request message; background `getUpdates` poller auto-registers contacts | Contact learned from first bot message; `send_message` delivered (verified live) |
| Barge-in must beat the bot's own voice + TV audio | Overlap classifier (vocative/name cues, conservative) + watchdog: any overlap > `BACKCHANNEL_CUT_MAX_SECS` cuts immediately | Bench: v1_interrupt ✓ yielded (stop 1.53 s, back 1.68 s); v1.5 backchannel ✓ kept talking |
| Bot answers a misheard partial and consumes the turn → real question goes unanswered | Known: ASR fires on a short confident fragment ("No.") before the full turn ends | Bench: 1/40 turn-taking miss traced to exactly this in `stages.jsonl` (partial ASR at tt=4.53 s, real turn ended 14.6 s) |

**References:**
- Model Context Protocol — modelcontextprotocol.io
- Full-Duplex-Bench (Lin et al.) v1.0/v1.5 — `fd-benchmark/` in repo; method: `fd-benchmark/docs/METHOD.md`, stages: `STAGES.md`
- Groq docs (Whisper, gpt-oss), Silero VAD, Piper TTS
- Repo: lava (this project) — `web/`, `mcp_servers/`, `ha/`, `fd-benchmark/`; run logs in `logs/*.log`

---

### Fill-in checklist before printing
- [x] Section 5 table: laptop run 2026-11-07 filled (40-clip round, `runs/charrie/2026-11-07_11-10-41/`)
- [x] Behavior results + root-caused no-reply trials documented
- [ ] **Pi run**: copy lava to Pi, re-run the same 40-clip round (`fdbench run --where pi ...`), swap its numbers into Section 5 (machine line, resources, table)
- [ ] Paste `latency_chart.svg` (from run dir) next to the table
- [ ] Confirm Pi model/RAM/cooling on the actual Pi before the Pi run
- [ ] Prompt excerpt: verify line numbers still match after any `config.py` edit
