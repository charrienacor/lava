"""Configuration: every environment read and system-wide tunable for the bridge.

Leaf module: it imports nothing from the rest of the package, so every other
module can pull its constants from here without risk of a cycle.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")
# Fallback chain, tried in order: Groq's daily quota is per model, so the
# next model still has a fresh quota when the primary is drained.
_chain = [m.strip() for m in os.getenv("GROQ_LLM_MODELS", "").split(",") if m.strip()]
if not _chain:
    _chain = [GROQ_LLM_MODEL]
if GROQ_LLM_MODEL not in _chain:
    _chain.insert(0, GROQ_LLM_MODEL)
LLM_CHAIN = _chain
TURN_DEADLINE_SECS = float(os.getenv("TURN_DEADLINE_SECS", "45"))
# Idle handling: after IDLE_PROMPT_SECS of silence the agent asks if you are
# still there (a different wording each time); IDLE_CHECK_SECS of total
# silence stops the mic - the next tap resumes the same conversation.
IDLE_CHECK_SECS = float(os.getenv("IDLE_CHECK_SECS", "70"))
IDLE_PROMPT_SECS = min(
    float(os.getenv("IDLE_PROMPT_SECS", "50")), IDLE_CHECK_SECS - 5.0
)
CITY = os.getenv("DEFAULT_CITY", "Quezon City")
TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Manila")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080/callback"
)
ASR_URL = os.getenv("ASR_MCP_URL", "http://127.0.0.1:8001/mcp")
TTS_URL = os.getenv("TTS_MCP_URL", "http://127.0.0.1:8002/mcp")
TOOLS_URL = os.getenv("TOOLS_MCP_URL", "http://127.0.0.1:8003/mcp")


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
# PCM bytes per websocket message: small enough that barge-in lands ~150 ms
# after the audio the user hears; smaller adds client scheduling overhead.
CHUNK_BYTES = 4800
HISTORY_TURNS = 7


SYSTEM_PROMPT = (
    f"You are Lava, a friendly full-duplex voice assistant living in {CITY} (timezone {TZ}). "
    "You are spoken to by voice and your replies are read aloud, so keep answers to at most "
    "two short, natural sentences. Never use markdown, bullet points, emojis, or stage "
    "directions - only words that sound good out loud. Use the available tools whenever they "
    "can answer the question (time, weather, search, wikipedia, timers and reminders, saved "
    "notes, music playback on Spotify, system status); reminders and notes are the same store - the notes "
    "tools. Tool results are ground truth for dynamic facts, so quote them exactly. If the "
    "user doubts an answer, call the tool again and stand by what it returns - never invent "
    "a different or empty result just to agree with them. If a tool returns nothing, say so "
    "honestly. Treat ambiguous or emotional remarks like 'I miss it', 'that one' or 'are "
    "you sure' as conversation, not commands: before adding or deleting anything, confirm "
    "what they want, and if a request could match several notes or timers, ask which one "
    "instead of guessing. After a tool result, speak the answer briefly and warmly. "
    "Sometimes the transcript contains a word the user spelled out letter by "
    "letter (it arrives already converted to plain text, e.g. 'Search for "
    "LAVA'); treat that spelling as exact - use it verbatim in searches, "
    "playback requests and answers, never 'correct' it to a similar-looking "
    "word. "
    "If the user interrupts or changes the subject, follow the new subject. "
    "The mic is in a room with a TV and phones, so some turns are their speech "
    "picked up by accident: fragments of show dialogue, ads, narration, or "
    "anyone who is clearly not talking to you. Reply to those with the single "
    "word SKIP and nothing else - it is never spoken aloud. When unsure, treat "
    "a plausible direct question or request as real and answer it. The same is "
    "true of singing: short song-title or artist fragments that arrive while "
    "music plays are usually the music, not the user."
)


# --- music mode --------------------------------------------------------------
# Spotify playing out loud (usually the web player tab, so the same laptop's
# speakers feed the mic) is the ghost in this agent: vocals survive the VAD,
# Whisper transcribes lyrics ("I declare a Benin song"), and innocent
# fragments have triggered plays. While music plays we:
#   1. tighten the VAD gates (MUSIC_MODE_* below),
#   2. handle control words (stop/pause/next/louder/volume 40/...) directly,
#      no LLM — sub-second and immune to hallucinated tool calls,
#   3. tell the LLM what the music is so lyric fragments get SKIPped.
MUSIC_MODE_GRACE = float(os.getenv("MUSIC_MODE_GRACE_SECS", "20"))
MUSIC_MODE_MIN_RATIO = float(os.getenv("MUSIC_MODE_MIN_RATIO", "0.45"))
MUSIC_MODE_VOLUME_DELTA = float(os.getenv("MUSIC_MODE_VOLUME_DELTA", "0.04"))
# How long the bridge keeps polling music_state after music stopped (Sees an
# external pause on your phone, e.g.) before dropping back to lazy checks.
MUSIC_POLL_IDLE_MAX = float(os.getenv("MUSIC_POLL_IDLE_MAX_SECS", "300"))
FAST_MUSIC_COMMANDS = os.getenv("FAST_MUSIC_COMMANDS", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# Fixed line fired once the mic is live (after the permission dialog).
# Pre-synthesized at startup, so it plays with ~10 ms to first audio.
# Deliberately not LLM-generated. Plays as the turn task, so barge-in cuts it
# like any other speech.
_GREETING = "Hi there! I'm Lava. What can I do today?"
# Spoken (rotating) when the session has gone quiet, before the mic turns off.
_IDLE_PROMPTS = [
    "Hi, are you still there?",
    "Still with me?",
    "You still there?",
    "Hello? Are you there?",
]
ASR_APOLOGY = "Sorry, my ears glitched. Could you say that again?"
LLM_APOLOGY = "Sorry, my brain hiccuped for a moment. Could you say that again?"
