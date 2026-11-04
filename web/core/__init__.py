"""Core modules of the Lava web bridge.

config      environment reads and system-wide tunables (leaf)
state       shared runtime state (sessions, history, speech cache, FD metrics)
mcp_client  persistent MCP connections + tool execution
llm         Groq streaming LLM, model fallback chain, ASR proofread
vad         Silero VAD + Full-Duplex-Bench overlap classification
speech      TTS cache, sentence splitting, pipelined Speaker
music       Spotify music-command fast path
turn        utterance flow (ASR -> LLM -> TTS) and barge-in handling
"""
