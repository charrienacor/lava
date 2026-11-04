"""Regression test: every utterance seen in logs/web.log 16:23-17:15,
plus lyric traps. Run: .venv/bin/python tests/test_music_parser.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web.core.music import parse_music_command

# (utterance, expected tool or None, expected args-subset, note)
CASES = [
    # --- the leaks caught in today's log (must NOT be what they were) ---
    ("Wake me up, wake me up.", None, None, "LYRIC - was volume up"),
    ("Up inside.", None, None, "LYRIC - was volume up"),
    ("Can you play next song?", "music_skip", {"direction": "next"}, "was music_play 'next song'"),
    ("Okay, play the next song.", "music_skip", {"direction": "next"}, "was music_play 'the next song'"),
    ("you play one click straight,", "music_play", {"query": "one click straight"}, "real command - keep"),
    # --- next-family phrasings (were LLM-dependent) ---
    ("you move to the next song.", "music_skip", {"direction": "next"}, "was LLM"),
    ("Move to the next.", "music_skip", {"direction": "next"}, ""),
    ("Go to the next track.", "music_skip", {"direction": "next"}, ""),
    ("Switch to the next song.", "music_skip", {"direction": "next"}, ""),
    ("Change the music.", None, None, "ambiguous - LLM handles (skip next)"),
    ("Next song.", "music_skip", {"direction": "next"}, "keep"),
    ("next", "music_skip", {"direction": "next"}, "keep"),
    ("Skip.", "music_skip", {"direction": "next"}, "keep"),
    ("Next Tuesday.", None, None, "not music - keep rejecting"),
    # --- volume: real commands must survive ---
    ("Turn it up.", "music_volume", {"level": "up"}, ""),
    ("Turn it down.", "music_volume", {"level": "down"}, ""),
    ("Louder.", "music_volume", {"level": "up"}, ""),
    ("Make it louder.", "music_volume", {"level": "up"}, ""),
    ("Quieter please.", "music_volume", {"level": "down"}, ""),
    ("Softer.", "music_volume", {"level": "down"}, ""),
    ("Crank it.", "music_volume", {"level": "up"}, ""),
    ("Blare it.", "music_volume", {"level": "up"}, ""),
    ("Raise the volume.", "music_volume", {"level": "up"}, ""),
    ("Bring the volume down.", "music_volume", {"level": "down"}, ""),
    ("Up the music.", "music_volume", {"level": "up"}, "log case - keep"),
    ("the volume to 70.", "music_volume", {"level": "70"}, "log case - keep"),
    ("Volume 40.", "music_volume", {"level": "40"}, ""),
    ("Turn it up to 40.", "music_volume", {"level": "40"}, ""),
    ("Turn it down to 20.", "music_volume", {"level": "20"}, ""),
    ("Max volume.", "music_volume", {"level": "max"}, ""),
    ("Full volume.", "music_volume", {"level": "max"}, ""),
    ("Mute.", "music_volume", {"level": "mute"}, ""),
    ("What's the volume?", "music_volume", {"level": ""}, "report"),
    ("Set the volume to 55.", "music_volume", {"level": "55"}, ""),
    # --- volume: lyric traps (must NOT trigger) ---
    ("I'm up for it.", None, None, "lyric-ish"),
    ("Don't stop me now.", None, None, "lyric - no volume words"),
    ("Take me down to the river.", None, None, "'take...down' without volume object"),
    ("Counting stars, shining bright.", None, None, "lyric"),
    ("We're going up tonight.", None, None, "bare 'up' no volume context"),
    ("Down by the water.", None, None, "bare 'down' no volume context"),
    ("Up all night.", None, None, "bare 'up'"),
    ("Take me higher.", None, None, "LYRIC - 'higher' without volume idiom"),
    ("Set it up.", None, None, "'set' not a volume verb"),
    ("Raise your glass.", None, None, "'raise' without volume object"),
    ("A hundred thousand miles.", None, None, "number without volume context"),
    ("Three little birds.", None, None, "number without volume context"),
    # --- stop/pause/resume/play/queue (keep working) ---
    ("stop the music", "music_pause", {}, "log case"),
    ("Can you stop the music?", "music_pause", {}, "log case"),
    ("Pause.", "music_pause", {}, ""),
    ("continue your...", "music_resume", {}, "log case"),
    ("Resume.", "music_resume", {}, ""),
    ("Play it.", "music_resume", {}, "log case"),
    ("Can you play?", "music_resume", {}, "log case"),
    ("Can you play music?", "music_play", {"query": "music"}, "log case - keep"),
    ("you play my chemical romance", "music_play", {"query": "my chemical romance"}, "log case - keep"),
    ("Can you play any album of phum space viphurit?", "music_play", {"query": "any album of phum space viphurit"}, "log case - keep (query lowercased)"),
    ("Play Bohemian Rhapsody.", "music_play", {"query": "bohemian rhapsody"}, ""),
    ("Put on some jazz.", "music_play", {"query": "some jazz"}, ""),
    ("Add Shape of You to the queue.", "music_queue", {"query": "shape of you"}, ""),
    ("Shuffle on.", "music_shuffle", {"state": "on"}, ""),
    ("Repeat the song.", "music_repeat", {"mode": "track"}, ""),
    ("Previous song.", "music_skip", {"direction": "previous"}, ""),
    ("What's playing?", "music_now_playing", {}, ""),
    ("Which song is this?", "music_now_playing", {}, ""),
    ("List my devices.", "music_devices", {}, ""),
    ("Restart the song.", "music_restart", {}, ""),
    ("Play it one more time.", "music_restart", {}, ""),
    # --- things that must stay OUT of the fast path ---
    ("What time is it?", None, None, "NOT_OURS"),
    ("Set a timer for 5 minutes.", None, None, "NOT_OURS"),
    ("What's the weather?", None, None, "NOT_OURS"),
    ("Who sings this song?", None, None, "LLM - keep"),
    ("I love Bruno Mars.", None, None, "statement - LLM/SKIP"),
    ("do your love", None, None, "lyric fragment - LLM/SKIP"),
    ("Rock song.", None, None, "fragment - LLM decides"),
    ("Search for the Eiffel Tower.", None, None, "other tool"),
]

fails = 0
for text, want_tool, want_args, note in CASES:
    got = parse_music_command(text)
    got_tool = got["tool"] if got else None
    ok = got_tool == want_tool
    if ok and want_args is not None:
        ok = all(got["args"].get(k) == v for k, v in want_args.items())
    mark = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
        print(f"{mark}  {text!r}\n      want {want_tool} {want_args}  got {got}  ({note})")
    else:
        print(f"{mark}  {text!r} -> {got_tool} {got['args'] if got else ''}")
print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
