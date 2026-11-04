"""Unit tests for the browser fast-path music command parser (web/server.py).

Run:  .venv/bin/python tests/test_music_parser.py
No network, no servers. Imports web.server (loads .env) and exercises
parse_music_command / maybe_music_command with music mode forced on.
"""
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import web.server as ws  # noqa: E402

ws.STATE["music_mode_until"] = 9e18  # force music mode on

CASES = [
    # stop / pause
    ("stop the music",        "music_pause", {}),
    ("stop",                  "music_pause", {}),
    ("pause the music",       "music_pause", {}),
    ("hey lava please can you pause the music", "music_pause", {}),
    ("stop the timer",        None, None),          # NOT_OURS -> LLM
    # resume
    ("resume the music",      "music_resume", {}),
    ("unpause",               "music_resume", {}),
    ("continue playing",      "music_resume", {}),
    ("continue the music",    "music_resume", {}),
    ("carry on",              "music_resume", {}),
    ("play it",               "music_resume", {}),
    ("play",                  "music_resume", {}),
    # next / previous
    ("next",                  "music_skip", {"direction": "next"}),
    ("skip",                  "music_skip", {"direction": "next"}),
    ("next, please",          "music_skip", {"direction": "next"}),
    ("skip this song",        "music_skip", {"direction": "next"}),
    ("next song",             "music_skip", {"direction": "next"}),
    ("go back",               "music_skip", {"direction": "previous"}),
    ("go back one song",      "music_skip", {"direction": "previous"}),
    ("previous",              "music_skip", {"direction": "previous"}),
    ("last song",             "music_skip", {"direction": "previous"}),
    # restart
    ("restart the song",      "music_restart", {}),
    ("rewind",                "music_restart", {}),
    ("play it again",         "music_restart", {}),
    ("start over",            "music_restart", {}),
    ("one more time",         "music_restart", {}),   # must beat \btime\b NOT_OURS
    # volume
    ("louder",                "music_volume", {"level": "up"}),
    ("turn it up",            "music_volume", {"level": "up"}),
    ("volume up",             "music_volume", {"level": "up"}),
    ("volume down",           "music_volume", {"level": "down"}),
    ("quieter",               "music_volume", {"level": "down"}),
    ("make it quieter",       "music_volume", {"level": "down"}),
    ("set volume to 40",      "music_volume", {"level": "40"}),
    ("turn it up to 40",      "music_volume", {"level": "40"}),        # absolute
    ("turn the volume down to 20", "music_volume", {"level": "20"}),   # absolute
    ("volume 30",             "music_volume", {"level": "30"}),
    ("louder by 10",          "music_volume", {"level": "up 10"}),
    ("turn the volume up by 20", "music_volume", {"level": "up 20"}),
    ("lower the volume by 15", "music_volume", {"level": "down 15"}),
    ("turn the music up",     "music_volume", {"level": "up"}),
    ("max volume",            "music_volume", {"level": "max"}),
    ("volume max",            "music_volume", {"level": "max"}),
    ("mute",                  "music_volume", {"level": "mute"}),
    ("mute the music",        "music_volume", {"level": "mute"}),
    ("what's the volume",     "music_volume", {"level": ""}),   # report current
    # now playing / devices
    ("what's playing",        "music_now_playing", {}),
    ("what is this song",     "music_now_playing", {}),
    ("which song is this",    "music_now_playing", {}),
    ("now playing",           "music_now_playing", {}),
    ("what song is this",     "music_now_playing", {}),
    ("what's the weather",    None, None),
    ("what time is it",       None, None),                      # NOT_OURS
    ("music devices",         "music_devices", {}),
    # shuffle / repeat
    ("shuffle on",            "music_shuffle", {"state": "on"}),
    ("shuffle off",           "music_shuffle", {"state": "off"}),
    ("shuffle",               "music_shuffle", {"state": ""}),
    ("repeat this song",      "music_repeat", {"mode": "track"}),
    ("repeat off",            "music_repeat", {"mode": "off"}),
    ("repeat the playlist",   "music_repeat", {"mode": "context"}),
    # queue
    ("queue Some Coldplay",   "music_queue", {"query": "some coldplay"}),
    ("add Borna to the queue", "music_queue", {"query": "borna"}),
    # play queries
    ("play Some Coldplay",    "music_play", {"query": "some coldplay", "kind": "track"}),
    ("play the song Borna",   "music_play", {"query": "borna", "kind": "track"}),
    ("play playlist Discover Weekly", "music_play",
     {"query": "discover weekly", "kind": "playlist"}),
    ("play album Rumours",    "music_play", {"query": "rumours", "kind": "album"}),
    ("put on some jazz",      "music_play", {"query": "some jazz", "kind": "track"}),
    ("play e.g. an incorrectly long query " * 2, None, None),
    # non-music words must not hijack outside music intent
    ("next tuesday",          None, None),
]


def main() -> int:
    failures = 0
    for text, want_tool, want_args in CASES:
        got = ws.parse_music_command(text)
        if want_tool is None:
            ok = got is None
        else:
            ok = (isinstance(got, dict) and got.get("tool") == want_tool
                  and got.get("args") == want_args)
        if not ok:
            failures += 1
            print(f"FAIL  {text!r}\n      want={want_tool} {want_args}\n      "
                  f"got ={(got or {}).get('tool') if got else None} "
                  f"{(got or {}).get('args')}")
    print(f"{len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
