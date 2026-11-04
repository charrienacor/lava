"""Unit tests for Full-Duplex-Bench v1.5 overlap classification (web/server.py).

The paper defines four overlap scenarios and a DESIRED behavior for each:

  USER_INTR      user barges in with a new request   -> cut, respond (low t_stop/t_resp)
  USER_BACKCH    brief acknowledgment ("uh-huh")     -> keep talking (RESUME)
  TALKING_OTHER  user addresses someone else         -> hold the floor (do NOT cut)
  BKG_SPEECH     far-field third-party talk          -> hold the floor (do NOT cut)

Lava maps them onto classify_overlap() (backchannel / talk_to_other /
interruption). Background speech mostly never reaches the classifier - the
VAD loudness gate and the music-mode lyric guard drop it upstream - so it is
covered here only at the "does not crash / defaults to interruption" level.

The critical property is precision: a REAL interruption must never be
swallowed as talk-to-other or backchannel. Every negative case below is a
phrase a user would plausibly say to the agent while it is talking.

Run:  .venv/bin/python tests/test_fullduplex.py
No network, no servers.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import web.server as ws  # noqa: E402

# (transcript, expected classify_overlap result)
CASES = [
    # --- USER_BACKCH: pure acknowledgements -> keep talking ---------------
    ("mm-hmm", "backchannel"),
    ("mmhmm", "backchannel"),
    ("uh huh", "backchannel"),
    ("yeah", "backchannel"),
    ("got it", "backchannel"),
    ("right", "backchannel"),
    ("sounds good", "backchannel"),

    # --- TALKING_OTHER: addressed to someone else -> hold the floor -------
    ("Hey John, can you grab the milk?", "talk_to_other"),
    ("Maria, are you coming?", "talk_to_other"),
    ("Wait Maria, what did he say?", "talk_to_other"),
    ("you guys, let's go", "talk_to_other"),
    ("everyone, heads up", "talk_to_other"),
    ("I was talking to you about that", "talk_to_other"),
    ("not to you, to him", "talk_to_other"),

    # --- USER_INTR: real interruptions -> must CUT (never swallowed) ------
    ("stop, play something else", "interruption"),
    ("Hey, what is the weather like?", "interruption"),   # vocative, no name
    ("Hey Lava, what time is it?", "interruption"),       # agent's own name
    ("Lava, what time is it?", "interruption"),           # agent's own name
    ("play bohemian rhapsody", "interruption"),
    ("Wait, that is wrong", "interruption"),
    ("Well, I think we should go", "interruption"),       # discourse marker
    ("So, what do you think?", "interruption"),           # discourse marker
    ("Yeah, but actually I want the other one", "interruption"),  # >4 words
    ("I want to change the plan now", "interruption"),
    ("", "interruption"),                                  # empty -> safe default
]

# --- fd_summary / _fd_record sanity ----------------------------------------


def test_fd_record_and_summary():
    ws.STATE["fd_stop_ms"].clear()
    ws.STATE["fd_resp_ms"].clear()
    for v in (100, 200, 300, 400, 500):
        ws._fd_record("fd_stop_ms", v)
    s = ws.fd_summary()
    assert s["t_stop_ms"]["n"] == 5
    assert s["t_stop_ms"]["p50"] == 300
    assert s["t_stop_ms"]["p95"] == 500
    assert s["t_resp_ms"] is None
    # rolling window caps at 200
    for v in range(300):
        ws._fd_record("fd_resp_ms", v)
    assert ws.fd_summary()["t_resp_ms"]["n"] == 200
    ws.STATE["fd_stop_ms"].clear()
    ws.STATE["fd_resp_ms"].clear()


def main():
    fails = 0
    for text, want in CASES:
        got = ws.classify_overlap(text)
        ok = got == want
        fails += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {want:<14} <- {text!r}")
    test_fd_record_and_summary()
    print("ok   fd_summary/_fd_record (window, percentiles)")
    print()
    if fails:
        print(f"{fails} FAILURE(S)")
        sys.exit(1)
    print(f"ALL PASS ({len(CASES)} classification cases + metrics)")


if __name__ == "__main__":
    main()
