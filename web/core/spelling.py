"""Voice spelling: decode spelled-out words ("L-A-V-A", "Lima Alpha Victor
Alpha") into plain text so the existing search/play flow can use the exact
spelling, when normal speech recognition would snap them to wrong words.

Integration: handle_spelling() runs in process_utterance() right after the
transcript frame and BEFORE the proofread pass - the decoded text is
canonical, so proofreading it could only mangle it. The reconstructed
sentence ("Search for LAVA") then flows through the normal pipeline
(music fast path or LLM tools); no separate search implementation.

Activation is deliberately strict (requirement: never read ordinary speech
as spelling): a cue phrase ("spell that as ...", "the letters are ...",
"search for the letters ...") must be present, and the words after it must
all resolve to letters. A spoken separator ("space", "pause") splits the
run into multiple words: "J O H N, space, W I C K" -> "JOHN WICK". A cue
with no letters in the same utterance
("I will spell it out.") arms a short expectation window for the next
utterance. Uncertain decodes (a letter that doubles as a common word:
"oh", "you", "why") trigger a spoken confirmation instead of acting.
"""

import os
import re
import time
from dataclasses import dataclass

from loguru import logger

from web.core.speech import send, speak_text
from web.core.state import _trim_history, done_stats

# --- timing -----------------------------------------------------------------
# How long the "did you spell X?" question stays open for a yes/no answer.
SPELLING_CONFIRM_TTL = float(os.getenv("SPELLING_CONFIRM_TTL", "30"))
# How long a bare letter sequence is accepted after a cue-only utterance.
SPELLING_EXPECT_TTL = float(os.getenv("SPELLING_EXPECT_TTL", "15"))
# A spelled word shorter than this is too easy to misread as normal speech.
MIN_LETTERS = 2
MAX_LETTERS = 20


# --- letter maps --------------------------------------------------------------
# NATO phonetic alphabet (strong: unambiguous).
_NATO = {
    "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliett": "J",
    "julyet": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}
# Ordinary letter names (strong: not common words in this position).
_LETTER_NAMES = {
    "ay": "A",
    "aye": "A",
    "bee": "B",
    "cee": "C",
    "dee": "D",
    "ee": "E",
    "ef": "E",
    "eff": "E",
    "gee": "G",
    "jee": "G",
    "aitch": "H",
    "hitch": "H",
    "eye": "I",
    "jay": "J",
    "kay": "K",
    "ell": "L",
    "el": "L",
    "em": "M",
    "en": "N",
    "pee": "P",
    "cue": "Q",
    "queue": "Q",
    "ar": "R",
    "ess": "S",
    "tee": "T",
    "vee": "V",
    "ex": "X",
    "xray": "X",
    "exray": "X",
    "wye": "Y",
    "zee": "Z",
    "zed": "Z",
    "zoo": "Z",
}
# Words that ARE common English words and only become letters here because a
# spelling cue preceded them (weak: trigger a confirmation).
_WEAK = {
    "oh": "O",
    "o": "O",
    "you": "U",
    "why": "Y",
    "an": "A",
    "eh": "E",
}
# Multi-word letter names, checked before single tokens (strong).
_MULTIWORD = {
    "double u": "W",
    "double you": "W",
    "double-u": "W",
    "x ray": "X",
    "ex ray": "X",
    "x-ray": "X",
    "ex-ray": "X",
}
# Explicit spoken word separators: the user marks where one spelled word
# ends and the next begins ("J O H N, space, W I C K"). "space" is the
# standard one; "pause" is the common mis-transcription of it, and both are
# rare enough in speech to be safe inside a strict spelling cue.
_SEPARATORS = {"space", "pause"}

# Letter names for the spoken read-back ("did you spell ell, ay, vee, ay?").
LETTER_NAMES_SPOKEN = {
    "A": "ay",
    "B": "bee",
    "C": "cee",
    "D": "dee",
    "E": "ee",
    "F": "eff",
    "G": "jee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "ell",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}


def _token_letter(tok: str) -> tuple[str, bool] | None:
    """Map one normalized token to (letter, strong). None = not a letter."""
    if len(tok) == 1 and tok.isalpha():
        return tok.upper(), True  # typed letter: the recognizer wrote a letter
    if tok in _NATO:
        return _NATO[tok], True
    if tok in _LETTER_NAMES:
        return _LETTER_NAMES[tok], True
    if tok in _WEAK:
        return _WEAK[tok], False
    return None


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace. Hyphens become
    spaces, so 'L-A-V-A' and 'double-u' tokenize predictably."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return [w for w in t.split() if w]


def decode_letters(text: str) -> tuple[list[list[str]], bool] | None:
    """Parse a string of letter tokens into ([words], all_strong), each word a
    list of letters. A spoken separator ("space", "pause") closes the current
    word and starts the next: "J O H N, space, W I C K" -> [["J","O","H","N"],
    ["W","I","C","K"]].

    Returns None when fewer than MIN_LETTERS tokens resolve. A token that
    does not resolve at all breaks the sequence (returns None) - partial
    guesses on a broken sequence are how this feature would start eating
    ordinary speech. A separator with no letters yet (leading "space") or a
    trailing "space" is ignored rather than fatal: Whisper often inserts
    punctuation the user did not speak."""
    toks = _normalize_tokens(text)
    if not toks:
        return None
    words: list[list[str]] = []
    cur: list[str] = []
    all_strong = True
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in _SEPARATORS:
            if cur:
                words.append(cur)
                cur = []
            i += 1
            continue
        # Multi-word names first ("double u" is two tokens).
        if i + 1 < len(toks):
            pair = f"{tok} {toks[i+1]}"
            if pair in _MULTIWORD:
                cur.append(_MULTIWORD[pair])
                i += 2
                continue
        hit = _token_letter(tok)
        if hit is None:
            return None
        letter, strong = hit
        cur.append(letter)
        all_strong = all_strong and strong
        i += 1
    if cur:
        words.append(cur)
    # Every word must be long enough to be a real spelled word; a lone stray
    # letter after a separator ("... space, Z") is a mis-hear, not a word.
    if not words or any(not (MIN_LETTERS <= len(w) <= MAX_LETTERS) for w in words):
        return None
    return words, all_strong


def decode_region(region: str) -> tuple[list[list[str]], bool] | None:
    """Decode the letter region after a cue. Falls back to per-character
    mapping when Whisper collapses the letter run into one word
    ('search for the letters LAVA'): inside a strict spelling cue, each
    character of that word IS a letter."""
    dec = decode_letters(region)
    if dec:
        return dec
    toks = _normalize_tokens(region)
    if (
        len(toks) == 1
        and toks[0].isalpha()
        and MIN_LETTERS <= len(toks[0]) <= MAX_LETTERS
    ):
        return [list(toks[0].upper())], True
    return None


# --- cue detection ------------------------------------------------------------
# A spelling cue: an explicit instruction that letters are coming. The
# alternatives are ordered longest-first so the match ends as far right as
# possible (the letter region starts right after it).
_CUE_RE = re.compile(
    r"""
    (?:
        (?:i|im|i'?m)\s+(?:gonna|going\s+to|will|just|about\s+to)\s+spell
        \s+(?:it|that|this|them)\s*(?:out|for\s+you|for\s+me)?
      | spell\s+(?:it|that|this|them|the\s+[a-z]+)
        \s*(?:as|out|it\s+out)?
      | spelled\s*(?:as|:)?
      | spelling\s+(?:is|:)?
      | (?:search|look\s*up|find|lookup)\s+(?:for\s+)?(?:the\s+)?letters?
      | the\s+letters?\s+(?:are|:)?
      | say\s+(?:each\s+|one\s+by\s+one\s+)?letter
      | letter\s+by\s+letter
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Connectors that can sit between the cue and the letters.
_CONNECTOR_RE = re.compile(r"^\s*(?:as|out|is|are|:|for\s+me|for\s+you|okay|right|ok)?[\s,]*")


def find_spelling(text: str) -> tuple[str, str] | None:
    """Locate a cue in `text`. Returns (prefix_before_or_incl_cue, letter_region)
    where letter_region is everything after the cue (connectors stripped).
    Returns None when no cue is present."""
    m = _CUE_RE.search(text)
    if not m:
        return None
    prefix = text[: m.end()]
    region = text[m.end() :]
    region = _CONNECTOR_RE.sub("", region, count=1)
    region = region.strip(" ,;:.-")
    return prefix, region


def has_spelling_cue(text: str) -> bool:
    return _CUE_RE.search(text) is not None


# --- confirmation parsing ------------------------------------------------------
# Bare fillers ("uh", "up") deliberately excluded: a filler is not a
# confirmation, and acting on an unconfirmed spelling is the failure this
# question exists to prevent.
_YES_RE = re.compile(
    r"^(?:y(?:e(?:s|p|ah|a|aa)?)?|yeah|yup|yus|yes|correct|right|"
    r"that'?s\s+(?:right|correct)|exactly|perfect|sure|absolutely|confirm(?:ed)?|"
    r"go\s+(?:ahead|for\s+it)|do\s+it|search\s+it|that'?s\s+it|right\s+)$",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"^(?:no|nope|nah|naw|wrong|that'?s\s+wrong|incorrect|not\s+quite|nope|"
    r"sorry|actually|no\s+it(?:'?s)?\s+not)$",
    re.IGNORECASE,
)


def parse_confirmation(text: str) -> bool | None:
    """Classify a reply to the spelling question. True/False, or None when the
    utterance is not a yes/no answer (the user moved on - or is re-spelling,
    which the cue detector picks up)."""
    t = re.sub(r"[^a-z' ]+", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return None
    if _YES_RE.match(t):
        return True
    if _NO_RE.match(t):
        return False
    return None


# --- the turn-level handler ----------------------------------------------------
@dataclass
class SpellingOutcome:
    kind: str  # "run" (use .text) | "done" (turn fully handled here)
    text: str = ""
    decoded: bool = False  # True when .text is a decoded/confirmed rewrite


def _readback(words: list[list[str]]) -> str:
    """Spoken form of a decoded spelling: letters comma-separated, words
    joined by ", space, " so the user can hear exactly where the split is
    ("jay, oh, aych, en, space, double you, eye, see, kay")."""
    return ", space, ".join(
        ", ".join(LETTER_NAMES_SPOKEN[l] for l in w) for w in words
    )


async def _speak_turn(
    ws, sess: dict, user_text: str, reply: str, stats: dict
) -> None:
    """Bookend a short spoken exchange the way the music fast path does:
    history pair, text frame, TTS, done frame."""
    history = sess["history"]
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    _trim_history(history)
    await send(ws, {"type": "text", "t": reply})
    try:
        await speak_text(ws, sess, reply, stats)
    except Exception as e:
        logger.warning(f"[web] spelling TTS failed: {e}")
    await send(
        ws,
        {
            "type": "done",
            "reply": reply,
            "stats": done_stats(
                stats, ttfa_ms=(time.perf_counter() - stats["turn_t0"]) * 1000
            ),
        },
    )


async def _ask_confirmation(
    ws,
    sess: dict,
    user_text: str,
    words: list[list[str]],
    rewritten: str,
    stats: dict,
) -> SpellingOutcome:
    q = f"Sorry, did you spell {_readback(words)}?"
    sess["pending_spelling"] = {
        "rewritten": rewritten,
        "letters": words,
        "asked_at": time.monotonic(),
    }
    logger.info(
        f"[web] spelling: asking to confirm {' '.join(''.join(w) for w in words)!r}"
    )
    await _speak_turn(ws, sess, user_text, q, stats)
    return SpellingOutcome(kind="done")


def _rewrite(prefix: str, word: str) -> str:
    """Rebuild the sentence with the decoded word in place of the letters.
    'Search for the letters LAVA' -> 'Search for LAVA'; everything else keeps
    its original shape ('Spell that as LAVA'), which the LLM understands."""
    p = re.sub(r"\s*the\s+letters?\s*$", " ", prefix, flags=re.IGNORECASE)
    p = p.strip(" ,;:.-")
    return f"{p} {word}".strip() if p else word


async def handle_spelling(
    ws, sess: dict, text: str, stats: dict
) -> SpellingOutcome:
    """Run the spelling state machine on a fresh transcript.

    "run"  -> the caller continues with outcome.text (possibly rewritten).
    "done" -> a question/ack was spoken; the caller returns."""
    now = time.monotonic()

    # 1) A pending "did you spell X?" resolves first: a bare yes/no must not
    #    fall through to the music fast path or the LLM.
    pending = sess.get("pending_spelling")
    if pending:
        if now - pending["asked_at"] > SPELLING_CONFIRM_TTL:
            logger.debug("[web] spelling: confirmation expired")
            sess.pop("pending_spelling", None)
        else:
            verdict = parse_confirmation(text)
            if verdict is not None:
                rewritten = pending["rewritten"]
                sess.pop("pending_spelling", None)
                if verdict:
                    logger.info(
                        f"[web] spelling: confirmed {rewritten!r}"
                    )
                    return SpellingOutcome(
                        kind="run",
                        text=f"Yes, {rewritten}",
                        decoded=True,
                    )
                logger.info("[web] spelling: user rejected the spelling")
                await _speak_turn(ws, sess, text, "Okay, no problem.", stats)
                return SpellingOutcome(kind="done")
            # Not a yes/no: the user moved on (or is re-spelling - the cue
            # detector below picks that up). Drop the pending and continue.
            sess.pop("pending_spelling", None)

    # 2) Letters arriving after a cue-only utterance ("I will spell it out").
    expecting = sess.get("spelling_expected")
    if expecting and now - expecting["at"] < SPELLING_EXPECT_TTL:
        sess.pop("spelling_expected", None)
        dec = decode_letters(text)
        if dec:
            words, strong = dec
            word = " ".join("".join(w) for w in words)
            rewritten = f"I spelled it out: {word}."
            if strong:
                logger.info(f"[web] spelling: {text!r} -> {word!r}")
                return SpellingOutcome(kind="run", text=rewritten, decoded=True)
            return await _ask_confirmation(ws, sess, text, words, rewritten, stats)
        # Not a letter sequence: the user said something else instead.
        return SpellingOutcome(kind="run", text=text)

    # 3) A cue (with or without the letters) in this utterance.
    found = find_spelling(text)
    if not found:
        return SpellingOutcome(kind="run", text=text)
    prefix, region = found
    dec = decode_region(region)
    if dec:
        words, strong = dec
        word = " ".join("".join(w) for w in words)
        rewritten = _rewrite(prefix, word)
        if strong:
            logger.info(f"[web] spelling: {text!r} -> {rewritten!r}")
            return SpellingOutcome(kind="run", text=rewritten, decoded=True)
        return await _ask_confirmation(ws, sess, text, words, rewritten, stats)
    # Cue present but no usable letters in this utterance: arm the expectation
    # window and invite the letters.
    logger.info(f"[web] spelling: cue without letters - expecting: {text!r}")
    sess["spelling_expected"] = {"at": now}
    await _speak_turn(ws, sess, text, "Go ahead, I'm listening.", stats)
    return SpellingOutcome(kind="done")
