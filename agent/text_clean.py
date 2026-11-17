"""Plain-text rendering for a voice agent — markdown and symbols out.

The LLM writes markdown even when it should speak: **bold**, bullets, code
spans, links. The chat bubble shows it and Piper reads it. Both paths get
the same cleaner: emphasis unwrapped, headings/bullets/rules flattened,
links reduced to their label, decorative symbols dropped, punctuation runs
collapsed — and sentences spaced when the model glued them
("end.Next" -> "end. Next"), so replies read as plain sentences.

Digit-anchored and word-boundary guarded where it matters: decimals
("3.5"), abbreviations ("A.M.", "U.S.A.") and plain words ("I am") survive.
"""

import re

# --- markdown removal ------------------------------------------------------
_CODE_SPAN = re.compile(r"`([^`]+)`")  # `code` -> code
_BOLD = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")  # **bold** / __bold__
_ITALIC = re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)|(?<!\w)_([^_\n]+)_(?!\w)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [label](url) -> label
_HEADING = re.compile(r"(?m)^#{1,6}\s+")  # ## Heading
_RULE = re.compile(r"(?m)^[\s\t]*([-*_][\s\t]*){3,}$")  # --- / *** rules
_BULLET = re.compile(r"(?m)^[\s\t]*[-*•·]+[\s\t]+")  # - item / * item
_STRAY = re.compile(r"[*#`]+")  # leftovers the patterns above missed
_SYMBOL = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"  # emoji, dingbats
)

# --- sentence flow ----------------------------------------------------------
_GLUE = re.compile(r"([a-z][.!?]+[\"'\u2019\u201d)\]]*)([A-Z\u201c])")
_PUNCT_RUN = re.compile(r"([!?])\1+")  # !!! -> !  (?! stays; different chars)
_SPACES = re.compile(r"[ \t]{2,}")
_LINES = re.compile(r"\n{2,}")


def plain_speech(text: str) -> str:
    """Markdown and decorations -> plain, sentence-spaced prose."""
    text = _CODE_SPAN.sub(r"\1", text)
    text = _BOLD.sub(lambda m: (m.group(1) or m.group(2) or ""), text)
    text = _ITALIC.sub(lambda m: (m.group(1) or m.group(2) or ""), text)
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _RULE.sub("", text)
    text = _BULLET.sub("", text)
    text = _STRAY.sub("", text)
    text = _SYMBOL.sub("", text)
    text = _PUNCT_RUN.sub(r"\1", text)
    # After the strips: "end.**Next**" is "end.Next" here, and gets its space.
    text = _GLUE.sub(r"\1 \2", text)
    text = _SPACES.sub(" ", text)
    text = _LINES.sub("\n", text)
    return text.strip()
