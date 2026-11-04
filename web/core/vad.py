"""Voice activity detection (Silero) and the Full-Duplex-Bench overlap
classification: backchannels, talk-to-others, and background speech."""

import math
import os
import re
import time

import loudness as _bs1770
import numpy as np
from loguru import logger
from pipecat.audio.vad import silero as _silero_mod
from pipecat.audio.vad.silero import SileroOnnxModel, SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from web.core.config import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    MUSIC_MODE_MIN_RATIO,
    MUSIC_MODE_VOLUME_DELTA,
    SAMPLE_RATE,
)
from web.core.state import STATE

# Backchannels: a short acknowledgment ("uh-huh", "mm-hmm", "yeah") while the
# bot is talking is NOT an interruption - the bot should keep speaking. The
# barge-in cut is therefore deferred until the span transcribes; if the text is
# a pure backchannel the cut is dropped and the bot continues. A watchdog
# (BACKCHANNEL_CUT_MAX_SECS) still forces the cut if a real interruption keeps
# the span open past the cap, so responsiveness is bounded.
BACKCHANNEL_CUT_MAX_SECS = float(os.getenv("BACKCHANNEL_CUT_MAX_SECS", "1.0"))
BACKCHANNEL_MAX_WORDS = int(os.getenv("BACKCHANNEL_MAX_WORDS", "4"))
_BACKCHANNEL_TEXTS = {
    "uhhuh",
    "uhhuhhuh",
    "uh",
    "uhhuhuh",
    "uhuh",
    "uh huh",
    "uhhuh uh",
    "mhmm",
    "mmhmm",
    "mhm",
    "mm",
    "hmm",
    "hmmhmm",
    "mhmhmm",
    "mmm",
    "mmmm",
    "yeah",
    "yep",
    "yea",
    "yup",
    "yess",
    "ya",
    "yeah yeah",
    "yep yep",
    "right",
    "sure",
    "exactly",
    "got it",
    "gotcha",
    "okay",
    "ok",
    "k",
    "kk",
    "cool",
    "nice",
    "great",
    "good",
    "perfect",
    "absolutely",
    "indeed",
    "of course",
    "for sure",
    "yes",
    "yes sir",
    "yes maam",
    "affirmative",
    "that's right",
    "thats right",
    "i see",
    "i got it",
    "understood",
    "roger",
    "copy that",
    "sounds good",
    "makes sense",
    "no problem",
    "no worries",
    "sure thing",
    "my bad",
    "oh okay",
    "oh i see",
}

# --- Full-Duplex-Bench v1.5: TALKING_OTHER and BKG_SPEECH ---------------------
# The other two overlap scenarios are speech that is NOT addressed to the
# agent. The desired behavior there is the opposite of an interruption:
# HOLD THE FLOOR (keep talking, do not cut). We detect the cheap, high-
# precision case - a vocative opener ("hey John", "Maria, ...") or an explicit
# "not you" cue ("you guys", "talk to you") - and drop the cut. Far-field
# background speech is already handled acoustically: it stays under the
# VAD's loudness gate, and the music-mode lyric guard drops what survives.
FD_HOLD_FLOOR = os.getenv("FD_HOLD_FLOOR", "true").lower() in ("1", "true", "yes", "on")
_FD_VOCATIVES = {
    "hey",
    "hi",
    "hello",
    "yo",
    "eh",
    "hey hey",
    "hmm",
    "uh",
    "ok",
    "okay",
    "alright",
    "alrighty",
    "come",
    "listen",
    "wait",
    "so",
    "well",
    "um",
    "uhh",
    "erm",
}
_FD_NOT_YOU_PATTERNS = (
    re.compile(r"\b(you guys|you all|everyone|everybody|folks|guys|babe|hon|honey)\b"),
    re.compile(r"\b(talk|talking|speaking|saying|said|asked|asking) (to|with) you\b"),
    re.compile(r"\bnot (to|for) you\b"),
)
# Sentence-initial discourse markers that are often followed by a comma but
# are NOT an address ("Well, ...", "So, ..."). Excluded from the
# "Name, ..." vocative rule so they keep cutting as normal turns.
_FD_DISCOURSE_MARKERS = {
    "well",
    "so",
    "but",
    "and",
    "or",
    "yet",
    "now",
    "then",
    "here",
    "there",
    "actually",
    "basically",
    "literally",
    "honestly",
    "okay",
    "ok",
    "alright",
    "hey",
    "hi",
    "hello",
    "yo",
    "um",
    "uh",
    "erm",
    "come",
    "listen",
    "wait",
    "right",
    "yeah",
    "yep",
    "sure",
    "no",
    "yes",
    "look",
    "mean",
    "guess",
}
# The agent's own name(s): "Hey Lava, ..." addresses the agent, so it is an
# interruption, not talk-to-other.
_FD_AGENT_NAMES = {"lava"}
# Compact lookup (spaces, hyphens and apostrophes removed): "mm-hmm",
# "mmhmm" and "mm hmm" must all classify the same way.
_BACKCHANNEL_COMPACT = {re.sub(r"[^a-z]", "", x) for x in _BACKCHANNEL_TEXTS}


# VAD (Silero, server-side; the browser just streams raw PCM)
# Silero, not an energy VAD: it scores each 32 ms window by how
# human-speech-like it is, so TV speech plays out while the user still
# triggers.


class _LoudnessTracker:
    """Rolling-window BS.1770 loudness, 0..1.

    Mirrors pipecat's AudioVolumeTracker + calculate_audio_volume (rolling
    400 ms window, integrated loudness, -110..-10 LUFS mapped to 0..1) so the
    VAD_MIN_VOLUME gate keeps its tuned scale without touching pipecat's
    private ``_volume_tracker`` member.
    """

    WINDOW_SECS = 0.4
    LOUD_LO = -110.0
    LOUD_HI = -10.0

    def __init__(self):
        self._sample_rate = 0
        self._window_bytes = 0
        self._buf = bytearray()
        # None once the window holds audio that hasn't been measured yet.
        self._volume: float | None = 0.0

    @property
    def volume(self) -> float:
        if self._volume is None:
            audio = (
                np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            level = _bs1770.integrated_loudness(audio, self._sample_rate)
            # Below the BS.1770 absolute gate this is -inf; nan shouldn't
            # occur. Both normalize to 0 (quiet).
            self._volume = (
                0.0
                if not math.isfinite(level)
                else max(
                    0.0,
                    min(1.0, (level - self.LOUD_LO) / (self.LOUD_HI - self.LOUD_LO)),
                )
            )
        return self._volume

    def update(self, audio: bytes, sample_rate: int):
        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate
            # Rounded up so the window is never a sample short of a gating
            # block, which loudness rejects.
            self._window_bytes = math.ceil(self.WINDOW_SECS * sample_rate) * 2
            self._buf.clear()
            self._volume = 0.0
        self._buf += audio
        if len(self._buf) > self._window_bytes:
            del self._buf[: len(self._buf) - self._window_bytes]
        if len(self._buf) == self._window_bytes:
            self._volume = None


# --- Shared Silero ONNX session -------------------------------------------------
# Without this, every browser tab pays the full cost: each SileroVADAnalyzer
# wraps a SileroOnnxModel, and that model builds its own
# onnxruntime.InferenceSession (tens of MB, ~100 ms init). The LSTM state that
# makes the model per-stream is NOT in the session: SileroOnnxModel passes it
# as a graph input on every call ("state" in ort_inputs) and keeps the
# returned state in its own _state/_context arrays. The InferenceSession is
# therefore stateless with respect to the stream, and every analyzer in the
# process shares one. No lock is needed: InferenceSession.run is thread-safe,
# and VADState() is constructed synchronously on the event loop, so builds and
# calls from different tabs serialize.
_SHARED_SESSION = None


def _shared_silero_session():
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        from importlib import resources as _ir

        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        path = str(_ir.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))
        _SHARED_SESSION = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
    return _SHARED_SESSION


class _SharedSileroModel(SileroOnnxModel):
    """SileroOnnxModel that adopts the process-wide session.

    Mirrors SileroOnnxModel.__init__ minus the InferenceSession build: same
    attributes, same reset, same supported sample rates.
    """

    def __init__(self, path, force_onnx_cpu=True):
        self.session = _shared_silero_session()
        self.reset_states()
        self.sample_rates = [8000, 16000]


def _make_analyzer(params: VADParams) -> SileroVADAnalyzer:
    """SileroVADAnalyzer wired to the process-wide ONNX session.

    SileroVADAnalyzer has no injection point, so the model class is swapped
    for the duration of the constructor (it resolves SileroOnnxModel from the
    module globals). Builds happen on the event loop, so the swap is atomic.
    """
    prev = _silero_mod.SileroOnnxModel
    _silero_mod.SileroOnnxModel = _SharedSileroModel
    try:
        return SileroVADAnalyzer(sample_rate=SAMPLE_RATE, params=params)
    finally:
        _silero_mod.SileroOnnxModel = prev


class VADState:
    """Silero VAD + loudness gate + span gate.

    1. Silero scores each 32 ms window by how speech-like it is (music and
       TV noise stay below CONFIDENCE even when loud).
    2. A BS.1770 loudness gate (MIN_VOLUME) rejects far-away speech: the
       user is close to the mic, the TV is not. The gate has a fast attack
       and a slow release, so speech onsets are not eaten by smoothing.
    3. A span gate drops spans shorter than MIN_SAMPLES or sparser than
       MIN_VOICED_RATIO.

    A span opens on the first gated-speech window and closes after
    STOP_SECS of non-speech (or the 30 s cap). Spans include PRE_ROLL of
    audio from before the first voiced window, so the first word survives
    the loudness gate's rise time.

    A barge hint (BARGE_MS of gated speech) fires once per span; the session
    decides whether it cuts the bot (only while a turn is live or bot audio is
    playing)."""

    CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    START_SECS = float(os.getenv("VAD_START_SECS", "0.2"))
    STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.2"))
    MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.7"))
    MIN_SAMPLES = int(
        float(os.getenv("VAD_MIN_VOICED_MS", "450")) * SAMPLE_RATE // 1000
    )
    # Lower close bar for spans that opened on a barge hint (near-field, already
    # passed the loudness gate): a quick backchannel ("mm-hmm", ~300 ms) closes
    # and gets classified instead of waiting out the 450 ms far-field bar.
    BARGE_ORIGIN_MIN_SAMPLES = int(
        float(os.getenv("VAD_BARGE_ORIGIN_MIN_VOICED_MS", "150")) * SAMPLE_RATE // 1000
    )

    BARGE_SAMPLES = int(float(os.getenv("VAD_BARGE_MS", "200")) * SAMPLE_RATE // 1000)
    MIN_VOICED_RATIO = float(os.getenv("VAD_MIN_VOICED_RATIO", "0.30"))
    PRE_ROLL_BYTES = int(
        float(os.getenv("VAD_PRE_ROLL_MS", "120")) * SAMPLE_RATE * 2 / 1000
    )
    # Loudness smoothing: symmetric, matching the dynamics VAD_MIN_VOLUME was
    # tuned against. Fast-attack/slow-release sounds better in theory but in a
    # room with a TV it glues background speech into giant spans that the
    # voiced-ratio gate then discards - along with the command inside them.
    ATTACK = float(os.getenv("VAD_ATTACK", "0.2"))
    RELEASE = float(os.getenv("VAD_RELEASE", "0.2"))
    # Music mode: while Spotify is playing out loud the mic hears it too.
    # Raise the bar so song vocals don't become turns (handled by the
    # command fast path and the LLM music note instead).
    MUSIC_RATIO = max(0.45, MUSIC_MODE_MIN_RATIO)
    MUSIC_DELTA = MUSIC_MODE_VOLUME_DELTA
    # Spans that fail the voiced-ratio gate are usually the user's command
    # glued to TV dialogue. Transcribe them anyway (Whisper's quality gate and
    # the LLM's SKIP instruction are the filters designed for this); spans with
    # almost no voiced audio still get dropped. Set false to restore hard
    # discarding.
    PROCESS_NOISY_SPANS = os.getenv("VAD_PROCESS_NOISY_SPANS", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    MAX_SAMPLES = SAMPLE_RATE * 30
    MAX_IDLE_BYTES = 2 * SAMPLE_RATE * 2
    STOP_FRAMES = int(STOP_SECS * SAMPLE_RATE / FRAME_SAMPLES)

    @classmethod
    def music_mode(cls) -> bool:
        return time.monotonic() < STATE["music_mode_until"]

    @classmethod
    def ratio_floor(cls, music: bool) -> float:
        return cls.MUSIC_RATIO if music else cls.MIN_VOICED_RATIO

    def __init__(self):
        self._analyzer = _make_analyzer(
            VADParams(
                confidence=self.CONFIDENCE,
                start_secs=self.START_SECS,
                stop_secs=self.STOP_SECS,
                min_volume=self.MIN_VOLUME,
            )
        )
        self._analyzer.set_sample_rate(SAMPLE_RATE)
        self._loudness = _LoudnessTracker()

        self._prev_volume = 0.0
        self._pending = b""
        self.buf = bytearray()
        self.first_voiced = None
        self.last_voiced = None
        self.utterance = None
        self.last_ratio = 1.0  # voiced ratio of the last completed span
        self.barge_hint = False
        self.barge_origin = False  # True if the current span opened on a barge
        self._barge_fired = False
        self._silent = 0
        self._voiced_frames = 0
        self._span_frames = 0

    def feed(self, pcm: bytes):
        self._pending += pcm

        music = self.music_mode()
        conf_floor = self.CONFIDENCE + (self.MUSIC_DELTA if music else 0.0)
        vol_floor = self.MIN_VOLUME + (self.MUSIC_DELTA if music else 0.0)

        if self.first_voiced is None and len(self.buf) > self.MAX_IDLE_BYTES:
            del self.buf[: len(self.buf) - self.MAX_IDLE_BYTES]
        while len(self._pending) >= FRAME_BYTES:
            frame = self._pending[:FRAME_BYTES]
            self._pending = self._pending[FRAME_BYTES:]
            self.buf.extend(frame)
            confidence = self._analyzer.voice_confidence(frame)
            self._loudness.update(frame, SAMPLE_RATE)
            volume = self._loudness.volume
            alpha = self.ATTACK if volume > self._prev_volume else self.RELEASE
            volume = self._prev_volume + alpha * (volume - self._prev_volume)
            self._prev_volume = volume
            speech = confidence >= conf_floor and volume >= vol_floor
            g = (len(self.buf) - FRAME_BYTES) // 2
            if speech:
                if self.first_voiced is None:
                    self.first_voiced = g
                    self._voiced_frames = 0
                    self._span_frames = 0
                    self._barge_fired = False
                self.last_voiced = g + FRAME_SAMPLES
                self._silent = 0
                self._voiced_frames += 1
                self._span_frames += 1
            else:
                if self.first_voiced is not None:
                    self._span_frames += 1
                    self._silent += 1
            span = (
                self.last_voiced - self.first_voiced
                if self.first_voiced is not None and self.last_voiced is not None
                else 0
            )

            if (
                self.first_voiced is not None
                and not self._barge_fired
                and self._voiced_frames * FRAME_SAMPLES >= self.BARGE_SAMPLES
            ):
                self._barge_fired = True
                self.barge_hint = True
                # This span opened over the bot's own voice: it is near-field
                # and will be decided by the backchannel/interruption logic
                # downstream, so let it close on a short pause (a quick
                # "mm-hmm" classifies instead of hanging out the 450 ms
                # far-field bar).
                self.barge_origin = True
            # A span that opened on a barge hint is near-field (it already
            # passed the loudness gate) and is being decided by the
            # backchannel/interruption logic downstream - let it close on a
            # short pause so a quick "mm-hmm" is classified instead of hanging.
            # Normal spans keep the full MIN_SAMPLES bar (far-field TV noise).
            min_close = (
                self.BARGE_ORIGIN_MIN_SAMPLES if self.barge_origin else self.MIN_SAMPLES
            )
            if (
                self.first_voiced is not None
                and span >= min_close
                and (self._silent >= self.STOP_FRAMES or span >= self.MAX_SAMPLES)
            ):
                voiced_ratio = (
                    self._voiced_frames / self._span_frames
                    if self._span_frames
                    else 0.0
                )
                voiced_ms = self._voiced_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE
                start = max(0, self.first_voiced * 2 - self.PRE_ROLL_BYTES)
                span_bytes = bytes(self.buf[start:])
                self.buf.clear()
                self.first_voiced = self.last_voiced = None
                self._silent = 0
                self.barge_hint = False
                self._barge_fired = False
                self.barge_origin = False
                self._voiced_frames = self._span_frames = 0
                floor = self.ratio_floor(music)
                self.last_ratio = voiced_ratio
                if voiced_ratio < floor and (
                    not self.PROCESS_NOISY_SPANS
                    or voiced_ms < self.MIN_SAMPLES * 1000 / SAMPLE_RATE
                ):
                    logger.info(
                        f"[vad] discarded span: {span / SAMPLE_RATE:.1f}s, only "
                        f"{voiced_ratio:.0%} speech ({voiced_ms:.0f} ms voiced)"
                        + (" [music gate]" if music else "")
                    )
                else:
                    if voiced_ratio < floor:
                        logger.info(
                            f"[vad] noisy span: {span / SAMPLE_RATE:.1f}s, "
                            f"{voiced_ratio:.0%} speech - transcribing anyway"
                        )
                    self.utterance = span_bytes


def _is_skip(text: str) -> bool:
    """True if the LLM's output is just its 'not addressed to me' marker.

    Skipped text is never spoken and is dropped from history."""
    return text.strip().strip(".!?\"' ").upper() == "SKIP"


def is_backchannel(text: str) -> bool:
    """True if the transcript is a pure acknowledgment, not a real turn.

    Decides whether speech that overlapped the bot's own voice was a
    backchannel (keep talking) or an interruption (cut). Conservative on
    purpose: anything longer than BACKCHANNEL_MAX_WORDS, or not a known
    filler, is treated as a real turn so a genuine interruption is never
    swallowed. Runs on the raw (pre-proofread) transcript, so a Whisper
    mis-hearing of "uh-huh" that lands on a normal word still cuts - the
    safe direction. Matching is space-insensitive ("mm-hmm", "mmhmm" and
    "mm hmm" are the same phrase)."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t.split()) > BACKCHANNEL_MAX_WORDS:
        return False
    return t.replace(" ", "") in _BACKCHANNEL_COMPACT


def is_not_addressed(text: str) -> bool:
    """True if the speech reads as being for someone else (TALKING_OTHER).

    Full-Duplex-Bench v1.5: the desired behavior for talk-to-other and
    background speech is to HOLD THE FLOOR, not cut. We only claim it on
    high-precision cues so a real interruption is never swallowed:
      1. explicit not-you cue - "you guys", "everyone", "talk to you",
         "not to you";
      2. an address - a proper-noun-ish capitalized word (a name) either
         followed by a comma ("Maria, are you coming?") or preceded by a
         vocative ("hey John, ..."). Whisper capitalizes sentence-initial
         words, so a lone capitalized first word proves nothing - the comma
         or the vocative is what marks it as an address. Discourse markers
         ("Well,", "So,") are excluded so they keep cutting as normal turns.
    Runs on the RAW transcript (before the proofread pass may rewrite it)."""
    if not FD_HOLD_FLOOR:
        return False
    t = text.strip()
    if not t or len(t.split()) > 12:
        return False
    low = t.lower()
    for pat in _FD_NOT_YOU_PATTERNS:
        if pat.search(low):
            return True
    parts = re.split(r"[\s]+", t)
    # "hey John, ..." / "wait Maria -" : vocative + capitalized name.
    if len(parts) >= 2 and parts[0].lower().rstrip(".,!") in _FD_VOCATIVES:
        nxt = parts[1].strip(".,!?\"'()")
        if (
            nxt[:1].isupper()
            and nxt.isalpha()
            and len(nxt) >= 2
            and nxt.lower() not in _FD_AGENT_NAMES
        ):
            return True
    # "Maria, ..." : capitalized name + comma (an address, not a sentence
    # start - Whisper keeps the comma right after the name).
    first = parts[0]
    if first.endswith(","):
        name = first[:-1].strip(".,!?\"'()")
        if (
            name[:1].isupper()
            and name.isalpha()
            and len(name) >= 2
            and name.lower() not in _FD_DISCOURSE_MARKERS
            and name.lower() not in _FD_AGENT_NAMES
        ):
            return True
    return False


def classify_overlap(text: str) -> str:
    """Full-Duplex-Bench v1.5 behavior class for speech that overlapped the
    bot's own voice. Order matters: backchannel first (the cheapest, safest
    keep-talking case), then talk-to-other (hold floor), else interruption
    (cut and respond). Background speech rarely reaches this - the VAD
    loudness gate and the music-mode lyric guard drop it upstream."""
    if is_backchannel(text):
        return "backchannel"
    if is_not_addressed(text):
        return "talk_to_other"
    return "interruption"
