"""Conversation flow: one utterance end-to-end (ASR -> LLM -> TTS) and the
barge-in handling that cuts the bot when the user talks over it."""

import asyncio
import base64
import json
import time

from aiohttp import web
from loguru import logger

from mcp_servers.common import pcm16_to_wav, trim_silence
from web.core.config import (
    ASR_APOLOGY,
    LLM_APOLOGY,
    SAMPLE_RATE,
    TURN_DEADLINE_SECS,
)
from web.core.llm import ASR_CORRECT_GRACE, _correct_asr, llm_stream
from web.core.mcp_client import asr_conn
from web.core.music import MUSIC_NOTE, maybe_music_command, run_music_command
from web.core.notifications import tool_frame
from web.core.speech import TEXT_FLUSH_SECS, Speaker, send, speak_text, split_sentences
from web.core.spelling import handle_spelling
from web.core.state import STATE, _fd_record, _trim_history, done_stats
from web.core.vad import VADState, _is_skip, classify_overlap


async def run_turn(
    ws: web.WebSocketResponse,
    sess: dict,
    user_text: str,
    stats: dict,
    music_note: bool = False,
):
    """ASR -> LLM(+tools) -> TTS for one utterance. Bounded by a hard deadline."""
    deadline = time.monotonic() + TURN_DEADLINE_SECS
    history = sess["history"]
    if music_note:
        # Transient context for THIS turn only (swept right after) so the LLM
        # knows the room noise it just heard may be the music itself.
        history.append({"role": "system", "content": MUSIC_NOTE})
    history.append({"role": "user", "content": user_text})
    reply_parts: list[str] = []
    speaker = Speaker(ws, sess, stats)
    speaker.start()
    llm_t0 = time.perf_counter()
    try:
        buf = ""
        spoken_any = False
        # LLM deltas are batched into one text frame per flush instead of one
        # JSON message per token: a token-per-frame stream made the browser
        # JSON-parse and reflow its chat bubble dozens of times per reply,
        # stealing main-thread time from audio scheduling while the bot talks.
        text_out: list[str] = []
        last_text_flush = time.perf_counter()

        async def flush_text():
            nonlocal last_text_flush
            if text_out:
                await send(ws, {"type": "text", "t": "".join(text_out)})
                text_out.clear()
                last_text_flush = time.perf_counter()

        async def _notify(phase: str, name: str, args: dict, result: str | None):
            """Push a command-notification frame to the UI as tools run.
            `phase` is "start" (optimistic) or "done" (final, with result).
            The server computes the display strings; the client only renders
            and collapses repeats by key."""
            await send(ws, tool_frame(phase, name, args, result))

        async for kind, payload in llm_stream(history, notify=_notify):
            if time.monotonic() > deadline:
                raise TimeoutError("turn deadline exceeded")
            if kind == "tool_call":
                # The tool is about to run: the optimistic toast already went
                # out via _notify("start"); nothing more to send here.
                await flush_text()
                continue
            buf += payload
            text_out.append(payload)
            while True:
                chunks, buf = split_sentences(buf, allow_clause=not spoken_any)
                if not chunks:
                    break
                for s in chunks:
                    s = s.strip()
                    if s and not _is_skip(s):
                        reply_parts.append(s)
                        # Show the sentence's text before its audio starts.
                        await flush_text()
                        await speaker.enqueue(s)
                        spoken_any = True
            if time.perf_counter() - last_text_flush >= TEXT_FLUSH_SECS:
                await flush_text()
        await flush_text()
        if buf.strip() and not _is_skip(buf):
            reply_parts.append(buf.strip())
            await speaker.enqueue(buf.strip())
        stats["llm_ms"] = (time.perf_counter() - llm_t0) * 1000
        await speaker.finish()
        reply = " ".join(reply_parts).strip()
        if reply:
            if music_note:
                logger.info(
                    f"[web] music turn ACCEPTED (clear question/request, "
                    f"answered): {user_text!r}"
                )
            history.append({"role": "assistant", "content": reply})
        else:
            if music_note:
                logger.info(
                    f"[web] music turn SKIPPED (LLM judged it the music, not "
                    f"addressed to me): {user_text!r}"
                )
            if history and history[-1].get("role") == "user":
                history.pop()

        _trim_history(history)
        await send(ws, {"type": "done", "reply": reply, "stats": done_stats(stats)})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[web] turn failed: {e}")
        # Cut anything still queued before apologizing, so the apology never
        # interleaves with leftover sentence audio.
        await speaker.aclose()
        try:
            await speak_text(ws, sess, LLM_APOLOGY, stats)
        except Exception as e:
            # The apology is best-effort; the error message below still ships.
            logger.debug(f"[web] apology TTS failed: {e}")
        await send(ws, {"type": "error", "message": str(e)[:200], "apologized": True})
    finally:
        await speaker.aclose()
        # A turn that never spoke (SKIP, error) leaves the t_resp anchor
        # unconsumed - clear it so it can't bleed into the next turn.
        sess["fd_cut_at"] = 0.0
        # The music note is per-turn context, never a kept memory: without the
        # sweep it would multiply across turns inside the trimmed window.
        for i in range(len(history) - 1, 0, -1):
            m = history[i]
            if m.get("role") == "system" and m.get("content") == MUSIC_NOTE:
                del history[i]


async def process_utterance(ws: web.WebSocketResponse, sess: dict, pcm: bytes):
    """One complete utterance: transcribe, then run the LLM/TTS turn.

    ASR and the overlap classification run BEFORE the turn lock: while a
    possible backchannel is being transcribed, the previous turn keeps
    talking (the lock is taken only once a real turn is committed)."""
    stats = {"turn_t0": time.perf_counter()}
    t0 = time.perf_counter()
    # Spans dug out around music/TV carry seconds of quiet tail; trimming
    # cuts upload size and helps Whisper's quality gate. trim_silence
    # returns None when there is nothing above the audio floor at all.
    trimmed = trim_silence(pcm, SAMPLE_RATE)
    if trimmed is None:
        logger.info("[web] utterance is all silence - skipped ASR")
        sess["barge_onset"] = 0.0
        await send(ws, {"type": "silence"})
        return
    if len(trimmed) < len(pcm):
        pcm = trimmed
    # Tell the client we are working NOW, before the (sometimes slow) ASR
    # round trip: without this the orb sat on "listening" for the whole
    # transcription and looked deaf. The browser ignores it while bot audio
    # is playing, so a backchannel over the bot never flickers the state.
    await send(ws, {"type": "processing"})
    try:
        res = await asr_conn.call(
            "transcribe",
            {
                "audio_wav_b64": base64.b64encode(
                    pcm16_to_wav(pcm, SAMPLE_RATE)
                ).decode("ascii"),
            },
        )
        if isinstance(res, dict) and res.get("error"):
            # The ASR tool reports backend failures in-band (it catches its
            # own exceptions so one bad upload can't kill the MCP session);
            # route them through the same failure path as a transport drop.
            raise RuntimeError(str(res["error"])[:200])
    except Exception as e:
        logger.error(f"[web] ASR failed: {e}")
        sess["barge_onset"] = 0.0
        if sess.pop("ducked", False):
            # The user spoke over the bot and their words are lost - cut the
            # bot like a confirmed interruption, so the apology isn't layered
            # over the very audio they were trying to stop.
            await handle_barge(ws, sess)
        try:
            await speak_text(ws, sess, ASR_APOLOGY, stats)
        except Exception as e:
            # Best-effort apology; the error message below still ships.
            logger.debug(f"[web] ASR apology TTS failed: {e}")
        await send(ws, {"type": "error", "message": f"ASR failed: {str(e)[:200]}"})
        return
    stats["asr_ms"] = (time.perf_counter() - t0) * 1000
    text = res.get("text", "").strip()
    logger.info(f"[web] ASR ({stats['asr_ms']:.0f} ms): {text!r}")
    music_mode = VADState.music_mode()

    # Overlap classification (deferred barge-in, Full-Duplex-Bench v1.5):
    # this span started over the bot. Decide FIRST, on the raw transcript,
    # which of the four overlap scenarios this is - before the transcript
    # message (the browser cuts its playback on that event) and before the
    # proofread pass:
    #   backchannel    ("mm-hmm")        -> keep talking, no cut
    #   talk_to_other  ("hey John, ...") -> hold the floor, no cut
    #   interruption   (real new intent) -> cut now, respond to the new text
    # A real interruption is already bounded upstream by the
    # BACKCHANNEL_CUT_MAX_SECS watchdog, which cuts if the span stays open.
    # The onset is carried in fd_pending_onset: the ws loop disarms
    # barge_onset at span consumption (so the watchdog can't double-fire
    # while this task is classifying) and hands the timestamp over here.
    onset = sess.pop("fd_pending_onset", 0.0) or sess.get("barge_onset")
    if onset:
        sess["barge_onset"] = 0.0
        old = sess.get("prev_turn")
        sess["prev_turn"] = None
        turn_live = bool(old and not old.done())
        if turn_live or time.monotonic() < sess["bot_audio_until"]:
            cls = classify_overlap(text)
            if cls == "backchannel":
                STATE["backchannels"] += 1
                STATE["fd_behaviors"]["backchannel"] += 1
                logger.info(
                    f"[web] overlap was a backchannel "
                    f"({time.monotonic() - onset:.2f} s) - bot keeps "
                    f"talking: {text!r}"
                )
            elif cls == "talk_to_other":
                STATE["fd_behaviors"]["talk_to_other"] += 1
                logger.info(
                    f"[web] overlap was talk-to-other "
                    f"({time.monotonic() - onset:.2f} s) - holding "
                    f"the floor: {text!r}"
                )
                # Same as a backchannel: the still-speaking task goes back in
                # charge so any later real interruption can cut it normally.
                # No client message at all - the bot keeps talking and the
                # fragment leaves no trace in the UI.
                if turn_live:
                    sess["turn_task"] = old
                if sess.pop("ducked", False):
                    await send(ws, {"type": "unduck"})
                return
            else:
                STATE["barge_ins"] += 1
                STATE["fd_behaviors"]["interruption"] += 1
                logger.info(
                    f"[web] barge-in confirmed "
                    f"({time.monotonic() - onset:.2f} s span): {text!r}"
                )
                sess["fd_cut_onset"] = onset  # t_stop anchor (FDB v1.5)
                await handle_barge(ws, sess, old)
                # Surface this turn's t_stop (recorded by handle_barge) in the
                # response turn's done message.
                stats["fd_stop_ms"] = sess.pop("fd_last_stop_ms", None)

    if not text:
        if sess.pop("ducked", False):
            await send(ws, {"type": "unduck"})
        await send(ws, {"type": "silence"})
        return
    # Show the transcript the moment ASR lands - holding it for the proofread
    # made every turn feel a beat slower even when the fix was a no-op.
    sess["last_voice"] = time.monotonic()
    await send(ws, {"type": "transcript", "text": text})

    # Voice spelling (web/core/spelling.py): decode spelled-out letters
    # ("L-A-V-A", "Lima Alpha Victor Alpha") into plain text before the
    # proofread or the LLM sees them - detection runs on the RAW transcript
    # because the proofreader would happily rewrite a letter run into a
    # "word". A decoded sentence is canonical, so it skips the proofread
    # pass; the reconstructed text then flows through the normal pipeline
    # (music fast path or LLM tools) unchanged.
    spelling_decoded = False
    outcome = await handle_spelling(ws, sess, text, stats)
    if outcome.kind == "done":
        return
    if outcome.decoded:
        spelling_decoded = True
        text = outcome.text
        await send(ws, {"type": "transcript_fix", "text": text})

    # Repair obvious mis-hearings (esp. proper nouns) using recent context.
    # Capped at ASR_CORRECT_GRACE: if the proofread hasn't answered by then,
    # the turn starts on the raw text instead of waiting out the full
    # ASR_CORRECT_TIMEOUT, so a throttled corrector costs ~nothing. Skipped
    # while music plays: those turns are commands (fast path, no repairing
    # needed) or lyric noise (dropped below) - the proofread would only add
    # its whole round-trip to the latency. Also skipped when Whisper reports
    # itself confident (see ASR_PROOFREAD_LOGPROB_MAX): a clean transcript
    # needs no repair and the pass would be pure dead air. And skipped when
    # the spelling decoder just rewrote the text: that output is canonical
    # and a "correction" could only mangle the exact letters.
    if not music_mode and not spelling_decoded:
        try:
            fixed = await _correct_asr(
                text,
                sess["history"],
                stats,
                timeout=ASR_CORRECT_GRACE,
                avg_logprob=res.get("avg_logprob"),
            )
            if fixed and fixed != text:
                logger.info(f"[web] transcript fixed: {text!r} -> {fixed!r}")
                text = fixed
                await send(ws, {"type": "transcript_fix", "text": fixed})
        except Exception as ce:
            logger.warning(f"[web] ASR correction failed, using raw: {ce}")
    try:
        await asyncio.wait_for(sess["turn_lock"].acquire(), timeout=8.0)
    except TimeoutError:
        logger.info("[web] dropping utterance (previous turn still running)")
        return
    try:
        # 1) Deterministic music commands: run directly, no LLM.
        cmd = maybe_music_command(text)
        if cmd:
            await run_music_command(ws, sess, text, stats, cmd)
            return
        # 2) Phantom-lyrics guard: while music plays, a sparsely-voiced span
        #    is almost always the song itself. The fast path above already
        #    earned real commands through; anything else here is noise.
        if music_mode and sess.get("last_ratio", 1.0) < VADState.ratio_floor(True):
            logger.info(
                f"[web] music turn REJECTED (sparse-voiced span, likely the "
                f"music itself; {sess.get('last_ratio', 0):.0%} voiced): {text!r}"
            )
            if sess.pop("ducked", False):
                await send(ws, {"type": "unduck"})
            await send(ws, {"type": "silence"})
            return
        await run_turn(ws, sess, text, stats, music_note=music_mode)
    finally:
        sess["turn_lock"].release()


async def handle_barge(ws: web.WebSocketResponse, sess: dict, task=None):
    """Confirmed real overlap: cut the bot now.

    `task` pins the turn task to cancel when the caller holds the
    authoritative reference (deferred classification) - defaulting to the
    session's current turn task keeps the classic watchdog path working."""
    turn = task if task is not None else sess["turn_task"]
    turn_live = bool(turn and not turn.done())
    if turn_live:
        turn.cancel()
        try:
            await turn
        except (asyncio.CancelledError, Exception) as e:
            # The turn is being cut on purpose; log only unexpected deaths.
            if not isinstance(e, asyncio.CancelledError):
                logger.debug(f"[web] cancelled turn raised: {e}")
    if task is None:
        sess["turn_task"] = None
    # Playback is being cut client-side; drop the schedule.
    sess["bot_audio_until"] = 0.0
    # The client restores its playback gain on the barge message itself.
    sess["ducked"] = False
    # Full-Duplex-Bench v1.5: t_stop = barge onset -> the moment the bot
    # actually stops (the "barge" message is what cuts the client's audio).
    onset = sess.get("fd_cut_onset")
    if onset:
        sess["fd_cut_onset"] = 0.0
        stop_ms = (time.monotonic() - onset) * 1000
        sess["fd_last_stop_ms"] = stop_ms
        _fd_record("fd_stop_ms", stop_ms)
    # t_resp anchor: the response turn's first audio (recorded in
    # Speaker._stream_worker) measures how fast the agent gets back to
    # speaking after yielding.
    sess["fd_cut_at"] = time.monotonic()
    await send(ws, {"type": "barge"})
    logger.debug(
        f"[web] barge message delivered (cid={sess.get('client_id')}, "
        f"ws closed: {ws.closed})"
    )
    if turn_live:
        logger.info("[web] barge-in: user talking over the bot")
    else:
        logger.debug("[web] barge-in: no live turn, clearing client playback")
