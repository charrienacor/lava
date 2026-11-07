"""Lava web bridge: HTTP/WebSocket routes, session ownership, and lifecycle.

Browser mic -> [ASR MCP :8001] -> Groq LLM + [tools MCP :8003] -> [TTS MCP :8002] -> browser

The heavy lifting lives in web/core/ (config, state, mcp_client, llm, vad,
speech, music, turn); this module wires the routes, owns each session, and runs
startup/shutdown plus the background loops (timer announcements, music-state
polling, idle watch).

Wire protocol (binary audio, low overhead):
  client -> server  binary frames: raw 16-bit LE mono PCM at 16 kHz, sent as
                    512-sample (32 ms) frames by the mic worklet. JSON text
                    frames carry control only ({"type": "ready"}).
  server -> client  binary frames: uint16 LE sample rate + 16-bit LE mono PCM,
                    one TTS chunk per frame. JSON text frames carry control
                    (text, transcript, barge, tool, done, error, silence, bye).

Run:  python -m web.server       (port 8080 by default)
"""

import asyncio
import json
import random
import secrets
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from loguru import logger

from mcp_servers import spotify
from mcp_servers.common import setup_logging
from web.core.config import (
    _GREETING,
    _IDLE_PROMPTS,
    ASR_APOLOGY,
    CITY,
    IDLE_CHECK_SECS,
    IDLE_PROMPT_SECS,
    LLM_APOLOGY,
    MUSIC_MODE_GRACE,
    MUSIC_POLL_IDLE_MAX,
    SPOTIFY_REDIRECT_URI,
    SYSTEM_PROMPT,
    WEB_HOST,
    WEB_PORT,
)
from web.core.mcp_client import asr_conn, exec_tool, load_tools, tools_conn, tts_conn
from web.core.notifications import describe, tool_frame
from web.core.speech import send, send_greeting, speak_text, synth_audio
from web.core.state import (
    CONVERSATIONS,
    SPEECH_CACHE,
    STATE,
    _prune_conversations,
    fd_summary,
)
from web.core.turn import handle_barge, process_utterance
from web.core.vad import BACKCHANNEL_CUT_MAX_SECS, VADState

# HTTP + WebSocket handlers


async def h_status(request: web.Request):
    return web.json_response(
        {
            "ok": True,
            "model": STATE["model"],
            "chain": STATE["chain"],
            "provider": "groq",
            "city": CITY,
            "tools": [t["function"]["name"] for t in STATE["tools"]],
            "sessions": STATE["sessions"],
            "music_playing": STATE["music_playing"],
            "backchannels": STATE["backchannels"],
            "barge_ins": STATE["barge_ins"],
            "fd_metrics": fd_summary(),
        }
    )


async def h_index(request: web.Request):
    return web.FileResponse(Path(__file__).parent / "index.html")


# Internal tools the bridge drives itself (poll loops) - never UI-runnable.
_RUN_BLOCKED = {"pop_finished_timers", "music_state"}


async def h_run(request: web.Request):
    """Execute one tool call directly, outside a voice turn (POST /api/run).

    The page's commands tray runs a row through here: the call takes the
    exact same exec_tool -> notifications pipeline a spoken command uses,
    so the classifier copy is the real one and the banner frames are the
    real ones - fanned out live over the WebSocket to every ready session.

    The response carries the same two frames (start/done) for tabs whose
    session is not ready yet (they render the outage locally instead),
    plus "broadcast" so the client knows which channel will show them."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "expected a JSON body"}, status=400)
    name = body.get("tool")
    args = body.get("args") if isinstance(body.get("args"), dict) else {}
    if not isinstance(name, str) or not name:
        return web.json_response({"ok": False, "error": "missing tool name"}, status=400)
    if name in _RUN_BLOCKED:
        return web.json_response({"ok": False, "error": f"{name} is internal"}, status=400)
    # Constrain to the tools the LLM itself may call (loaded from the tools
    # MCP by the startup task). Empty set = still loading; an unknown name
    # will then fail honestly at the tools MCP instead of here.
    allowed = {t["function"]["name"] for t in STATE["tools"]}
    if allowed and name not in allowed:
        return web.json_response({"ok": False, "error": f"unknown tool {name}"}, status=400)

    sent = {"n": 0}

    async def _notify(phase, tname, targs, tresult):
        frame = tool_frame(phase, tname, targs, tresult)
        for ws in list(STATE["live_sessions"]):
            s = STATE["by_ws"].get(ws)
            if not s or not s.get("ready"):
                continue
            try:
                await send(ws, frame)
                sent["n"] += 1
            except Exception:
                pass  # one dead client must never block the others

    result = await exec_tool(name, args, notify=_notify)
    err = describe(name, args, result)["err"]
    frames = [
        tool_frame("start", name, args, None),
        tool_frame("done", name, args, result),
    ]
    return web.json_response(
        {
            "ok": True,
            "status": "error" if err else "ok",
            "broadcast": sent["n"] > 0,
            "frames": frames,
            "result": result,
        }
    )


def _icon_handler(filename: str):
    """Serve the pre-rendered lava-orb icons (Safari ignores data-URI favicons)."""

    async def h(request: web.Request):
        return web.FileResponse(
            Path(__file__).parent / filename,
            headers={"Cache-Control": "no-cache"},
        )

    return h


async def h_spotify_auth(request: web.Request):
    """One-time Spotify link: bounce the user to Spotify's consent page."""
    if not spotify.configured():
        return web.Response(
            text="Spotify is not configured: add "
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET "
            "to .env and restart.",
            status=500,
        )
    # CSRF guard: bind the redirect back to THIS click (localhost only,
    # but a stale callback link in history should not mint a token).
    state = secrets.token_urlsafe(16)
    request.app["_spotify_state"] = state
    return web.HTTPFound(spotify.authorize_url(SPOTIFY_REDIRECT_URI, state))


async def h_spotify_callback(request: web.Request):
    code = request.query.get("code")
    err = request.query.get("error")
    if err or not code:
        return web.Response(
            text=f"Spotify authorization failed: {err or 'no code'}", status=400
        )
    expected = request.app.get("_spotify_state")
    if not expected or request.query.get("state") != expected:
        logger.warning("[web] Spotify callback state mismatch - ignoring")
        return web.Response(
            text="Spotify link failed (state mismatch) - try again from /spotify/auth.",
            status=400,
        )
    ok = await spotify.exchange_code(code, SPOTIFY_REDIRECT_URI)
    if not ok:
        logger.error("[web] Spotify token exchange failed")
        return web.Response(
            text="Token exchange failed - check the bridge logs.", status=502
        )
    logger.info("[web] Spotify linked - refresh token stored")
    return web.Response(
        text="<html><body style='font-family:sans-serif;background:#0a0706;"
        "color:#f2e9e2;display:flex;justify-content:center;align-items:center;"
        "height:100vh'><div><h2>Spotify linked &check;</h2>"
        "<p>You can close this tab and ask Lava to play something.</p>"
        "</div></body></html>",
        content_type="text/html",
    )


async def h_ws(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=25)
    await ws.prepare(request)

    client_id = request.query.get("cid") or None
    STATE["sessions"] += 1
    STATE["live_sessions"].add(ws)

    stored = CONVERSATIONS.get(client_id) if client_id else None
    prior = stored[0] if stored else None
    sess = {
        "client_id": client_id,
        "history": list(prior)
        if prior
        else [{"role": "system", "content": SYSTEM_PROMPT}],
        "vad": VADState(),
        "turn_lock": asyncio.Lock(),
        "turn_task": None,
        "ready": False,
        "greeting_sent": False,
        "bot_audio_until": 0.0,
        "last_voice": time.monotonic(),
        "last_ratio": 1.0,  # voiced ratio of the latest completed span
        # Deferred barge-in (Full-Duplex-Bench style): when speech starts over
        # the bot we record the onset but do NOT cut yet. The span is
        # transcribed once it closes; a pure backchannel ("mm-hmm") keeps the
        # bot talking, a real interruption cuts. See the ws loop below.
        "barge_onset": 0.0,  # monotonic time the barge hint fired (0 = none)
        "barge_bot_live": False,  # was a turn live (or audio playing) at onset
        "last_barge": 0.0,  # monotonic time of the last actual cut
        "prev_turn": None,  # task that was speaking when this span started
        # Full-Duplex-Bench v1.5 anchors: fd_cut_onset = barge onset (t_stop
        # reference, consumed by handle_barge); fd_cut_at = cut instant
        # (t_resp reference, consumed by the response turn's first audio).
        "fd_cut_onset": 0.0,
        "fd_cut_at": 0.0,
        # True while a deferred-overlap onset has ducked the client's
        # playback (the duck is released by unduck/barge).
        "ducked": False,
    }
    resumed = bool(prior)
    logger.info(f"[web] session started (total {STATE['sessions']}, resumed={resumed})")
    idle_task = asyncio.create_task(idle_watch(ws, sess))
    STATE["by_ws"][ws] = sess

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "ready":
                    sess["ready"] = True
                    sess["last_voice"] = time.monotonic()
                    logger.info("[web] client ready - mic live, greeting")
                    if not sess["greeting_sent"]:
                        sess["greeting_sent"] = True

                        sess["turn_task"] = asyncio.create_task(send_greeting(ws, sess))
                continue
            if msg.type != WSMsgType.BINARY:
                continue
            if not sess["ready"]:
                continue
            sess["vad"].feed(msg.data)

            if sess["vad"].barge_hint:
                sess["vad"].barge_hint = False
                turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
                # Deferred barge-in (Full-Duplex-Bench style): a barge onset
                # over the bot is not yet a cut. Record it and let the span
                # close; process_utterance then classifies the transcript -
                # a pure backchannel ("mm-hmm") keeps the bot talking, real
                # speech cuts. The watchdog below bounds how long a span may
                # stay open before we cut anyway.
                if (
                    turn_live or time.monotonic() < sess["bot_audio_until"]
                ) and not sess["barge_onset"]:
                    sess["barge_onset"] = time.monotonic()
                    sess["barge_bot_live"] = turn_live
                    # Duck now, decide later: speech over the bot is
                    # heard the moment it starts, so drop the playback
                    # gain immediately; when the span's transcript
                    # classifies as a backchannel / talk-to-other the
                    # gain comes back ("unduck"), a real interruption
                    # hard-cuts ("barge"). The client also auto-restores
                    # if neither message arrives.
                    sess["ducked"] = True
                    await send(ws, {"type": "duck"})
                    logger.debug(
                        f"[web] barge onset over the bot "
                        f"(cid={sess.get('client_id')}) - ducked, "
                        "cut deferred until the span classifies"
                    )
                continue
            # Watchdog: a span still open BACKCHANNEL_CUT_MAX_SECS after its
            # barge onset is real speech (words keep coming, the bot is being
            # talked over too long) - cut like a classic barge-in.
            if (
                sess["barge_onset"]
                and sess["vad"].utterance is None
                and time.monotonic() - sess["barge_onset"] > BACKCHANNEL_CUT_MAX_SECS
            ):
                sess["fd_cut_onset"] = sess["barge_onset"]  # t_stop anchor
                sess["barge_onset"] = 0.0
                turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
                if turn_live or time.monotonic() < sess["bot_audio_until"]:
                    STATE["barge_ins"] += 1
                    STATE["fd_behaviors"]["interruption"] += 1
                    logger.info(
                        "[web] barge-in: span open past the "
                        f"{BACKCHANNEL_CUT_MAX_SECS:.1f} s watchdog"
                    )
                    await handle_barge(ws, sess)
                continue
            if sess["vad"].utterance is None:
                continue
            utterance = sess["vad"].utterance
            sess["vad"].utterance = None
            sess["last_ratio"] = sess["vad"].last_ratio
            sess["last_voice"] = time.monotonic()
            # In a deferred overlap the classification (inside
            # process_utterance) must decide between backchannel and cut -
            # canceling the live turn here would cut before that verdict.
            # Hand the speaking task over via prev_turn: sess["turn_task"]
            # is about to become this span's own process task.
            deferred = bool(sess["barge_onset"])
            if deferred:
                # Disarm the watchdog NOW (not after ASR inside the task):
                # the span is consumed, so the watchdog's "span still open"
                # premise is false. Leaving barge_onset set until the task
                # runs let the watchdog double-fire the cut one frame later
                # and cancel the very task classifying this span.
                sess["fd_pending_onset"] = sess["barge_onset"]
                sess["barge_onset"] = 0.0
            sess["prev_turn"] = sess["turn_task"]
            if sess["turn_task"] and not sess["turn_task"].done() and not deferred:
                sess["turn_task"].cancel()
            sess["turn_task"] = asyncio.create_task(
                process_utterance(ws, sess, utterance)
            )
    except Exception as e:
        import traceback as _tb

        logger.error(f"[web] ws error: {e}\n{_tb.format_exc()}")
    finally:
        idle_task.cancel()
        STATE["live_sessions"].discard(ws)
        STATE["by_ws"].pop(ws, None)
        if sess["turn_task"] and not sess["turn_task"].done():
            sess["turn_task"].cancel()

        cid = sess.get("client_id")
        if cid:
            hist = sess["history"]
            if len(hist) > 1:
                CONVERSATIONS[cid] = (hist, time.time())
            else:
                CONVERSATIONS.pop(cid, None)
            _prune_conversations()
        STATE["sessions"] -= 1
        logger.info(f"[web] session ended ({STATE['sessions']} active)")
        if not ws.closed:
            await ws.close()
    return ws


async def idle_watch(ws: web.WebSocketResponse, sess: dict):
    """Check in on a quiet session, then close it if the silence continues.

    The browser streams raw PCM continuously - silence included - so presence
    means detected speech or an active turn, not incoming frames. After
    IDLE_PROMPT_SECS of quiet the agent asks if the user is still there
    (speaking over the check-in barges it like any other speech); if the
    silence reaches IDLE_CHECK_SECS the mic stops and the orb waits for a
    tap (history is restored by client id)."""
    prompted = False
    try:
        while not ws.closed:
            if VADState.music_mode():
                # Music listening is presence: never check in (and never time
                # out) while the room is playing something.
                prompted = False
                await asyncio.sleep(2)
                continue
            quiet = time.monotonic() - sess["last_voice"]
            turn_live = bool(sess["turn_task"] and not sess["turn_task"].done())
            # Still counting as present while the bot's own audio is queued or
            # playing (a long reply can outlast the turn task by tens of
            # seconds) - otherwise the agent check-ins over its own voice.
            playing = time.monotonic() < sess["bot_audio_until"]
            if turn_live or playing or quiet < IDLE_PROMPT_SECS:
                prompted = False
                await asyncio.sleep(1)
                continue
            if prompted:
                logger.info(
                    "[web] idle timeout - no answer to the check-in, closing session"
                )
                await send(ws, {"type": "bye"})
                return
            prompted = True
            prompt = random.choice(_IDLE_PROMPTS)
            logger.info(f"[web] idle {quiet:.0f}s - checking in")
            await send(ws, {"type": "text", "t": prompt})
            try:
                await speak_text(ws, sess, prompt, {})
            except Exception as e:
                logger.warning(f"[web] idle check-in TTS failed: {e}")
            await send(ws, {"type": "done", "reply": prompt, "stats": {}})
            await asyncio.sleep(max(1.0, IDLE_CHECK_SECS - IDLE_PROMPT_SECS))
    except asyncio.CancelledError:
        pass


async def poll_music_state():
    """Keep STATE's Spotify playback mirror fresh (drives music-aware VAD).

    music_state on the tools MCP is a cached read (refreshed from Spotify at
    most every ~4 s server-side), so polling at 1 Hz is nearly free while a
    session is live. Outside music we still poll for a short window after a
    music tool ran, to catch an external pause on the phone; long-idle
    sessions cost nothing."""
    if not tools_conn.connected or not STATE["live_sessions"]:
        return
    now = time.monotonic()
    playing = STATE["music_playing"]
    recent = now - STATE["music_seen_at"] < MUSIC_POLL_IDLE_MAX
    if not playing and not recent:
        return
    try:
        res = await tools_conn.call("music_state", {})
    except Exception:
        return
    is_playing = bool(res.get("is_playing")) if isinstance(res, dict) else False
    if is_playing:
        STATE["music_seen_at"] = now
        STATE["music_mode_until"] = now + MUSIC_MODE_GRACE
        for sess in STATE["by_ws"].values():
            sess["last_voice"] = max(sess["last_voice"], now)
    if is_playing != playing:
        logger.info(
            f"[web] spotify playback mirror: {'playing' if is_playing else 'stopped'}"
        )
        if not is_playing:
            # Silence the idle check-in for the music-mode grace window after
            # the song ends, so 'still there?' never fires over the outro.
            for sess in STATE["by_ws"].values():
                sess["last_voice"] = now + MUSIC_MODE_GRACE
    STATE["music_playing"] = is_playing


async def announce_timers():
    """Speak finished timers (proactive speech). Browser-started timers would
    otherwise finish silently - the web bridge has no other listener."""
    await asyncio.sleep(5)
    running = 0
    next_timer_poll = 0.0  # fire on the first tick that has a session
    while True:
        await asyncio.sleep(1.0)
        await poll_music_state()
        # Only drain finished timers when a session could actually announce
        # them: popping with nobody connected used to both burn an MCP round
        # trip and silently discard the announcement (the timer was gone by
        # the time the user came back).
        if not tools_conn.connected or not STATE["live_sessions"]:
            continue
        now = time.monotonic()
        if now < next_timer_poll:
            continue
        try:
            res = await tools_conn.call("pop_finished_timers", {})
        except Exception as e:
            # Tools MCP hiccup; the next tick retries (deadline unchanged).
            logger.debug(f"[web] timer poll failed: {e}")
            continue
        running = res.get("running", 0)
        # 1 Hz while a timer runs (finish lands within a second); 5 s
        # otherwise, which is the normal state for hours.
        next_timer_poll = now + (1.0 if running > 0 else 5.0)
        for t in res.get("finished", []):
            label = (t["label"] or "").strip() or "timer"
            # Avoid "your timer timer" when the label already says timer.
            tail = "" if label.lower().endswith("timer") else " timer"
            msg = f"Heads up, your {label}{tail} is done."
            logger.info(f"[web] timer finished: {t['label']}")
            for ws in list(STATE["live_sessions"]):
                await send(ws, {"type": "text", "t": msg})
                try:
                    sess = STATE["by_ws"].get(ws)
                    await speak_text(ws, sess, msg, {})
                    await send(ws, {"type": "done", "reply": msg, "stats": {}})
                except Exception as e:
                    # One dead client shouldn't stop the others hearing it.
                    logger.debug(f"[web] timer announce failed for a session: {e}")


async def _preseed_speech_cache():
    """Pre-synthesize the fixed lines so the greeting and error apologies
    play with no TTS wait at all."""
    for text in (_GREETING, ASR_APOLOGY, LLM_APOLOGY, *_IDLE_PROMPTS):
        try:
            pcm, sr = await synth_audio(text)
            SPEECH_CACHE[text] = (pcm, sr)
        except Exception as e:
            logger.warning(f"[web] pre-synth failed for {text!r}: {e}")


async def on_startup(app: web.Application):
    app["llm_task"] = asyncio.create_task(load_tools())
    app["timer_task"] = asyncio.create_task(announce_timers())
    async def _start_one(conn):
        try:
            await conn.start()
            logger.info(f"[web] connected to {conn.name} MCP at {conn.url}")
        except Exception as e:
            logger.error(f"[web] cannot reach {conn.name} MCP at {conn.url}: {e}")

    # Connect ASR and TTS in parallel: each start() awaits its own
    # supervisor's readiness, so the two handshakes overlap at boot
    # instead of serializing. Per-conn errors are logged, never raised.
    await asyncio.gather(_start_one(asr_conn), _start_one(tts_conn))
    # Only after the TTS connection is up - the preseed calls the synthesizer.
    app["preseed_task"] = asyncio.create_task(_preseed_speech_cache())


async def on_shutdown(app: web.Application):
    for conn in (asr_conn, tts_conn, tools_conn):
        await conn.close()
    await spotify.close()  # release the pooled Spotify client


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", h_index)
    app.router.add_get("/api/status", h_status)
    app.router.add_post("/api/run", h_run)
    app.router.add_get("/ws", h_ws)
    app.router.add_get("/favicon.png", _icon_handler("favicon.png"))
    app.router.add_get("/spotify/auth", h_spotify_auth)
    app.router.add_get("/callback", h_spotify_callback)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    setup_logging("web")
    web.run_app(make_app(), host=WEB_HOST, port=WEB_PORT)
