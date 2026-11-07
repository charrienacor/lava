"""Persistent MCP connections to the ASR, TTS and tools servers, plus the
tool-loading and tool-execution helpers the rest of the bridge calls into."""

import asyncio
import contextlib
import json
import os
import time
from urllib.parse import urlsplit

from loguru import logger
from mcp import Client

from web.core.config import ASR_URL, FAST_MUSIC_COMMANDS, TOOLS_URL, TTS_URL
from web.core.state import STATE

# MCP plumbing (persistent connections to the three MCP servers)


class MCPTransportError(RuntimeError):
    """The MCP connection itself failed (socket drop, protocol error, server
    unreachable) - as opposed to a tool returning an error result. Only this
    exception is eligible for reconnect-and-retry in MCPConn.call(); tool
    errors are terminal, so a "skip track" whose error text happens to
    mention "closed" can never be executed twice."""


class MCPConn:
    """Persistent MCP client session held in one background task.

    A connection that drops after a successful start (backend crash, restart)
    is re-established every couple of seconds, so a revived MCP server is
    picked up without restarting the bridge. In-flight calls that hit the
    drop - a dead "stop the music" used to be exactly that - are retried once
    on a fresh connection."""

    RECONNECT_SECS = 2.0
    HEALTH_SECS = 10.0

    def __init__(self, url: str, name: str):
        self.url, self.name = url, name
        host = urlsplit(url)
        self._host = host.hostname or "127.0.0.1"
        self._port = host.port or (443 if host.scheme == "https" else 80)
        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._closing: asyncio.Event | None = None
        self._ready: asyncio.Future | None = None
        self._fail: BaseException | None = None

    async def start(self):
        async with self._start_lock:
            await self._kick()

    async def _kick(self, force: bool = False):
        """(Re)establish the session. force=True tears down a live task first.

        A plain start() returns instantly when a supervisor task exists, which
        once let callers "succeed" into a self._client=None state and crash on
        the next attribute access; force=True is for the retry path in call()."""
        if not force and self._task and not self._task.done():
            await self._ready
            return
        if force:
            self._closing.set()  # let the supervise loop exit
            task = self._task
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, 1.0)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    task.cancel()
        self._closing = asyncio.Event()
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._own())
        await self._ready

    async def _own(self):
        first = True
        try:
            while not self._closing.is_set():
                try:
                    async with Client(self.url, read_timeout_seconds=30) as client:
                        self._client = client
                        self._fail = None
                        if not self._ready.done():
                            self._ready.set_result(None)
                        await self._supervise(client)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    self._fail = e
                    if first and not self._ready.done():
                        # Never connected: surface it once (startup reports the
                        # error) instead of spinning forever.
                        self._ready.set_exception(
                            ConnectionError(
                                f"Cannot reach {self.name} MCP server "
                                f"at {self.url}: {e}"
                            )
                        )
                        return
                    logger.error(
                        f"[web] {self.name} MCP session ended: {e}; "
                        f"reconnecting in {self.RECONNECT_SECS:.0f}s"
                    )
                    await asyncio.sleep(self.RECONNECT_SECS)
                first = False
        except asyncio.CancelledError:
            pass
        finally:
            self._client = None
            self._task = None

    async def _supervise(self, client):
        """Wait for close, probing the backend port so a silent death is
        noticed (the MCP client only errors on the next call otherwise).

        The probe is a raw TCP connect: the old empty POST /mcp made every
        server fabricate and terminate an MCP transport every 10 s ("Created
        new transport ... 400 Bad Request / Terminating session" churn)."""
        closer = asyncio.create_task(self._closing.wait())
        try:
            while not self._closing.is_set():
                await asyncio.wait({closer}, timeout=self.HEALTH_SECS)
                if self._closing.is_set():
                    return
                try:
                    _, writer = await asyncio.open_connection(self._host, self._port)
                except OSError as e:
                    raise ConnectionError(
                        f"{self._host}:{self._port} not listening: {e}"
                    ) from e
                with contextlib.suppress(Exception):
                    writer.close()
                    await writer.wait_closed()
        finally:
            closer.cancel()

    async def _exchange(self, tool: str, args: dict):
        client = self._client
        try:
            result = await client.call_tool(tool, args)
        except Exception as e:
            # The client raised: socket drop, protocol error, server
            # unreachable. The connection itself failed, so call() may
            # reconnect and retry.
            raise MCPTransportError(f"{self.name}.{tool}: {e}") from e
        if result.is_error:
            # The tool ran and reported failure. Terminal: retrying would
            # execute it twice (skipping two tracks).
            raise RuntimeError(f"{self.name}.{tool} failed: {result.content}")
        try:
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)
        except Exception as e:
            # Garbage back over the wire - treat as a corrupt connection.
            raise MCPTransportError(
                f"{self.name}.{tool} returned an unparseable result: {e}"
            ) from e

    async def call(self, tool: str, args: dict):
        if not self._client:
            # Self-heal: the connection was never established (MCP was down at
            # bridge startup) or died. Try once now - raises if still down.
            try:
                await self.start()
            except Exception as e:
                reason = f" (last failure: {self._fail or e})"
                raise RuntimeError(
                    f"{self.name} MCP server not connected{reason}"
                ) from e
        try:
            return await self._exchange(tool, args)
        except MCPTransportError as e:
            # The connection itself died - reconnect once and re-issue the
            # call. Tool-result errors (plain RuntimeError) never reach
            # here, so a failed "skip track" is never skipped twice.
            logger.warning(
                f"[web] {self.name}.{tool} connection dropped "
                f"({str(e)[:80]}) - reconnecting once"
            )
            self._client = None
            try:
                async with self._start_lock:
                    await self._kick(force=True)
            except Exception as e2:
                raise RuntimeError(
                    f"{self.name} MCP server not connected "
                    f"(last failure: {e2 or self._fail})"
                ) from e
            if not self._client:
                raise RuntimeError(
                    f"{self.name} MCP server not connected (last failure: {self._fail})"
                )
            return await self._exchange(tool, args)

    async def close(self):
        if self._closing is None:
            return
        self._closing.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, 3)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    @property
    def connected(self) -> bool:
        return self._client is not None


asr_conn = MCPConn(ASR_URL, "asr")
tts_conn = MCPConn(TTS_URL, "tts")
tools_conn = MCPConn(TOOLS_URL, "tools")


async def load_tools():
    """Pull the tool schemas from the tools MCP server (retry until it is up)."""
    while True:
        try:
            await tools_conn.start()
            lt = await tools_conn._client.list_tools()
            STATE["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": t.input_schema
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in lt.tools
                if t.name not in ("pop_finished_timers", "music_state")
            ]
            logger.info(f"[web] loaded {len(STATE['tools'])} tools from tools MCP")
            if FAST_MUSIC_COMMANDS:
                logger.info("[web] fast music commands enabled (music mode)")
            return
        except Exception as e:
            logger.warning(f"[web] tools MCP not ready ({e}); retrying in 3 s")
            await asyncio.sleep(3)


TOOL_TIMEOUT_SECS = float(os.getenv("TOOL_TIMEOUT_SECS", "12"))


async def exec_tool(name: str, args: dict, notify=None) -> str:
    """Run one tools-MCP call. `notify` (optional) is an async callable
    invoked as notify(phase, name, args, result) with phase "start" before
    the call and "done" after it - the bridge uses it to push command
    notifications to the UI. A broken notify never affects the tool result."""
    t0 = time.perf_counter()
    if notify:
        try:
            await notify("start", name, args, None)
        except Exception:
            pass
    try:
        res = await asyncio.wait_for(tools_conn.call(name, args), TOOL_TIMEOUT_SECS)
        out = json.dumps(res) if not isinstance(res, str) else res
    except TimeoutError:
        out = json.dumps({"error": f"{name} timed out after {TOOL_TIMEOUT_SECS:.0f}s"})
    except Exception as e:
        out = json.dumps({"error": str(e)})
    logger.info(
        f"[web] tool {name}({json.dumps(args)}) -> {out[:120]} ({(time.perf_counter() - t0) * 1000:.0f} ms)"
    )
    if notify:
        try:
            await notify("done", name, args, out)
        except Exception:
            pass
    return out
