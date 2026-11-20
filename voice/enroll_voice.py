"""Enroll the owner's voiceprint for the speaker-gated ASR server.

Takes WAV files of the user speaking (2-3 different samples, ~5-10 s each,
any recorder app), sends each to the `enroll` tool on the asr MCP server,
and averages them into data/voiceprint.json. The owner's name comes from
data/profile.yaml. Until enrolled, the ASR gate is fail-open.

    .venv/bin/python voice/enroll_voice.py sample1.wav [sample2.wav ...]
"""

import asyncio
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# mcp_client.py lives in agent/ — this script's own folder (voice/) is what
# Python puts on sys.path when run as a script, so agent/ is added beside it.
sys.path.insert(0, str(ROOT / "agent"))

from dotenv import load_dotenv  # noqa: E402  (.env must load before clients)

load_dotenv(ROOT / ".env", override=True)

from mcp_client import build_mcp_clients  # noqa: E402


async def main(wav_paths: list[Path]) -> None:
    if not wav_paths:
        sys.exit(
            "usage: .venv/bin/python voice/enroll_voice.py sample1.wav [sample2.wav ...]"
        )
    missing = [p for p in wav_paths if not p.exists()]
    if missing:
        sys.exit(f"not found: {', '.join(map(str, missing))}")

    asr = build_mcp_clients().get("asr")
    if asr is None:
        sys.exit("could not build the 'asr' MCP client (check mcp_servers.json/.env)")

    # One session for all samples (each _open_session spawns the server).
    await asr.start()
    session = await asr._open_session(as_owner=False)
    try:
        for path in wav_paths:
            audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            result = await session.call_tool("enroll", {"audio_b64": audio_b64})
            print(f"{path.name}: {result.content[0].text}")
    finally:
        await asr._drop_session(session)
        await asr.close()


if __name__ == "__main__":
    asyncio.run(main([Path(arg) for arg in sys.argv[1:]]))
