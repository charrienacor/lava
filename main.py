"""Launch the voice agent — same CLI as `python agent/pipeline.py`.

    python main.py --transport websocket   # RPi local client
    python main.py --check-mcp             # connect to every MCP server, exit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

from pipeline import cli  # noqa: E402

cli()
