import sys
from pathlib import Path

# Make agent/ importable (tests live in agent/tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

# .env so build_mcp_clients()/create_llm()-adjacent code sees real values.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
