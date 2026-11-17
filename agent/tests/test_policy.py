"""Policy consistency: TOMLs, registry, dependency manifests must agree."""

import json
import tomllib
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1] / "tools"
SERVERS_JSON = Path(__file__).resolve().parents[2] / "mcp" / "mcp_servers.json"
ROOT = Path(__file__).resolve().parents[2]


def _tomls():
    return sorted(TOOL_DIR.glob("*.toml"))


def test_every_toml_declares_its_server_name():
    for path in _tomls():
        data = tomllib.loads(path.read_text())
        assert data["server"]["name"] == path.stem, f"{path.name}: server.name mismatch"


def test_confirm_implies_enabled():
    for path in _tomls():
        for tool in tomllib.loads(path.read_text()).get("tool", []):
            if tool.get("confirm"):
                assert tool.get("enabled"), (
                    f"{path.stem}:{tool['name']} is confirm-gated but disabled"
                )


def test_registry_matches_tomls():
    registry = set(json.loads(SERVERS_JSON.read_text())["mcpServers"])
    assert registry == {p.stem for p in _tomls()}


def test_pyproject_matches_requirements():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    def name_of(spec: str) -> str:
        return spec.split(";")[0].split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip().lower()

    py_names = {name_of(d) for d in pyproject["project"]["dependencies"]}
    req_names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            req_names.add(name_of(line))
    assert py_names == req_names, f"drift: only-py={py_names - req_names}, only-req={req_names - py_names}"
