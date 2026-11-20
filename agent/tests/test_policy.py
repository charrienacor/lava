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


def test_confirm_gated_tools_declare_identity():
    """Approval binding is TOML policy: every confirm-gated tool declares
    `identity = [...]` — the argument keys its spoken "yes" binds to
    (tool_gateway._identity). Unmapped gated tools still work via the
    strict full-args fallback, but the known ones pin their bindings here
    so a TOML typo can't quietly loosen an approval."""
    expected = {
        ("github", "merge_pull_request"): {"owner", "repo", "pullNumber"},
        ("github", "create_or_update_file"): {"owner", "repo", "branch", "path"},
        ("github", "push_files"): {"owner", "repo", "branch"},
        ("github", "create_repository"): {"name", "org"},
        ("github", "delete_file"): {"owner", "repo", "branch", "path"},
        ("spotify", "spotify_add_tracks"): {"playlist_id", "uris"},
        ("spotify", "spotify_create_playlist"): {"name"},
    }
    for (server, action), keys in expected.items():
        data = tomllib.loads((TOOL_DIR / f"{server}.toml").read_text())
        tool = next(t for t in data["tool"] if t["name"] == action)
        assert set(tool.get("identity", [])) == keys, (
            f"{server}:{action} identity drifted from the pin"
        )


def test_identity_declared_only_on_gated_tools():
    """identity without confirm is dead policy — the gate never reads it."""
    for path in _tomls():
        for tool in tomllib.loads(path.read_text()).get("tool", []):
            if tool.get("identity"):
                assert tool.get("confirm"), (
                    f"{path.stem}:{tool['name']} declares identity but is "
                    f"not confirm-gated"
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
