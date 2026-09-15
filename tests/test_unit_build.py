"""The build script must give every runtime the same base tag and protocol revision."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _build_calls(tmp_path, monkeypatch, ref=None):
    calls = tmp_path / "calls.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_CALLS", str(calls))
    monkeypatch.delenv("OPENPI_REF", raising=False)
    if ref is not None:
        monkeypatch.setenv("OPENPI_REF", ref)
    subprocess.run(["bash", str(ROOT / "docker/build.sh"), "review-test"], check=True)
    return [json.loads(line) for line in calls.read_text().splitlines()]


@pytest.mark.parametrize("override", [None, "a" * 40])
def test_runtime_builds_share_the_base_and_openpi_commit(tmp_path, monkeypatch, override):
    calls = _build_calls(tmp_path, monkeypatch, override)
    assert len(calls) == 3
    assert calls[0] == ["build", "-f", "docker/Dockerfile.base", "-t", "umi-arena-base:review-test", "."]
    refs = []
    for runtime, args in zip(("openpi", "torch"), calls[1:], strict=True):
        assert args[args.index("-f") + 1] == f"docker/Dockerfile.{runtime}"
        assert args[args.index("-t") + 1] == f"umi-arena-{runtime}:review-test"
        build_args = [args[i + 1] for i, arg in enumerate(args) if arg == "--build-arg"]
        assert "BASE=umi-arena-base:review-test" in build_args
        refs.append(next(arg.removeprefix("OPENPI_REF=") for arg in build_args if arg.startswith("OPENPI_REF=")))
    assert refs[0] == refs[1]
    assert re.fullmatch(r"[a-f0-9]{40}", refs[0])
    if override is not None:
        assert refs[0] == override


def test_checker_client_uses_the_default_image_commit(tmp_path, monkeypatch):
    calls = _build_calls(tmp_path, monkeypatch)
    image_ref = next(arg.removeprefix("OPENPI_REF=") for arg in calls[1] if arg.startswith("OPENPI_REF="))
    requirement = re.search(r"openpi-client @ git\+https://github.com/Physical-Intelligence/openpi\.git@([a-f0-9]{40})", (ROOT / "check.py").read_text())
    assert requirement is not None
    assert requirement.group(1) == image_ref
