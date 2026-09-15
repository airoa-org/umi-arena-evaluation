"""check.py the way a contestant runs it: `uv run check.py --submission <dir>`,
against the built images, with the server started from the submission.

The dummy adapter passes; each broken adapter fails the check that names its
defect; real checkpoints pass when they are present. Skipped without `uv`, the
images, or a GPU. The lerobot fixtures live under `UMI_ARENA_HF_DIR` (default
`~/umi-arena-hf`) with a warmed Hub cache in its `home/`; the `--hf` case needs
network to huggingface.co.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.image, pytest.mark.gpu, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "check.py"
BROKEN = ROOT / "tests" / "broken"
HF_DIR = Path(os.environ.get("UMI_ARENA_HF_DIR", Path.home() / "umi-arena-hf"))
CKPT = Path(os.environ.get("UMI_ARENA_TEST_CHECKPOINT", "/nonexistent"))


@pytest.fixture(scope="session")
def uv() -> str:
    path = shutil.which("uv")
    if not path:
        pytest.skip("uv not installed; check.py is run through `uv run`")
    return path


def _check(uv, tmp_path, submission, image, *extra, calls=5):
    report = tmp_path / "report.json"
    r = subprocess.run(
        [uv, "run", "-q", str(CHECK), "--submission", str(submission), "--image", image,
         "--calls", str(calls), "--keepalive", "1", "--handshake-timeout", "5", "--report", str(report),
         "--env", "UMI_CHECK_SLEEP_S=3", *extra],
        capture_output=True, text=True, timeout=1500)
    data = json.loads(report.read_text())
    return r.returncode, data, r.stdout + r.stderr


def _status(data, name_prefix):
    hits = [c for c in data["checks"] if c["name"].startswith(name_prefix)]
    assert hits, f"no check named {name_prefix!r} in {[c['name'] for c in data['checks']]}"
    return hits[-1]["status"], hits[-1]["detail"]


def _fixture_submission(docker_tmp: Path, model: str) -> Path:
    """A Hub fixture plus the lerobot adapter, as test_hf_models assembles it."""
    src = HF_DIR / model
    if not (src / "config.json").is_file():
        pytest.skip(f"{model} not present in {HF_DIR}")
    d = docker_tmp / model
    d.mkdir(parents=True, exist_ok=True)
    for f in ("policy.py", "umi_arena.yaml"):
        shutil.copy(ROOT / "adapters" / "lerobot_hf" / f, d / f)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("."):
            (d / f.name).symlink_to(f)
    return d


# The submission symlinks into HF_DIR, so it is mounted at the same path; the backbone cache goes the intake route.
HF_ENV = ["--volume", f"{HF_DIR}:{HF_DIR}", "--hf-cache", str(HF_DIR / "home")]


# --- the pass case, no weights ------------------------------------------------------

def test_dummy_adapter_passes(uv, gpu_torch_image, tmp_path):
    code, data, out = _check(uv, tmp_path, ROOT / "adapters" / "dummy", gpu_torch_image)
    assert code == 0, out
    assert data["verdict"].startswith("PASS"), data["verdict"]
    assert data["load_s"] > 0 and data["timing"]["p50_ms"] > 0
    assert data["image_digest"].startswith("sha256:")


# --- each defect fails the check that names it -------------------------------------

@pytest.mark.parametrize("adapter,name,expect", [
    ("cols15", "call 0: server answered", "16 columns"),
    ("rows8", "call 0: server answered", "at least 16 rows"),
    ("nan", "call 0: server answered", "non-finite"),
    ("needs_center", "call 0: server answered", "observation.image.center"),
    ("prompt_cached", "prompt change: server answered", "prompt changed"),
])
def test_defects_are_reported_with_the_servers_message(uv, gpu_openpi_image, tmp_path, adapter, name, expect):
    """The runner rejects the chunk and sends its error text; the checker shows it."""
    code, data, _ = _check(uv, tmp_path, BROKEN / adapter, gpu_openpi_image)
    assert code == 1
    status, detail = _status(data, name)
    assert status == "fail" and expect in detail, detail


def test_a_call_longer_than_the_keepalive_drops_the_connection(uv, gpu_openpi_image, tmp_path):
    """3 s of inference against a 1 s keepalive: the connection closes mid-call,
    exactly what inference_node would suffer at 20 s."""
    code, data, _ = _check(uv, tmp_path, BROKEN / "slow", gpu_openpi_image)
    assert code == 1
    status, detail = _status(data, "call 0: connection stays open")
    assert status == "fail" and "closed" in detail, detail


def test_the_wrong_image_fails_at_load_with_the_entrypoints_message(uv, gpu_openpi_image, tmp_path):
    code, data, _ = _check(uv, tmp_path, ROOT / "adapters" / "dummy", gpu_openpi_image)  # dummy declares torch
    assert code == 1
    status, detail = _status(data, "runner comes up")
    assert status == "fail" and "umi-arena-torch" in detail, detail
    assert not subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True,
                              text=True).stdout.count("umi-check-"), "container left behind"


# --- real weights -------------------------------------------------------------------

def test_the_pi05_checkpoint_passes(uv, gpu_openpi_image, tmp_path):
    if not (CKPT / "params").is_dir() or not (CKPT / "policy.py").is_file():
        pytest.skip(f"no serveable pi0.5 submission at {CKPT}")
    code, data, out = _check(uv, tmp_path, CKPT, gpu_openpi_image, "--env", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.6")
    assert code == 0, out
    print(f"\npi0.5: load {data['load_s']}s, {data['timing']}")


@pytest.mark.parametrize("model", ["smolvla_yubi16", "groot_yubi16"])
def test_the_lerobot_fixtures_pass(uv, gpu_torch_image, docker_tmp, tmp_path, model):
    code, data, out = _check(uv, tmp_path, _fixture_submission(docker_tmp, model), gpu_torch_image, *HF_ENV)
    assert code == 0, out
    print(f"\n{model}: load {data['load_s']}s, {data['timing']}")


def test_hf_intake_fails_a_repo_without_an_adapter(uv, gpu_openpi_image, tmp_path):
    """--hf fetches lerobot/smolvla_base (no umi_arena.yaml, no policy.py): the
    checker warns about the missing runtime, then the runner refuses it."""
    report = tmp_path / "report.json"
    r = subprocess.run([uv, "run", "-q", str(CHECK), "--hf", "lerobot/smolvla_base", "--image", gpu_openpi_image,
                        "--calls", "3", "--report", str(report)],
                       capture_output=True, text=True, timeout=1500, env={**os.environ, "HF_HOME": str(HF_DIR / "home")})
    data = json.loads(report.read_text())
    assert r.returncode == 1
    assert _status(data, "umi_arena.yaml declares a runtime")[0] == "warn"
    status, detail = _status(data, "runner comes up")
    assert status == "fail" and "policy.py" in detail, detail
    assert data["submission"].startswith("lerobot/smolvla_base@")
