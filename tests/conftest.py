"""Shared fixtures.

The suite is layered so it stays useful before the image exists:

  unit      pure Python, no image, no GPU        (always runs)
  protocol  runner over a real websocket         (always runs)
  image     needs umi-arena-openpi / umi-arena-torch (skipped if absent)
  gpu       needs the image *and* a real GPU     (skipped if absent)
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGES = {
    "openpi": os.environ.get("UMI_ARENA_IMAGE_OPENPI", "umi-arena-openpi:dev"),
    "torch": os.environ.get("UMI_ARENA_IMAGE_TORCH", "umi-arena-torch:dev"),
}

sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def runner_module():
    return _load("umi_arena_runner_under_test", ROOT / "runner.py")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


def _docker() -> str | None:
    return shutil.which("docker")


def _image_exists(image: str) -> bool:
    if not _docker():
        return False
    return subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(
        ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


@pytest.fixture(scope="session")
def openpi_image() -> str:
    if not _image_exists(IMAGES["openpi"]):
        pytest.skip(f"image {IMAGES['openpi']} not built")
    return IMAGES["openpi"]


@pytest.fixture(scope="session")
def torch_image() -> str:
    if not _image_exists(IMAGES["torch"]):
        pytest.skip(f"image {IMAGES['torch']} not built")
    return IMAGES["torch"]


@pytest.fixture(scope="session")
def gpu_openpi_image(openpi_image: str) -> str:
    if not _gpu_available():
        pytest.skip("no GPU on this host")
    return openpi_image


@pytest.fixture(scope="session")
def gpu_torch_image(torch_image: str) -> str:
    if not _gpu_available():
        pytest.skip("no GPU on this host")
    return torch_image


# Docker on this host may be a snap, whose confinement hides /tmp and any
# dot-directory under $HOME. Bind mounts have to come from somewhere the daemon
# can actually read, so tests that mount use this instead of pytest's tmp_path.
_DOCKER_TMP_ROOT = Path(
    os.environ.get("UMI_ARENA_TEST_TMP", Path.home() / "umi-arena-test-tmp"))


@pytest.fixture(scope="session")
def docker_tmp_root() -> Path:
    _DOCKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    yield _DOCKER_TMP_ROOT
    shutil.rmtree(_DOCKER_TMP_ROOT, ignore_errors=True)


@pytest.fixture
def docker_tmp(docker_tmp_root: Path, request) -> Path:
    """A scratch directory the docker daemon can bind-mount."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.node.name)
    d = docker_tmp_root / safe
    # Containers can leave root-owned files (e.g. __pycache__) behind, which
    # rmtree cannot clear as this user; tolerate that rather than erroring out.
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def docker_run(image: str, script: str, *, gpu: bool = False, mounts=(), env=(), timeout=600):
    """Run a bash snippet in the image, bypassing the entrypoint."""
    cmd = ["docker", "run", "--rm", "--entrypoint", "bash"]
    if gpu:
        # --gpus needs the NVIDIA container toolkit, not a runtime registered as "nvidia" in daemon.json.
        cmd += ["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
    for src, dst in mounts:
        # `src` may carry docker mount options, e.g. "/path:ro".
        cmd += ["-v", f"{src}:{dst}" if ":" not in src else
                f"{src.rsplit(':', 1)[0]}:{dst}:{src.rsplit(':', 1)[1]}"]
    for k, v in env:
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, "-lc", script]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
