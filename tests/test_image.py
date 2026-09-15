"""Tests that need the built images (no GPU)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import docker_run

pytestmark = pytest.mark.image

ROOT = Path(__file__).resolve().parents[1]


# --- what each image contains ------------------------------------------------

def test_openpi_image_carries_only_its_venv(openpi_image):
    r = docker_run(openpi_image, "ls /opt/venv && echo runtime=$UMI_ARENA_RUNTIME")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["openpi", "runtime=openpi"], r.stdout


def test_torch_image_carries_only_its_venv(torch_image):
    r = docker_run(torch_image, "ls /opt/venv && echo runtime=$UMI_ARENA_RUNTIME")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["torch", "runtime=torch"], r.stdout


def test_openpi_venv_lerobot_has_no_policy_models(openpi_image):
    """Why the runtimes are separate images: openpi pins a lerobot that ships
    only dataset and normalisation plumbing. If openpi ever pins a lerobot that
    does ship models, one image might suffice and this test should be revisited."""
    r = docker_run(openpi_image, "/opt/venv/openpi/bin/python -c "
                                 "'from lerobot.policies.factory import get_policy_class' "
                                 "2>&1 | tail -1")
    assert "Error" in r.stdout or r.returncode != 0, (
        "openpi's lerobot now exposes the policy factory: " + r.stdout)


def test_torch_venv_can_resolve_policy_classes(torch_image):
    r = docker_run(torch_image, "/opt/venv/torch/bin/python -c '"
                                "from lerobot.policies.factory import get_policy_class; "
                                "print(get_policy_class(\"smolvla\").__name__)'")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SmolVLA" in r.stdout


@pytest.mark.parametrize("module", ["torch", "lerobot", "transformers",
                                    "openpi_client.msgpack_numpy", "websockets"])
def test_module_imports_in_the_torch_venv(torch_image, module):
    r = docker_run(torch_image, f"/opt/venv/torch/bin/python -c 'import {module}'")
    assert r.returncode == 0, f"{module}: {r.stdout}\n{r.stderr}"


@pytest.mark.parametrize("module", [
    "openpi", "openpi.transforms", "openpi.training.config",
    "openpi.policies.policy_config", "openpi_client.msgpack_numpy",
    "jax", "torch", "lerobot", "websockets", "numpy",
])
def test_module_imports_in_the_openpi_venv(openpi_image, module):
    r = docker_run(openpi_image, f"/opt/venv/openpi/bin/python -c 'import {module}'")
    assert r.returncode == 0, f"{module}: {r.stdout}\n{r.stderr}"


def test_cv2_imports(openpi_image):
    """Regression: the stock robot-deploy-pi0 image has no libGL.so.1, so an
    adapter that imports cv2 fails on load there."""
    r = docker_run(openpi_image, "/opt/venv/openpi/bin/python -c 'import cv2; print(cv2.__version__)'")
    assert r.returncode == 0, r.stdout + r.stderr


def test_yubi_transforms_are_importable(openpi_image):
    """The YUBI transforms ship in umi_arena and import against upstream openpi."""
    r = docker_run(openpi_image, "/opt/venv/openpi/bin/python -c "
                                 "'from umi_arena.yubi import transforms as y; "
                                 "from umi_arena.yubi.config import LeRobotYubiDataConfig; "
                                 "print(sorted(y.make_yubi_example()))'")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "observation.image.left" in r.stdout


@pytest.mark.slow
def test_host_skipped_suites_run_in_the_openpi_venv(openpi_image):
    """test_protocol.py and test_unit_yubi.py skip on a host without websockets/openpi; the venv has both."""
    r = docker_run(openpi_image, "cd /work && /opt/venv/openpi/bin/python -m pytest "
                          "tests/test_protocol.py tests/test_unit_yubi.py -q -p no:cacheprovider -o addopts=''",
                   mounts=[(f"{ROOT}:ro", "/work")], timeout=600)
    output = r.stdout[-3000:] + r.stderr[-1000:]
    assert r.returncode == 0, output
    assert " passed" in r.stdout, output
    assert " skipped" not in r.stdout, output


@pytest.mark.parametrize("fixture,venv", [("openpi_image", "openpi"), ("torch_image", "torch")])
def test_runner_and_helpers_are_installed(request, fixture, venv):
    image = request.getfixturevalue(fixture)
    r = docker_run(image, f"test -f /opt/umi_arena/runner.py && "
                          f"/opt/venv/{venv}/bin/python -c "
                          "'from umi_arena.openpi_yubi import discover_asset_id; print(\"ok\")'")
    assert r.returncode == 0, r.stdout + r.stderr


def test_jax_is_pinned_to_the_expected_version(openpi_image):
    r = docker_run(openpi_image, "/opt/venv/openpi/bin/python -c 'import jax; print(jax.__version__)'")
    assert r.stdout.strip() == "0.5.3", r.stdout


def test_openpi_is_upstream_at_the_pinned_commit(openpi_image):
    r = docker_run(openpi_image,
                   "git -C /opt/src/openpi remote get-url origin && "
                   "git -C /opt/src/openpi rev-parse HEAD && printenv OPENPI_REF && "
                   "/opt/venv/openpi/bin/python -c 'import openpi; print(openpi.__file__)'")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 4, r.stdout
    assert lines[0] == "https://github.com/Physical-Intelligence/openpi.git", r.stdout
    assert lines[1] == lines[2], r.stdout
    assert lines[3] == "/opt/src/openpi/src/openpi/__init__.py", r.stdout


def test_openpi_dependency_overrides_match_upstream(openpi_image):
    script = """
from importlib.metadata import version
from pathlib import Path
import tomllib

project = tomllib.loads(Path('/opt/src/openpi/pyproject.toml').read_text())
for requirement in project['tool']['uv']['override-dependencies']:
    package, expected = requirement.split('==')
    actual = version(package)
    assert actual == expected, f'{package}: expected {expected}, got {actual}'
"""
    r = docker_run(openpi_image, f"/opt/venv/openpi/bin/python - <<'PY'\n{script}\nPY")
    assert r.returncode == 0, r.stdout + r.stderr


# --- entrypoint behaviour ----------------------------------------------------

def _entrypoint(image, submission_dir=None, extra=()):
    cmd = ["docker", "run", "--rm"]
    if submission_dir is not None:
        cmd += ["-v", f"{submission_dir}:/submission:ro"]
    cmd += list(extra) + [image]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def _submission(docker_tmp: Path, name: str, yaml: str | None) -> Path:
    d = docker_tmp / name
    d.mkdir()
    if yaml is not None:
        (d / "umi_arena.yaml").write_text(yaml)
    (d / "policy.py").write_text(
        "class Policy:\n"
        "    def __init__(self, checkpoint_dir): raise SystemExit(17)\n")
    return d


def test_no_submission_mounted_fails_clearly(openpi_image):
    r = _entrypoint(openpi_image)
    assert r.returncode == 2
    assert "no submission" in (r.stdout + r.stderr).lower()


def test_missing_runtime_defaults_to_openpi(openpi_image, docker_tmp):
    r = _entrypoint(openpi_image, _submission(docker_tmp, "sub", None))
    assert "defaulting to openpi" in r.stdout, r.stdout + r.stderr
    assert "runtime=openpi" in r.stdout


def test_missing_runtime_is_refused_by_the_torch_image(torch_image, docker_tmp):
    """The default is openpi, so a torch submission must say so."""
    r = _entrypoint(torch_image, _submission(docker_tmp, "sub", None))
    assert r.returncode == 2
    assert "umi-arena-torch" in (r.stdout + r.stderr)


@pytest.mark.parametrize("fixture,runtime", [("openpi_image", "openpi"), ("torch_image", "torch")])
def test_entrypoint_runs_the_declared_runtime(request, docker_tmp, fixture, runtime):
    image = request.getfixturevalue(fixture)
    r = _entrypoint(image, _submission(docker_tmp, f"sub-{runtime}", f"runtime: {runtime}\n"))
    assert f"runtime={runtime}" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("fixture,other", [("openpi_image", "torch"), ("torch_image", "openpi")])
def test_entrypoint_refuses_the_other_runtime(request, docker_tmp, fixture, other):
    image = request.getfixturevalue(fixture)
    r = _entrypoint(image, _submission(docker_tmp, f"sub-{other}", f"runtime: {other}\n"))
    assert r.returncode == 2
    out = r.stdout + r.stderr
    assert f"runtime '{other}'" in out and f"umi-arena-{other}" in out, out


def test_unknown_runtime_is_refused(openpi_image, docker_tmp):
    r = _entrypoint(openpi_image, _submission(docker_tmp, "sub", "runtime: tensorflow\n"))
    assert r.returncode == 2
    assert "tensorflow" in (r.stdout + r.stderr)


def test_hf_cache_mount_hides_the_login_token(openpi_image, docker_tmp):
    """The mounts check.py builds for --hf-cache expose the model files and not the token beside them."""
    import importlib.util, sys
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("umi_check", ROOT / "check.py")
    check = importlib.util.module_from_spec(spec); sys.modules["umi_check"] = check; spec.loader.exec_module(check)
    cache = docker_tmp / "cache"; (cache / "hub").mkdir(parents=True); (cache / "token").write_text("hf_secret")
    args = SimpleNamespace(port=8000, env=[], volume=[], image=openpi_image, hf_cache=cache)
    cmd = check.container_command(args, "x", docker_tmp)
    mounts = [(m.rsplit(":", 2)[0] + ":ro", m.rsplit(":", 2)[1]) for m in (cmd[i + 1] for i, a in enumerate(cmd) if a == "-v") if "/hf-cache/" in m]
    r = docker_run(openpi_image, "test -d /hf-cache/hub && test ! -e /hf-cache/token && echo hidden", mounts=mounts)
    assert r.returncode == 0 and "hidden" in r.stdout, r.stdout + r.stderr

