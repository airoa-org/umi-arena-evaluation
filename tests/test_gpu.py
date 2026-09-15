"""Tests that need the built image *and* a physical GPU.

These are the ones that would have caught the two hardware surprises we hit:
Blackwell (sm_120) code generation, and the cold-start latency that exceeds the
websocket keepalive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import docker_run

pytestmark = [pytest.mark.image, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[1]
CKPT = os.environ.get(
    "UMI_ARENA_TEST_CHECKPOINT",
    str(Path.home() / "umi-arena-checkpoints/pi05_yubi"),
)


# --- the GPU is usable from both frameworks ---------------------------------

def test_jax_sees_the_gpu(gpu_openpi_image):
    r = docker_run(gpu_openpi_image,
                   "/opt/venv/openpi/bin/python -c "
                   "'import jax; print(jax.devices())'", gpu=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cuda" in r.stdout.lower()


def test_jax_compiles_for_this_architecture(gpu_openpi_image):
    """JAX 0.5.3 must compile for sm_120 with its bundled ptxas. A matmul goes
    PTX -> ptxas, so this fails loudly if the bundled ptxas is too old."""
    r = docker_run(gpu_openpi_image,
                   "/opt/venv/openpi/bin/python -c '"
                   "import jax.numpy as jnp; x = jnp.ones((256, 256)); "
                   "print(float((x @ x).sum()))'", gpu=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert float(r.stdout.strip()) == pytest.approx(256 ** 3)


def test_no_ptxas_workaround_is_needed(gpu_openpi_image):
    """The evaluation harness links the host ptxas for Jetson (sm_101). If the image
    ever needs the same crutch on the workstation, this test is where we find
    out — it runs with the workaround explicitly off."""
    r = docker_run(gpu_openpi_image,
                   "/opt/venv/openpi/bin/python -c '"
                   "import jax.numpy as jnp; jnp.ones((64, 64)) @ jnp.ones((64, 64)); "
                   "print(\"compiled\")'",
                   gpu=True, env=[("OPENPI_LINK_HOST_PTXAS", "0")])
    assert "compiled" in r.stdout, r.stdout + r.stderr


def test_torch_sees_the_gpu(gpu_torch_image):
    """The torch venv's cu128 build must target this GPU (sm_120)."""
    r = docker_run(gpu_torch_image,
                   "/opt/venv/torch/bin/python -c '"
                   "import torch; print(torch.cuda.is_available(), "
                   "torch.cuda.get_device_capability() if torch.cuda.is_available() else None, "
                   "torch.__version__)'", gpu=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.split()[0] == "True", (
        "torch cannot use this GPU; the torch runtime needs a separate cu128 venv. "
        + r.stdout)


# --- the real model ----------------------------------------------------------

def _have_checkpoint() -> bool:
    return (Path(CKPT) / "params").is_dir()


@pytest.mark.slow
def test_asset_id_is_discovered_from_a_real_checkpoint(gpu_openpi_image):
    if not _have_checkpoint():
        pytest.skip(f"no checkpoint at {CKPT}")
    r = docker_run(gpu_openpi_image,
                   "/opt/venv/openpi/bin/python -c '"
                   "from umi_arena.openpi_yubi import discover_asset_id; "
                   "print(discover_asset_id(\"/ckpt\"))'",
                   gpu=True, mounts=[(CKPT, "/ckpt")],
                   env=[("PYTHONPATH", "/opt/umi_arena")])
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip(), "no asset id found"


@pytest.mark.slow
def test_real_checkpoint_loads_without_the_config_registry(gpu_openpi_image, docker_tmp):
    """The reference adapter rests on this: openpi serves a checkpoint from a TrainConfig
    built in-process, with nothing copied into the openpi tree."""
    if not _have_checkpoint():
        pytest.skip(f"no checkpoint at {CKPT}")
    script = docker_tmp / "load.py"
    script.write_text(
        "import numpy as np, time\n"
        "import openpi.models.pi0_config as pi0_config\n"
        "from openpi.policies import policy_config\n"
        "from openpi.training.config import AssetsConfig, DataConfig, TrainConfig\n"
        "from umi_arena.openpi_yubi import LeRobotYubiDataConfig, discover_asset_id\n"
        "cfg = TrainConfig(\n"
        "    name='test',\n"
        "    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=32),\n"
        "    data=LeRobotYubiDataConfig(\n"
        "        assets=AssetsConfig(asset_id=discover_asset_id('/ckpt')),\n"
        "        base_config=DataConfig(prompt_from_task=True)))\n"
        "p = policy_config.create_trained_policy(cfg, '/ckpt')\n"
        "rng = np.random.default_rng(0)\n"
        "img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)\n"
        "obs = {'observation.image.left': img, 'observation.image.right': img.copy(),\n"
        "       'observation.pose.left_hand_root_to_right_hand_root.absolute':\n"
        "           np.array([0.23,-0.05,0.27,-0.58,-0.09,0.72,0.34], dtype=np.float32),\n"
        "       'observation.joint_states': np.array([0.24,0.78], dtype=np.float32),\n"
        "       'prompt': 'pick up the cup'}\n"
        "t0 = time.monotonic(); a1 = np.asarray(p.infer(obs)['actions']); cold = time.monotonic()-t0\n"
        "t0 = time.monotonic(); a2 = np.asarray(p.infer(obs)['actions']); warm = time.monotonic()-t0\n"
        "import json; print(json.dumps({'shape': list(a1.shape), 'dtype': str(a1.dtype),\n"
        "                               'cold_s': cold, 'warm_s': warm,\n"
        "                               'finite': bool(np.isfinite(a1).all())}))\n")
    r = docker_run(gpu_openpi_image,
                   "/opt/venv/openpi/bin/python /work/load.py 2>/dev/null | tail -1",
                   gpu=True,
                   mounts=[(CKPT, "/ckpt"), (str(docker_tmp), "/work")],
                   env=[("PYTHONPATH", "/opt/umi_arena"),
                        ("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")],
                   timeout=1200)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["shape"][1] == 16, info
    assert info["finite"], info
    # The measurements this suite exists to protect.
    assert info["warm_s"] < 1.6, f"steady-state inference exceeds the budget: {info}"
    print(f"\ncold={info['cold_s']:.1f}s warm={info['warm_s']*1000:.0f}ms shape={info['shape']}")


@pytest.mark.slow
def test_cold_start_justifies_prewarming(gpu_openpi_image):
    """Not a pass/fail on the number itself — a guard on the reasoning. If the
    first call ever drops below the 20 s websocket keepalive, pre-warming stops
    being mandatory and the docs should change."""
    if not _have_checkpoint():
        pytest.skip(f"no checkpoint at {CKPT}")
    pytest.skip("covered by test_real_checkpoint_loads_without_the_config_registry")
