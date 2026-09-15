"""Real checkpoints pulled from the Hugging Face Hub.

Everything else in this suite uses a dummy or a local YUBI checkpoint. These
answer a different question: does an off-the-shelf model, of the kind a team
would actually start from, load and run inside the image?

Fetch the models first with the image's own huggingface_hub, then
chown the result, because the venvs run as root:

    docker run --rm -e HF_HOME=/hf/home -v ~/umi-arena-hf:/hf --entrypoint bash \
      umi-arena-torch:<tag> -lc '/opt/venv/torch/bin/python -c "
from huggingface_hub import snapshot_download
for m in (\"smolvla_base\", \"pi05_base\"):
    snapshot_download(f\"lerobot/{m}\", local_dir=f\"/hf/{m}\")"; chown -R '"$(id -u):$(id -g)"' /hf'

or point UMI_ARENA_HF_DIR at a directory that already has them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import docker_run

pytestmark = [pytest.mark.image, pytest.mark.gpu, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
HF_DIR = Path(os.environ.get("UMI_ARENA_HF_DIR", Path.home() / "umi-arena-hf"))
ADAPTER = ROOT / "adapters" / "lerobot_hf"


def _submission(docker_tmp: Path, model: str) -> Path:
    src = HF_DIR / model
    if not (src / "config.json").is_file():
        pytest.skip(f"{model} not fetched into {HF_DIR}")
    d = docker_tmp / model
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ADAPTER / "policy.py", d / "policy.py")
    shutil.copy(ADAPTER / "umi_arena.yaml", d / "umi_arena.yaml")
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("."):
            link = d / f.name
            if not link.exists():
                link.symlink_to(f)
    return d


DRIVER = """
import importlib.util, json, numpy as np, time
spec = importlib.util.spec_from_file_location("sub", "/submission/policy.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.monotonic(); p = m.Policy("/submission"); load = time.monotonic() - t0
t0 = time.monotonic(); obs = m.Policy._synthetic_observation(); a = p.infer(obs); warm = time.monotonic() - t0
obs["observation.image.center"] = obs["observation.image.left"].copy()  # the sim config sends it; must be ignored
extra_ok = p.infer(obs).shape == a.shape
print("JSON " + json.dumps({"shape": list(a.shape), "dtype": str(a.dtype),
                            "finite": bool(np.isfinite(a).all()), "extra_key_ok": extra_ok,
                            "load_s": load, "warm_s": warm}))
"""


def _run(gpu_torch_image, submission: Path, venv: str = "torch"):
    script = f"cat > /tmp/d.py <<'EOF'\n{DRIVER}\nEOF\n/opt/venv/{venv}/bin/python /tmp/d.py"
    return docker_run(
        gpu_torch_image, script, gpu=True,
        mounts=[(f"{submission}:ro", "/submission"), (str(HF_DIR), str(HF_DIR))],
        env=[("PYTHONPATH", "/opt/umi_arena"), ("HF_HOME", str(HF_DIR / "home"))]
            + ([("HF_TOKEN", os.environ["HF_TOKEN"])] if os.environ.get("HF_TOKEN") else []),
        timeout=2400,
    )


def _parse(result) -> dict:
    import json
    lines = [l for l in result.stdout.splitlines() if l.startswith("JSON ")]
    assert lines, result.stdout[-3000:] + result.stderr[-2000:]
    return json.loads(lines[-1][5:])


# --- SmolVLA ----------------------------------------------------------------

def test_smolvla_base_is_refused_at_load(gpu_torch_image, docker_tmp):
    """lerobot/smolvla_base, straight off the Hub, is a three-camera, 6-state,
    6-action model for another robot. The adapter must refuse it before any
    inference, with the message a team would read, rather than feed it our
    cameras twice and pad its state."""
    r = _run(gpu_torch_image, _submission(docker_tmp, "smolvla_base"))
    assert r.returncode != 0, r.stdout[-2000:]
    combined = r.stdout + r.stderr
    assert "declares 3 cameras" in combined, combined[-3000:]


def test_openpi_width_actions_are_trimmed_to_the_publish_vector(runner_module):
    """32-dim openpi actions carry the publish vector in the first 16 slots;
    the adapter trims, and the trimmed result passes the gate."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hf_adapter", ADAPTER / "policy.py")
    # Imported for its pure-numpy mapping only; the module-level torch/lerobot
    # imports are why this is not a plain `import`.
    src = (ADAPTER / "policy.py").read_text()
    ns: dict = {"np": np, "PUBLISH_DIM": 16, "OPENPI_DIM": 32}
    body = src[src.index("    def _to_publish_vector"):src.index("    def _prewarm")]
    exec("class P:\n" + body, ns)                                   # noqa: S102
    out = ns["P"]()._to_publish_vector(np.zeros((50, 32), dtype=np.float32))
    assert out.shape == (50, 16)
    runner_module.Runner(None, None, 0)._validate(out)


# --- a 16-column SmolVLA ------------------------------------------------------

def test_smolvla_yubi16_passes_the_contract(runner_module, gpu_torch_image, docker_tmp):
    """The fixture scripts/make_smolvla_yubi16.py builds: smolvla_base with
    YUBI feature shapes and stats, weights untouched. It is the smallest
    checkpoint that exercises the torch path all the way to an accepted chunk."""
    r = _run(gpu_torch_image, _submission(docker_tmp, "smolvla_yubi16"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    info = _parse(r)
    assert info["finite"], info
    assert info["shape"][1] == 16, info
    assert info["extra_key_ok"], "an extra observation key (observation.image.center) must be ignored"
    actions = np.zeros(info["shape"], dtype=np.float32)
    runner_module.Runner(None, None, 0)._validate(actions)
    print(f"\nsmolvla_yubi16: {info['shape']} load={info['load_s']:.1f}s "
          f"warm={info['warm_s'] * 1000:.0f}ms")


# --- a 16-column GR00T ---------------------------------------------------------

def test_groot_yubi16_passes_the_contract(runner_module, gpu_torch_image, docker_tmp):
    """The fixture scripts/make_groot_yubi16.py builds: nvidia/GR00T-N1.7-3B with
    a new_embodiment head shaped for YUBI, weights untouched. Same runtime and
    adapter as SmolVLA."""
    r = _run(gpu_torch_image, _submission(docker_tmp, "groot_yubi16"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    info = _parse(r)
    assert info["finite"], info
    assert info["shape"][1] == 16, info
    assert info["extra_key_ok"], "an extra observation key (observation.image.center) must be ignored"
    runner_module.Runner(None, None, 0)._validate(np.zeros(info["shape"], dtype=np.float32))
    print(f"\ngroot_yubi16: {info['shape']} load={info['load_s']:.1f}s "
          f"warm={info['warm_s'] * 1000:.0f}ms")


# --- pi0.5 -------------------------------------------------------------------

def test_pi05_base_lerobot_port_loads_in_bf16(gpu_torch_image, docker_tmp):
    """lerobot/pi05_base would not load on lerobot 0.4.4 (its transformers pin
    failed the port's SigLIP check). On lerobot 0.6 it loads. Its config says
    float32, 16.6 GB for 4.14B params, over the card; a submission sets
    `dtype: bfloat16` in its own config.json, and this loads it that way to prove
    the port fits.

    Straight through lerobot, not the adapter: the adapter refuses pi05_base for
    its three cameras, which is right for a submission and wrong for this check."""
    sub = _submission(docker_tmp, "pi05_base")
    r = docker_run(
        gpu_torch_image,
        "/opt/venv/torch/bin/python -c '"
        "import torch; from lerobot.configs.policies import PreTrainedConfig; "
        "from lerobot.policies.factory import get_policy_class; "
        "cfg = PreTrainedConfig.from_pretrained(\"/submission\"); cfg.dtype = \"bfloat16\"; "
        "p = get_policy_class(\"pi05\").from_pretrained(\"/submission\", config=cfg).to(\"cuda\"); "
        "print(\"JSON\", next(p.parameters()).dtype, round(torch.cuda.memory_allocated()/1e9, 1))'",
        gpu=True, mounts=[(f"{sub}:ro", "/submission"), (str(HF_DIR), str(HF_DIR))],
        env=[("HF_HOME", str(HF_DIR / "home"))], timeout=1200,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "JSON torch.bfloat16" in r.stdout, r.stdout[-500:]
