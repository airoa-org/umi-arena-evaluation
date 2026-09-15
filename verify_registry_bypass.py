#!/usr/bin/env python
"""S1: load a real pi0.5 checkpoint WITHOUT writing anything into the openpi tree.

Shows that `create_trained_policy` takes a TrainConfig
object, so the `config.py` registry (and the file-copy step in
doc/setup/common.md) can be bypassed entirely. The YUBI transforms come from
`umi_arena.yubi`, which depends on upstream openpi only.
"""

import os
import time
from pathlib import Path

import numpy as np

import openpi.models.pi0_config as pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

from umi_arena.openpi_yubi import LeRobotYubiDataConfig, discover_asset_id

REPO = Path(os.environ.get("DEPLOY_REPO", "/workspace"))
CKPT = str(REPO / "checkpoints/yubi/pi05_yubi_pickplace/pi05_scpb_full_ah32_task_balanced/136000")
FIXTURE = REPO / ".umi_arena_proto/out/golden_obs.msgpack"

asset_id = discover_asset_id(CKPT)
print("discovered asset_id:", asset_id)

train_config = TrainConfig(
    name="umi_arena_submission",          # arbitrary — never looked up in a registry
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=32),
    data=LeRobotYubiDataConfig(
        assets=AssetsConfig(asset_id=asset_id),
        base_config=DataConfig(prompt_from_task=True),
    ),
)
print("TrainConfig built in-process:", type(train_config).__module__)

t0 = time.monotonic()
policy = _policy_config.create_trained_policy(train_config, CKPT)
print(f"checkpoint loaded in {time.monotonic() - t0:.1f}s")

from openpi_client import msgpack_numpy  # noqa: E402

obs_list = msgpack_numpy.unpackb(FIXTURE.read_bytes())["observations"]
print(f"replaying {len(obs_list)} recorded observations from the golden fixture")

for i, obs in enumerate(obs_list):
    t0 = time.monotonic()
    out = policy.infer(obs)
    dt = time.monotonic() - t0
    actions = np.asarray(out["actions"])
    tag = "  (first call includes JIT)" if i == 0 else ""
    print(f"  call {i}: actions {actions.shape} {actions.dtype} in {dt:.2f}s{tag}")
    if i == 0:
        print("  first timestep:", np.round(actions[0], 4))

print("\nRESULT: registry bypass works; real checkpoint served through an "
      "in-process TrainConfig.")
