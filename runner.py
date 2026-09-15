#!/usr/bin/env python3
"""Prototype UMI Arena runner.

Speaks the evaluation harness's policy-server contract (websocket + msgpack-numpy)
and delegates to a submission adapter (`policy.py` exposing `class Policy`).

This is the thing that, in the real design, lives inside the fixed runner image
so teams never implement websockets themselves.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# websockets and openpi_client are imported where used, so the contract logic is unit-testable outside the image.

ACTION_DIM = 16
# The harness adopts the first rows of each chunk, at most 16 in its default configs; shorter chunks cannot fill that.
MIN_CHUNK = 16


def load_adapter(adapter_dir: Path, checkpoint_dir: str):
    policy_py = adapter_dir / "policy.py"
    if not policy_py.is_file():
        raise FileNotFoundError(f"adapter has no policy.py: {policy_py}")
    spec = importlib.util.spec_from_file_location("umi_arena_submission", policy_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Policy"):
        raise AttributeError("policy.py does not define `Policy`")
    return module.Policy(checkpoint_dir)


def describe(obs: dict) -> str:
    parts = []
    for k in sorted(obs):
        v = obs[k]
        if isinstance(v, np.ndarray):
            parts.append(f"{k}: {v.dtype} {v.shape}")
        else:
            parts.append(f"{k}: {type(v).__name__}={v!r}"[:70])
    return "\n    ".join(parts)


class Runner:
    def __init__(self, policy, fixture_out: Path | None, fixture_count: int):
        self.policy = policy
        self.fixture_out = fixture_out
        self.fixture_count = fixture_count
        self.captured: list[dict] = []
        self.calls = 0
        self.metadata = {"server": "umi-arena-runner-proto", "action_dim": ACTION_DIM}

    def _validate(self, actions) -> np.ndarray:
        a = np.asarray(actions)
        if a.dtype != np.float32:
            # openpi hands back float64; normalise rather than reject.
            logging.info("casting actions from %s to float32", a.dtype)
            a = a.astype(np.float32)
        if a.ndim != 2:
            raise ValueError(f"actions must be 2D [chunk, dim], got shape {a.shape}")
        if a.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must have {ACTION_DIM} columns, got {a.shape}")
        if a.shape[0] < MIN_CHUNK:
            raise ValueError(f"actions must have at least {MIN_CHUNK} rows, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("actions contain non-finite values")
        return a

    async def handler(self, websocket) -> None:
        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()
        # Contract: a metadata dict goes out first; inference_node blocks on this recv at startup.
        await websocket.send(packer.pack(self.metadata))
        logging.info("client connected; metadata sent")
        while True:
            obs = msgpack_numpy.unpackb(await websocket.recv())
            self.calls += 1
            if self.calls == 1:
                logging.info("first observation:\n    %s", describe(obs))
            if len(self.captured) < self.fixture_count:
                self.captured.append(obs)
                self._flush()
            t0 = time.monotonic()
            actions = self._validate(self.policy.infer(obs))
            dt_ms = (time.monotonic() - t0) * 1000.0
            if self.calls <= 3 or self.calls % 20 == 0:
                logging.info("call %d: actions %s in %.1f ms", self.calls, actions.shape, dt_ms)
            await websocket.send(packer.pack({"actions": actions, "server_timing": {"infer_ms": dt_ms}}))

    def _flush(self) -> None:
        if self.fixture_out is None:
            return
        from openpi_client import msgpack_numpy

        self.fixture_out.parent.mkdir(parents=True, exist_ok=True)
        packer = msgpack_numpy.Packer()
        self.fixture_out.write_bytes(packer.pack({"observations": self.captured}))


async def main_async(args) -> None:
    import websockets.asyncio.server as ws_server

    policy = load_adapter(Path(args.adapter_dir), args.checkpoint_dir)
    runner = Runner(policy, Path(args.fixture_out) if args.fixture_out else None, args.fixture_count)

    async def wrapped(ws):
        import websockets.exceptions

        try:
            await runner.handler(ws)
        except websockets.exceptions.ConnectionClosed:
            logging.info("client disconnected after %d calls", runner.calls)
        except Exception:
            logging.exception("handler error after %d calls", runner.calls)
            # openpi's protocol: a str frame is the server's error text, which the client raises or prints.
            try:
                await ws.send(traceback.format_exc())
            except Exception:
                pass

    async with ws_server.serve(wrapped, args.host, args.port, compression=None, max_size=None):
        logging.info("umi-arena runner listening on ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--fixture-out", default="")
    p.add_argument("--fixture-count", type=int, default=5)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[runner] %(message)s")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
