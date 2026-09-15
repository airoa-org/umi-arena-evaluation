"""The websocket contract, exercised against a real running runner.

These start `runner.py` as a subprocess with the dummy adapter and talk to it
the way the harness's inference_node does: msgpack-numpy over a websocket,
metadata dict first. No GPU, no model.

Skipped when `websockets` / `openpi_client` are not importable on the host; the
checker, run against the image, covers the same ground there.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets", reason="websockets not installed on the host")
pytest.importorskip("openpi_client", reason="openpi_client not installed on the host")

import websockets.sync.client  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PORT = 8177


def make_obs(h: int = 480, w: int = 640) -> dict:
    """An observation shaped like the one the harness actually sends."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {
        "observation.image.left": img,
        "observation.image.right": img.copy(),
        "observation.image.center": img.copy(),  # some harness configs add it; the contract does not
        "observation.pose.left_hand_root_to_right_hand_root.absolute":
            np.array([0.23, -0.05, 0.27, -0.58, -0.09, 0.72, 0.34], dtype=np.float32),
        "observation.joint_states": np.array([0.24, 0.78], dtype=np.float32),
        "prompt": "Pick up the cup with your left hand",
    }


def _start_runner(adapter: Path, port: int, *extra: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "runner.py"), "--port", str(port), "--adapter-dir", str(adapter), *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"runner exited early:\n{proc.stdout.read()}")
        try:
            with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", open_timeout=1):
                return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    pytest.fail("runner did not start")


@pytest.fixture(scope="module")
def running_runner(tmp_path_factory):
    fixture_out = tmp_path_factory.mktemp("fx") / "obs.msgpack"
    proc = _start_runner(ROOT / "adapters" / "dummy", PORT, "--fixture-out", str(fixture_out), "--fixture-count", "2")
    yield fixture_out
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def client(running_runner):
    conn = websockets.sync.client.connect(
        f"ws://127.0.0.1:{PORT}", compression=None, max_size=None, open_timeout=5)
    yield conn
    conn.close()


def _infer(conn, obs):
    conn.send(msgpack_numpy.Packer().pack(obs))
    return msgpack_numpy.unpackb(conn.recv())


# --- the handshake -----------------------------------------------------------

def test_server_sends_metadata_before_any_request(client):
    """inference_node blocks on this recv at startup; a server that skips it
    hangs the harness rather than failing visibly."""
    meta = msgpack_numpy.unpackb(client.recv())
    assert isinstance(meta, dict)
    assert meta.get("action_dim") == 16


# --- the request/response cycle ---------------------------------------------

def test_returns_a_valid_action_chunk(client):
    msgpack_numpy.unpackb(client.recv())          # metadata
    out = _infer(client, make_obs())
    actions = np.asarray(out["actions"])
    assert actions.ndim == 2 and actions.shape[1] == 16
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()


def test_reports_server_timing(client):
    msgpack_numpy.unpackb(client.recv())
    out = _infer(client, make_obs())
    assert "server_timing" in out
    assert out["server_timing"]["infer_ms"] >= 0


def test_dummy_action_layout_survives_the_round_trip(client):
    """Identity rotation deltas and absolute gripper values must arrive intact —
    this is what the publish layer slices up."""
    msgpack_numpy.unpackb(client.recv())
    a = np.asarray(_infer(client, make_obs())["actions"])[0]
    assert a[6] == pytest.approx(1.0)      # left.dqw
    assert a[13] == pytest.approx(1.0)     # right.dqw
    assert a[14] == pytest.approx(0.7)     # hand_left_motor_joint
    assert a[15] == pytest.approx(0.7)     # hand_right_motor_joint


def test_many_sequential_calls_on_one_connection(client):
    """The harness holds one connection open for the whole episode."""
    msgpack_numpy.unpackb(client.recv())
    for _ in range(25):
        out = _infer(client, make_obs())
        assert np.asarray(out["actions"]).shape[1] == 16


def test_native_resolution_images_are_accepted(client):
    """480x640 is what the harness sends. A runner that assumed 224 would break."""
    msgpack_numpy.unpackb(client.recv())
    out = _infer(client, make_obs(480, 640))
    assert np.asarray(out["actions"]).shape[1] == 16


def test_reconnect_is_supported(running_runner):
    """inference_node reconnects after a cold-start timeout; the server must
    accept a second connection and re-send metadata."""
    for _ in range(3):
        conn = websockets.sync.client.connect(
            f"ws://127.0.0.1:{PORT}", compression=None, max_size=None, open_timeout=5)
        meta = msgpack_numpy.unpackb(conn.recv())
        assert meta.get("action_dim") == 16
        assert np.asarray(_infer(conn, make_obs())["actions"]).shape[1] == 16
        conn.close()


# --- golden fixture capture --------------------------------------------------

def test_captures_a_replayable_fixture(client, running_runner):
    msgpack_numpy.unpackb(client.recv())
    obs = make_obs()
    _infer(client, obs)
    _infer(client, obs)
    time.sleep(0.5)
    assert running_runner.is_file(), "fixture was not written"
    captured = msgpack_numpy.unpackb(running_runner.read_bytes())["observations"]
    assert len(captured) >= 1
    first = captured[0]
    assert set(first) == set(obs)
    assert first["observation.image.left"].shape == (480, 640, 3)
    assert first["observation.image.left"].dtype == np.uint8
    np.testing.assert_array_equal(
        first["observation.pose.left_hand_root_to_right_hand_root.absolute"],
        obs["observation.pose.left_hand_root_to_right_hand_root.absolute"])
    assert first["prompt"] == obs["prompt"]


def test_a_rejected_chunk_comes_back_as_a_str_frame():
    """openpi's protocol: a str frame is the server's error text; inference_node raises it."""
    proc = _start_runner(ROOT / "tests" / "broken" / "cols15", PORT + 1)
    try:
        with websockets.sync.client.connect(f"ws://127.0.0.1:{PORT + 1}", max_size=None) as conn:
            msgpack_numpy.unpackb(conn.recv())
            conn.send(msgpack_numpy.Packer().pack(make_obs()))
            frame = conn.recv()
        assert isinstance(frame, str) and "16 columns" in frame, frame[-300:]
    finally:
        proc.kill()


@pytest.mark.parametrize("from_fixture", [False, True])
def test_checker_rejects_a_policy_that_requires_center(tmp_path, from_fixture):
    from test_unit_check import _load_check

    check = _load_check()
    proc = _start_runner(ROOT / "tests" / "broken" / "needs_center", PORT + 2)
    try:
        with websockets.sync.client.connect(f"ws://127.0.0.1:{PORT + 2}", max_size=None) as conn:
            msgpack_numpy.unpackb(conn.recv())
            assert _infer(conn, make_obs())["actions"].shape == (16, 16)
        if from_fixture:
            fixture = tmp_path / "sim.msgpack"
            fixture.write_bytes(msgpack_numpy.Packer().pack({"observations": [make_obs()]}))
            observations = check.load_fixtures(fixture)
        else:
            observations = [check.synthetic_observation()]
        report = check.Report()
        args = SimpleNamespace(keepalive=20.0, handshake_timeout=5.0, slow_ms=1600.0)
        with pytest.raises(check.CheckFailed):
            check.run_protocol(f"ws://127.0.0.1:{PORT + 2}", observations, args, report)
        failed = [c for c in report.checks if c["status"] == "fail"]
        assert failed[-1]["name"] == "call 0: server answered"
        assert "KeyError" in failed[-1]["detail"] and "observation.image.center" in failed[-1]["detail"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
