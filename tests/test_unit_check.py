"""check.py's host-side logic: the runtime parser and the docker command it builds."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_check():
    spec = importlib.util.spec_from_file_location("umi_check", ROOT / "check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["umi_check"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text,expected", [
    ("runtime: openpi\n", "openpi"),
    ("runtime: torch\n", "torch"),
    ("runtime : torch\n", "torch"),
    ('runtime : "torch"  # PyTorch\n', "torch"),
    ("runtime:openpi\n", "openpi"),
    ("  runtime:   openpi   \n", "openpi"),
    ("runtime: openpi  # the JAX one\n", "openpi"),
    ('runtime: "torch"\n', "torch"),           # quoted values are valid YAML
    ("runtime: 'openpi'\n", "openpi"),
    ("name: x\nruntime: torch\nother: y\n", "torch"),
    ("runtime: openpi\nruntime: torch\n", "openpi"),   # first declaration wins
    ("", None),
    ("name: x\n", None),
    ("myruntime: torch\n", None),            # must not match a suffix
])
def test_parse_runtime(tmp_path, text, expected):
    (tmp_path / "umi_arena.yaml").write_text(text)
    assert _load_check().parse_runtime(tmp_path) == expected


def test_parse_runtime_without_a_yaml_is_none(tmp_path):
    assert _load_check().parse_runtime(tmp_path) is None


@pytest.mark.parametrize("text", [
    "runtime:\n", "runtime: !!!\n", 'runtime: "torch\n',
    "runtime: torch extra\n", "runtime: torch#comment\n",
])
def test_invalid_runtime_is_reported_before_startup(tmp_path, monkeypatch, text):
    check = _load_check()
    (tmp_path / "umi_arena.yaml").write_text(text)
    report = tmp_path / "report.json"
    monkeypatch.setattr(check, "preflight", lambda *args: pytest.fail("invalid runtime reached Docker preflight"))
    assert check.main(["--submission", str(tmp_path), "--calls", "1", "--report", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["checks"] == [{"name": "umi_arena.yaml declares a runtime", "status": "fail",
                                 "detail": f"invalid runtime declaration in {tmp_path / 'umi_arena.yaml'}; expected runtime: openpi or runtime: torch"}]


def test_synthetic_observation_has_only_the_real_robot_contract_keys():
    obs = _load_check().synthetic_observation()
    assert set(obs) == {
        "observation.image.left", "observation.image.right",
        "observation.pose.left_hand_root_to_right_hand_root.absolute",
        "observation.joint_states", "prompt",
    }


def _args(**kw):
    return SimpleNamespace(port=8000, env=list(kw.get("env", [])), volume=[], image="umi-arena-openpi:t",
                           hf_cache=kw.get("hf_cache"))


def _flags(cmd, flag="-e"):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


def test_a_host_hf_token_is_never_forwarded_on_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    cmd = _load_check().container_command(_args(), "umi-check-8000", tmp_path)
    assert not any(f.startswith("HF_TOKEN") for f in _flags(cmd)) and "hf_secret" not in " ".join(cmd)


def test_an_explicit_env_token_passes_through(tmp_path):
    cmd = _load_check().container_command(_args(env=["HF_TOKEN=hf_mine"]), "umi-check-8000", tmp_path)
    assert _flags(cmd) == ["UMI_ARENA_PORT=8000", "HF_TOKEN=hf_mine"]
    assert cmd[-1] == "umi-arena-openpi:t" and f"{tmp_path}:/submission:ro" in _flags(cmd, "-v")


def test_hf_cache_mounts_the_model_files_and_not_the_login_files(tmp_path):
    cache = tmp_path / "cache"; (cache / "hub").mkdir(parents=True); (cache / "modules").mkdir()
    (cache / "token").write_text("hf_secret")
    cmd = _load_check().container_command(_args(hf_cache=cache), "umi-check-8000", tmp_path)
    mounts = _flags(cmd, "-v")
    assert f"{cache.resolve() / 'hub'}:/hf-cache/hub:ro" in mounts and f"{cache.resolve() / 'modules'}:/hf-cache/modules:ro" in mounts
    assert not any(m.startswith(f"{cache.resolve()}:") for m in mounts), "the cache root, and its token, must not be mounted"
    assert "HF_HOME=/hf-cache" in _flags(cmd) and "HF_HUB_OFFLINE=1" in _flags(cmd)


def test_hf_cache_without_modules_mounts_hub_only(tmp_path):
    cache = tmp_path / "cache"; (cache / "hub").mkdir(parents=True)
    cmd = _load_check().container_command(_args(hf_cache=cache), "umi-check-8000", tmp_path)
    assert [m for m in _flags(cmd, "-v") if "/hf-cache/" in m] == [f"{cache.resolve() / 'hub'}:/hf-cache/hub:ro"]


def test_hf_cache_without_a_hub_dir_is_rejected_at_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        _load_check().parse_args(["--submission", str(tmp_path), "--hf-cache", str(tmp_path)])


class _FakeClient:
    """Stands in for Client: metadata and per-call round-trip times are scripted."""
    plan: dict = {}

    def __init__(self, url, *, keepalive_s):
        self.calls = 0

    def metadata(self, timeout_s):
        m = self.plan.get("metadata", {"server": "fake", "action_dim": 16})
        if isinstance(m, Exception):
            raise m
        return m

    def infer(self, obs):
        dts = self.plan.get("dts", [10.0])
        dt = dts[min(self.calls, len(dts) - 1)]; self.calls += 1
        return {"actions": np.zeros((16, 16), dtype=np.float32), "server_timing": {"infer_ms": dt}}, dt

    def close(self):
        pass


def _protocol(plan, calls=3):
    check = _load_check(); _FakeClient.plan = plan
    args = SimpleNamespace(keepalive=20.0, handshake_timeout=5.0, slow_ms=1600.0)
    report = check.Report()
    obs = [check.synthetic_observation(i, check.PROMPTS[0]) for i in range(calls)]
    try:
        check.run_protocol("ws://fake", obs, args, report, connect=_FakeClient)
    except check.CheckFailed:
        pass
    return {c["name"]: c["status"] for c in report.checks}


def test_protocol_happy_path_passes_every_stage():
    got = _protocol({})
    assert got["metadata arrives on connect"] == "pass" and got["3 calls return valid action chunks"] == "pass"
    assert got["first call is pre-warmed"] == "pass" and got["calls are within the published budget"] == "pass"
    assert got["prompt can change mid-connection"] == "pass" and got["reconnect works"] == "pass"


@pytest.mark.parametrize("metadata,failed", [
    (TimeoutError(), "metadata arrives on connect"),
    (["not", "a", "dict"], "metadata is a dict"),
    ({"action_dim": 15}, "metadata declares action_dim 16"),
])
def test_protocol_rejects_bad_metadata(metadata, failed):
    got = _protocol({"metadata": metadata})
    assert got[failed] == "fail" and "reconnect works" not in got, got


def test_protocol_warns_when_the_first_call_is_not_pre_warmed():
    got = _protocol({"dts": [30000.0, 100.0, 100.0]})
    assert got["first call is pre-warmed"] == "warn" and got["calls are within the published budget"] == "warn"


def test_protocol_warns_when_calls_exceed_the_target():
    got = _protocol({"dts": [2000.0]})
    assert got["first call is pre-warmed"] == "pass" and got["calls are within the published budget"] == "warn"


def _stub_codec(monkeypatch, payload):
    """openpi_client is not on the host; load_fixtures' own logic is what is under test."""
    import types
    pkg, mod = types.ModuleType("openpi_client"), types.ModuleType("openpi_client.msgpack_numpy")
    mod.unpackb = lambda _bytes: payload
    pkg.msgpack_numpy = mod
    monkeypatch.setitem(sys.modules, "openpi_client", pkg)
    monkeypatch.setitem(sys.modules, "openpi_client.msgpack_numpy", mod)


def test_fixtures_are_the_observations_list(monkeypatch, tmp_path):
    f = tmp_path / "obs.msgpack"; f.write_bytes(b"x")
    _stub_codec(monkeypatch, {"observations": [{"prompt": "a"}, {"prompt": "b"}]})
    assert [o["prompt"] for o in _load_check().load_fixtures(f)] == ["a", "b"]


def test_fixtures_drop_center_images_and_other_non_contract_keys(monkeypatch, tmp_path):
    check = _load_check()
    obs = check.synthetic_observation()
    recorded = {**obs, "observation.image.center": obs["observation.image.left"],
                "action.joint_states": np.zeros((16, 2), dtype=np.float32)}
    f = tmp_path / "obs.msgpack"; f.write_bytes(b"x")
    _stub_codec(monkeypatch, {"observations": [recorded]})
    loaded, = check.load_fixtures(f)
    assert set(loaded) == set(obs)
    for key, value in obs.items():
        np.testing.assert_array_equal(loaded[key], value)
    assert "observation.image.center" in recorded


@pytest.mark.parametrize("payload", [{}, {"observations": []}, [1, 2]])
def test_a_fixture_file_without_observations_is_refused(monkeypatch, tmp_path, payload):
    f = tmp_path / "obs.msgpack"; f.write_bytes(b"x")
    _stub_codec(monkeypatch, payload)
    with pytest.raises(ValueError, match="observations"):
        _load_check().load_fixtures(f)


def test_a_checker_bug_still_writes_the_report(monkeypatch, tmp_path):
    check = _load_check()
    monkeypatch.setattr(check, "preflight", lambda args, report: (_ for _ in ()).throw(RuntimeError("boom")))
    report = tmp_path / "report.json"
    assert check.main(["--submission", str(tmp_path), "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    err = [c for c in data["checks"] if c["name"] == "checker error"]
    assert err and err[0]["status"] == "fail" and "RuntimeError: boom" in err[0]["detail"]


def test_zero_calls_is_rejected_at_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        _load_check().parse_args(["--submission", str(tmp_path), "--calls", "0"])


def test_a_failed_hub_fetch_lands_in_the_report(monkeypatch, tmp_path):
    check = _load_check()
    monkeypatch.setattr(check, "fetch_hf", lambda repo, rev: (_ for _ in ()).throw(RuntimeError("no network")))
    report = tmp_path / "out" / "report.json"
    assert check.main(["--hf", "team/model", "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    fetched = [c for c in data["checks"] if c["name"] == "submission fetched from the Hub"]
    assert fetched and fetched[0]["status"] == "fail" and "no network" in fetched[0]["detail"]
