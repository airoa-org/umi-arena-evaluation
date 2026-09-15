"""Unit tests for the runner's contract enforcement and adapter loading.

No image, no GPU, no websocket — just the logic that decides whether a
submission's output is acceptable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest


# --- action validation -------------------------------------------------------

class _Const:
    def __init__(self, value):
        self.value = value

    def infer(self, obs):
        return self.value


def _runner(runner_module, value):
    return runner_module.Runner(_Const(value), None, 0)


def test_accepts_float32_chunk(runner_module):
    a = np.zeros((16, 16), dtype=np.float32)
    out = _runner(runner_module, a)._validate(a)
    assert out.shape == (16, 16) and out.dtype == np.float32


def test_casts_float64_to_float32(runner_module):
    """openpi returns float64; the runner normalises rather than rejecting."""
    a = np.zeros((16, 16), dtype=np.float64)
    out = _runner(runner_module, a)._validate(a)
    assert out.dtype == np.float32


@pytest.mark.parametrize("chunk", [16, 32, 50])
def test_accepts_chunk_at_least_16(runner_module, chunk):
    """The harness adopts the first 16 rows; 16 or more is fine, longer is harmless."""
    a = np.zeros((chunk, 16), dtype=np.float32)
    assert _runner(runner_module, a)._validate(a).shape == (chunk, 16)


@pytest.mark.parametrize("chunk", [1, 8, 15])
def test_rejects_short_chunk(runner_module, chunk):
    """Fewer than 16 rows cannot fill the adopted window, so it is rejected."""
    a = np.zeros((chunk, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        _runner(runner_module, a)._validate(a)


@pytest.mark.parametrize("bad", [15, 17, 7, 32])
def test_rejects_wrong_action_dim(runner_module, bad):
    a = np.zeros((16, bad), dtype=np.float32)
    with pytest.raises(ValueError, match="columns"):
        _runner(runner_module, a)._validate(a)


@pytest.mark.parametrize("shape", [(16,), (2, 16, 16), ()])
def test_rejects_wrong_rank(runner_module, shape):
    a = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        _runner(runner_module, a)._validate(a)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(runner_module, bad):
    a = np.zeros((16, 16), dtype=np.float32)
    a[3, 5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _runner(runner_module, a)._validate(a)


def test_rejects_non_array(runner_module):
    with pytest.raises((ValueError, TypeError)):
        _runner(runner_module, "not an array")._validate("not an array")


# --- adapter loading ---------------------------------------------------------

def _write_adapter(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "submission"
    d.mkdir(exist_ok=True)
    (d / "policy.py").write_text(textwrap.dedent(body))
    (d / "umi_arena.yaml").write_text("runtime: openpi\n")
    return d


def test_loads_adapter_and_passes_checkpoint_dir(runner_module, tmp_path):
    d = _write_adapter(tmp_path, """
        class Policy:
            def __init__(self, checkpoint_dir):
                self.checkpoint_dir = checkpoint_dir
            def infer(self, obs):
                return None
    """)
    policy = runner_module.load_adapter(d, "/some/ckpt")
    assert policy.checkpoint_dir == "/some/ckpt"


def test_missing_policy_py_is_a_clear_error(runner_module, tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="policy.py"):
        runner_module.load_adapter(d, "")


def test_policy_py_without_Policy_class(runner_module, tmp_path):
    d = _write_adapter(tmp_path, "def infer(obs): return None\n")
    with pytest.raises(AttributeError, match="Policy"):
        runner_module.load_adapter(d, "")


def test_adapter_import_error_propagates(runner_module, tmp_path):
    """A team's broken import must surface, not be swallowed."""
    d = _write_adapter(tmp_path, "import a_package_that_does_not_exist\n")
    with pytest.raises(ModuleNotFoundError):
        runner_module.load_adapter(d, "")


def test_constructor_failure_propagates(runner_module, tmp_path):
    d = _write_adapter(tmp_path, """
        class Policy:
            def __init__(self, checkpoint_dir):
                raise RuntimeError("bad checkpoint")
            def infer(self, obs):
                return None
    """)
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        runner_module.load_adapter(d, "")
