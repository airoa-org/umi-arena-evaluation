"""Unit tests for umi_arena helpers and the entrypoint's runtime selection."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from umi_arena.openpi_yubi import discover_asset_id


# --- discover_asset_id -------------------------------------------------------

def _ckpt(tmp_path: Path, *asset_ids: str) -> Path:
    root = tmp_path / "ckpt"
    (root / "params").mkdir(parents=True)
    for aid in asset_ids:
        d = root / "assets" / aid
        d.mkdir(parents=True)
        (d / "norm_stats.json").write_text(json.dumps({"norm_stats": {}}))
    return root


def test_finds_a_nested_asset_id(tmp_path):
    """The real id is the training repo_id, several levels deep."""
    root = _ckpt(tmp_path, "yubi/fix_actions_tf/frame_souce_table/socks_cup_penholder_box_merged")
    assert discover_asset_id(str(root)) == \
        "yubi/fix_actions_tf/frame_souce_table/socks_cup_penholder_box_merged"


def test_finds_a_flat_asset_id(tmp_path):
    assert discover_asset_id(str(_ckpt(tmp_path, "yubi"))) == "yubi"


def test_missing_assets_dir_names_the_path(tmp_path):
    root = tmp_path / "ckpt"
    (root / "params").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        discover_asset_id(str(root))


def test_ambiguous_asset_id_is_refused_not_guessed(tmp_path):
    """Two candidates must fail loudly — silently picking one would score the
    submission against the wrong normalisation."""
    root = _ckpt(tmp_path, "yubi/a", "yubi/b")
    with pytest.raises(ValueError, match="ambiguous"):
        discover_asset_id(str(root))


# --- entrypoint runtime selection -------------------------------------------
# Drives the real entrypoint.sh via --print-runtime, so the parser under test is
# the one that ships, not a copy of it.

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


def _parse(text: str, tmp_path: Path) -> str:
    (tmp_path / "umi_arena.yaml").write_text(text)
    r = subprocess.run(["bash", str(ENTRYPOINT), "--print-runtime"],
                       env={**os.environ, "UMI_ARENA_SUBMISSION": str(tmp_path)},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


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
    ("runtime: openpi", "openpi"),          # no trailing newline
    ("", ""),
    ("name: x\n", ""),
    ("myruntime: torch\n", ""),            # must not match a suffix
])
def test_runtime_parser(text, expected, tmp_path):
    assert _parse(text, tmp_path) == expected


def test_first_declaration_wins(tmp_path):
    assert _parse("runtime: openpi\nruntime: torch\n", tmp_path) == "openpi"


def test_missing_yaml_yields_empty_not_a_crash(tmp_path):
    r = subprocess.run(["bash", str(ENTRYPOINT), "--print-runtime"],
                       env={**os.environ, "UMI_ARENA_SUBMISSION": str(tmp_path)},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


# --- entrypoint runtime check, before any venv is touched -------------------

def _entrypoint(tmp_path: Path, yaml: str | None, image_runtime: str):
    if yaml is not None:
        (tmp_path / "umi_arena.yaml").write_text(yaml)
    return subprocess.run(["bash", str(ENTRYPOINT)],
                          env={**os.environ, "UMI_ARENA_SUBMISSION": str(tmp_path),
                               "UMI_ARENA_RUNTIME": image_runtime},
                          capture_output=True, text=True, timeout=30)


def test_missing_runtime_defaults_to_openpi(tmp_path):
    r = _entrypoint(tmp_path, None, "openpi")
    assert "defaulting to openpi" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 2
    assert "image is missing executable /opt/venv/openpi/bin/python" in r.stderr


@pytest.mark.parametrize("text", [
    "runtime:\n", "runtime: !!!\n", 'runtime: "torch\n',
    "runtime: torch extra\n", "runtime: torch#comment\n",
])
def test_invalid_runtime_is_refused_without_defaulting(tmp_path, text):
    r = _entrypoint(tmp_path, text, "openpi")
    assert r.returncode == 2
    assert "invalid runtime declaration" in r.stderr
    assert "defaulting" not in r.stdout


def test_missing_runtime_field_defaults_to_openpi(tmp_path):
    r = _entrypoint(tmp_path, "name: example\n", "openpi")
    assert r.returncode == 2
    assert "defaulting to openpi" in r.stdout
    assert "image is missing executable /opt/venv/openpi/bin/python" in r.stderr


def test_missing_runtime_python_has_a_clear_error(tmp_path):
    r = _entrypoint(tmp_path, "runtime: torch\n", "torch")
    assert r.returncode == 2
    assert "image is missing executable /opt/venv/torch/bin/python" in r.stderr


def test_declared_runtime_must_match_the_image(tmp_path):
    r = _entrypoint(tmp_path, "runtime: torch\n", "openpi")
    assert r.returncode == 2
    assert "runtime 'torch'" in r.stderr and "umi-arena-openpi" in r.stderr, r.stderr


def test_default_runtime_is_refused_by_another_image(tmp_path):
    r = _entrypoint(tmp_path, None, "torch")
    assert r.returncode == 2 and "umi-arena-torch" in r.stderr, r.stdout + r.stderr
