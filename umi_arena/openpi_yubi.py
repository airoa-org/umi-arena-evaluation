"""Glue for `runtime: openpi` submissions: the YUBI data config and the asset
id its norm stats were saved under."""

from __future__ import annotations

from pathlib import Path

__all__ = ["LeRobotYubiDataConfig", "discover_asset_id"]


def discover_asset_id(checkpoint_dir: str) -> str:
    """Find the asset id a checkpoint's norm stats were saved under.

    openpi defaults the asset id to the training `repo_id`, which a submitter has
    no reason to remember and which we should not make them declare. The stats
    are at <checkpoint>/assets/<asset_id>/norm_stats.json, so just look.
    """
    assets = Path(checkpoint_dir) / "assets"
    hits = sorted(assets.rglob("norm_stats.json"))
    if not hits:
        raise FileNotFoundError(
            f"no norm_stats.json under {assets} — the checkpoint is missing its assets/ directory"
        )
    if len(hits) > 1:
        found = ", ".join(str(h.parent.relative_to(assets)) for h in hits)
        raise ValueError(f"ambiguous asset id, found {len(hits)}: {found}")
    return str(hits[0].parent.relative_to(assets))


def __getattr__(name: str):
    # Deferred so importing this module does not pull in JAX on the torch venv.
    if name == "LeRobotYubiDataConfig":
        from umi_arena.yubi.config import LeRobotYubiDataConfig

        return LeRobotYubiDataConfig
    raise AttributeError(name)
