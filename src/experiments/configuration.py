"""Validate experiment settings and reserve deterministic independent seeds."""

import json
from pathlib import Path

import numpy as np

from config import Config
from infrastructure.identity import dependency_versions, digest
from infrastructure.provenance import implementation_fingerprint
from infrastructure.storage import atomic_json


def prepare_experiment(run_dir, cfg, *, batch_size=2_000, M_validation=10_000):
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (batch_size, M_validation)
    ):
        raise ValueError("batch_size and M_validation must be positive integers")
    directory = Path(run_dir)
    if cfg.M_train < 2 or cfg.M_test < 2:
        raise ValueError("Training and test sizes must be at least two")
    if (
        not cfg.H_grid
        or not cfg.mu_grid
        or len(set(cfg.H_grid)) != len(cfg.H_grid)
        or len(set(cfg.mu_grid)) != len(cfg.mu_grid)
        or not all(0 < h < 1 for h in cfg.H_grid)
        or not np.isfinite(cfg.mu_grid).all()
    ):
        raise ValueError("Require unique finite grids and 0 < H < 1")
    if (directory / "config.json").exists() and not (directory / "experiment.json").exists():
        raise ValueError("Existing legacy run: choose a new RUN_ID")
    used, seeds = set(), {}
    for h in cfg.H_grid:
        seeds[str(float(h))] = {}
        for role in ("train", "validation", "test"):
            seed = int(digest(["fresh-experiment", cfg.seed, h, role])[:8], 16)
            while seed in used:
                seed = (seed + 1) % 2**32
            used.add(seed)
            seeds[str(float(h))][role] = seed
    manifest = {
        "schema_version": 2,
        "kind": "fresh_grid",
        "config": cfg.to_dict(),
        "execution": {"batch_size": batch_size, "M_validation": M_validation},
        "seeds": seeds,
        "versions": dependency_versions(),
        "implementation": implementation_fingerprint("training"),
        "family_size": len(cfg.H_grid) * len(cfg.mu_grid),
    }
    path = directory / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Experiment parameters/environment changed: use a new RUN_ID")
    else:
        atomic_json(path, manifest)
        atomic_json(directory / "config.json", cfg.to_dict())
    return manifest


def _cfg(manifest):
    raw = dict(manifest["config"])
    for name in ("H_grid", "mu_grid"):
        raw[name] = tuple(raw[name])
    return Config(**raw)
