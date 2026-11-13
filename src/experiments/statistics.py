"""Read saved reward moments and derive node statistics without training or simulation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_analysis.inversion import invert_band
from experiments.configuration import _cfg
from experiments.moments import Moments
from experiments.paired import _attach_paired_statistics
from infrastructure.identity import digest


def _record(h, mu, moments, stop0, validation, cfg, manifest, confidence=None):
    statistics = {
        "n": moments.n,
        "V": moments.mean,
        "SE": float(np.sqrt(max(0.0, moments.m2) / (moments.n - 1) / moments.n)),
    }
    if confidence is not None:
        statistics = moments.intervals(confidence, manifest["family_size"])
    row = {
        "H": h,
        "mu": mu,
        **statistics,
        "complete": moments.n == cfg.M_test,
        "stopped_at_zero": stop0,
        "validation_mean": validation,
    }
    row["excess"] = row["V"] - mu * cfg.T
    for name in ("ci_low", "ci_high", "band_low", "band_high"):
        if name in row:
            row["excess_" + name] = row[name] - mu * cfg.T
    return row


def _load_experiment_statistics(run_dir, confidence):
    directory = Path(run_dir)
    manifest = json.loads((directory / "experiment.json").read_text())
    cfg = _cfg(manifest)
    identity = digest(manifest)
    rows = []
    for path in (directory / "states").glob("*.json"):
        state = json.loads(path.read_text())
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        moments = Moments(**state["moments"])
        if moments.n >= 2:
            rows.append(
                _record(
                    state["H"],
                    state["mu"],
                    moments,
                    state["stopped_at_zero"],
                    state["validation_mean"],
                    cfg,
                    manifest,
                    confidence,
                )
            )
    columns = [
        "H",
        "mu",
        "n",
        "V",
        "SE",
        "ci_low",
        "ci_high",
        "band_low",
        "band_high",
        "complete",
        "stopped_at_zero",
        "validation_mean",
        "excess",
        "excess_ci_low",
        "excess_ci_high",
        "excess_band_low",
        "excess_band_high",
    ]
    if confidence is None:
        columns = [
            "H",
            "mu",
            "n",
            "V",
            "SE",
            "complete",
            "stopped_at_zero",
            "validation_mean",
            "excess",
        ]
    nodes = pd.DataFrame(rows, columns=columns).sort_values(["H", "mu"]).reset_index(drop=True)
    nodes.attrs.update(
        expected_H=list(cfg.H_grid),
        expected_mu=list(cfg.mu_grid),
        family_size=manifest["family_size"],
        confidence=confidence,
    )
    return manifest, nodes


def load_experiment_statistics(run_dir, *, confidence=0.95, gain_mode="legacy"):
    """Read saved moments; optionally attach verified paired statistics, never fit."""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if gain_mode not in ("legacy", "paired"):
        raise ValueError("gain_mode must be legacy or paired")
    manifest, nodes = _load_experiment_statistics(run_dir, confidence)
    nodes.attrs["gain_mode"] = gain_mode
    if gain_mode == "paired":
        nodes = _attach_paired_statistics(Path(run_dir), manifest, nodes, confidence)
    return manifest, nodes


def load_experiment(run_dir, epsilon=0.03, *, confidence=0.95):
    """Analyze saved moments; epsilon=None loads raw estimates without intervals."""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be positive and finite")
    manifest, nodes = _load_experiment_statistics(
        run_dir, confidence if epsilon is not None else None
    )
    cfg = _cfg(manifest)
    threshold = epsilon
    if threshold is None:
        return manifest, nodes, pd.DataFrame()
    boundaries = []
    for h in cfg.H_grid:
        view = nodes[nodes.H.eq(h)].sort_values("mu")
        for side in (1, 2):
            row = {
                "H": h,
                "side": side,
                "epsilon": threshold,
                "old_mu": None,
                "estimate": None,
                "mu_low": None,
                "mu_high": None,
                "window_low": min(cfg.mu_grid),
                "window_high": max(cfg.mu_grid),
                "status": "incomplete_test",
            }
            if len(view) == len(cfg.mu_grid) and view.complete.all() and len(view) >= 2:
                prefix = "" if side == 1 else "excess_"
                row.update(
                    invert_band(
                        view.mu,
                        view["V" if side == 1 else "excess"],
                        view[prefix + "band_low"],
                        view[prefix + "band_high"],
                        threshold,
                        side,
                    )
                )
            boundaries.append(row)
    return manifest, nodes, pd.DataFrame(boundaries)
