"""Saved stopping rules, practical boundaries and value table of a grid run.

A run directory written by ``osfbm.experiment.run_experiment`` contains
``experiment.json`` (grids and configuration), ``policies/<key>.json`` (Ridge
coefficients per exercise date) and ``states/<key>.json`` (test moments).
This module turns them into three read-only lookups used by the backtest:

* ``PolicyBank`` — nearest trained rule for an estimated pair (H, mu);
* ``Boundaries`` — practical boundaries mu_1(H), mu_2(H) for a tolerance eps;
* ``ValueTable`` — estimated value V(H, mu) of the trained rule, for the
  theoretical prediction of the expected reward.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osfbm.confidence import FixedPolicy, _digest, _key
from osfbm.experiment import load_experiment, _cfg

CASH, HOLD, POLICY = "cash", "hold", "policy"


def _nearest(grid, value):
    grid = np.asarray(grid, dtype=float)
    return float(grid[int(np.argmin(np.abs(grid - value)))])


class PolicyBank:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        manifest_path = self.run_dir / "experiment.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No experiment.json in {self.run_dir}: run 03-grid first")
        self.manifest = json.loads(manifest_path.read_text())
        self.identity = _digest(self.manifest)
        self.cfg = _cfg(self.manifest)
        self.H_grid = np.sort(np.asarray(self.cfg.H_grid, dtype=float))
        self.mu_grid = np.sort(np.asarray(self.cfg.mu_grid, dtype=float))
        self._cache = {}

    @property
    def D(self):
        return self.cfg.n_exercise

    def nearest_node(self, H, mu):
        return _nearest(self.H_grid, H), _nearest(self.mu_grid, mu)

    def has_policy(self, h, mu):
        return (self.run_dir / "policies" / f"{_key(h, mu)}.json").exists()

    def policy(self, h, mu):
        node = (float(h), float(mu))
        if node not in self._cache:
            path = self.run_dir / "policies" / f"{_key(h, mu)}.json"
            if not path.exists():
                raise FileNotFoundError(f"No trained policy for H={h}, mu={mu} in {self.run_dir}")
            artifact = json.loads(path.read_text())
            if artifact["identity"] != self.identity:
                raise ValueError("Policy belongs to a different experiment")
            self._cache[node] = FixedPolicy.from_dict(artifact["policy"])
        return self._cache[node]

    def nearest(self, H, mu):
        h, m = self.nearest_node(H, mu)
        return h, m, self.policy(h, m)

    def coverage(self):
        """Fraction of grid nodes with a saved policy."""
        total = len(self.H_grid) * len(self.mu_grid)
        found = sum(self.has_policy(h, m) for h in self.H_grid for m in self.mu_grid)
        return found / total


def point_boundaries(nodes, mu_grid, epsilon):
    """Practical boundaries per H from a node table, as in 04-grid-analysis.

    mu_1: last grid node with V <= eps before leaving the region;
    mu_2: first node with V - mu*T <= eps after entering it. NaN when the
    transition is not enclosed by the grid or occurs more than once.
    """
    expected = np.sort(np.asarray(mu_grid, dtype=float))
    rows = []
    for h, group in nodes.groupby("H"):
        group = group.sort_values("mu")
        row = {"H": float(h), "mu1": np.nan, "mu2": np.nan, "mu1_status": "", "mu2_status": ""}
        complete = (len(group) == len(expected) and bool(group["complete"].all())
                    and np.allclose(group["mu"].to_numpy(), expected))
        for side in (1, 2):
            if not complete:
                row[f"mu{side}_status"] = "incomplete"
                continue
            inside = (group["V"] if side == 1 else group["excess"]).to_numpy() <= epsilon
            changes = np.diff(inside.astype(int))
            if np.any(changes > 0 if side == 1 else changes < 0):
                row[f"mu{side}_status"] = "multiple_transitions"
                continue
            idx = np.flatnonzero(inside)
            if not len(idx):
                row[f"mu{side}_status"] = "expand_left" if side == 1 else "expand_right"
                continue
            i = int(idx[-1] if side == 1 else idx[0])
            if (side == 1 and i == len(group) - 1) or (side == 2 and i == 0):
                row[f"mu{side}_status"] = "expand_right" if side == 1 else "expand_left"
                continue
            row[f"mu{side}"] = float(group["mu"].iloc[i])
            row[f"mu{side}_status"] = "found"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("H").reset_index(drop=True)


@dataclass(frozen=True)
class Boundaries:
    """mu_1(H), mu_2(H) interpolated linearly in H; missing mu_2 means +inf."""
    H: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    epsilon: float
    label: str = ""

    @classmethod
    def from_frame(cls, frame, epsilon, label=""):
        frame = frame.sort_values("H")
        return cls(frame["H"].to_numpy(float), frame["mu1"].to_numpy(float),
                   frame["mu2"].to_numpy(float), float(epsilon), label)

    @classmethod
    def from_csv(cls, path, epsilon, label=None):
        path = Path(path)
        return cls.from_frame(pd.read_csv(path), epsilon, path.parent.name if label is None else label)

    @classmethod
    def from_run(cls, run_dir, epsilon, label=None):
        manifest, nodes, _ = load_experiment(run_dir, epsilon=None)
        frame = point_boundaries(nodes, manifest["config"]["mu_grid"], epsilon)
        return cls.from_frame(frame, epsilon, Path(run_dir).name if label is None else label)

    @classmethod
    def brownian(cls, epsilon=0.0, label="H=1/2"):
        """Exact Brownian boundaries mu_1 = mu_2 = 0 for every H."""
        H = np.array([0.0, 1.0])
        return cls(H, np.zeros(2), np.zeros(2), float(epsilon), label)

    def _interp(self, values, H, missing):
        """Linear in H inside the table; clamped to the table's H range.

        A boundary that is NaN at an edge of the table (not localised on the
        mu grid) is treated as ``missing`` for every H beyond the last finite node.
        """
        H = np.clip(np.asarray(H, dtype=float), self.H[0], self.H[-1])
        ok = np.isfinite(values)
        if not ok.any():
            return np.full_like(H, missing)
        Hs, vs = self.H[ok], values[ok]
        out = np.interp(H, Hs, vs)
        return np.where((H < Hs[0] - 1e-12) | (H > Hs[-1] + 1e-12), missing, out)

    def mu1_at(self, H):
        return self._interp(self.mu1, H, -np.inf)

    def mu2_at(self, H):
        return self._interp(self.mu2, H, np.inf)

    def regime(self, H, mu):
        if mu <= self.mu1_at(H):
            return CASH
        if mu >= self.mu2_at(H):
            return HOLD
        return POLICY

    def frame(self):
        return pd.DataFrame({"H": self.H, "mu1": self.mu1, "mu2": self.mu2})


class ValueTable:
    """Nearest-node lookup of the trained rule's value V(H, mu) and its SE."""

    def __init__(self, nodes, H_grid, mu_grid, label=""):
        self.H_grid = np.sort(np.asarray(H_grid, dtype=float))
        self.mu_grid = np.sort(np.asarray(mu_grid, dtype=float))
        table = nodes.pivot(index="H", columns="mu", values="V")
        self.V = table.reindex(index=self.H_grid, columns=self.mu_grid)
        se = nodes.pivot(index="H", columns="mu", values="SE")
        self.SE = se.reindex(index=self.H_grid, columns=self.mu_grid)
        self.label = label

    @classmethod
    def from_run(cls, run_dir, label=None):
        manifest, nodes, _ = load_experiment(run_dir, epsilon=None)
        return cls(nodes[nodes["complete"]], manifest["config"]["H_grid"],
                   manifest["config"]["mu_grid"], Path(run_dir).name if label is None else label)

    def nearest_node(self, H, mu):
        return _nearest(self.H_grid, H), _nearest(self.mu_grid, mu)

    def value(self, H, mu):
        h, m = self.nearest_node(H, mu)
        v = self.V.loc[h, m]
        return float(v) if np.isfinite(v) else np.nan

    def se(self, H, mu):
        h, m = self.nearest_node(H, mu)
        s = self.SE.loc[h, m]
        return float(s) if np.isfinite(s) else np.nan
