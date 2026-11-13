"""Load saved policies and values; select the nearest trained parameter node."""

import json
from pathlib import Path

import numpy as np

from experiments.configuration import _cfg
from experiments.policy import FixedPolicy
from experiments.statistics import load_experiment
from infrastructure.identity import digest, node_key


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
        self.identity = digest(self.manifest)
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
        return (self.run_dir / "policies" / f"{node_key(h, mu)}.json").exists()

    def policy(self, h, mu):
        node = (float(h), float(mu))
        if node not in self._cache:
            path = self.run_dir / "policies" / f"{node_key(h, mu)}.json"
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
        return cls(
            nodes[nodes["complete"]],
            manifest["config"]["H_grid"],
            manifest["config"]["mu_grid"],
            Path(run_dir).name if label is None else label,
        )

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
