"""Boundary tables, regime classification and interpolation in H."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.statistics import load_experiment
from infrastructure.identity import digest

CASH, HOLD, POLICY = "cash", "hold", "policy"


def point_boundaries(nodes, mu_grid, epsilon):
    """Practical boundaries per H from a node table, from saved experiment statistics.

    mu_1: last grid node with V <= eps before leaving the region;
    mu_2: first node with V - mu*T <= eps after entering it. NaN when the
    transition is not enclosed by the grid or occurs more than once.
    """
    expected = np.sort(np.asarray(mu_grid, dtype=float))
    rows = []
    for h, group in nodes.groupby("H"):
        group = group.sort_values("mu")
        row = {"H": float(h), "mu1": np.nan, "mu2": np.nan, "mu1_status": "", "mu2_status": ""}
        complete = (
            len(group) == len(expected)
            and bool(group["complete"].all())
            and np.allclose(group["mu"].to_numpy(), expected)
        )
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
    epsilon: float | None
    label: str = ""
    mode: str = "fixed"
    gain_mode: str = "legacy"
    alpha: float | None = None
    source: str = ""
    epsilon_values: np.ndarray | None = None
    statuses: dict = field(default_factory=dict)

    @classmethod
    def from_analysis(
        cls,
        run_dir,
        *,
        boundary_mode="relative",
        gain_mode="paired",
        alpha=0.2,
        epsilon=0.06,
        confidence=0.95,
    ):
        """Load an existing 04 export only; no fitting, simulation or replay."""
        directory = Path(run_dir)
        if boundary_mode not in ("fixed", "relative") or (
            boundary_mode == "relative" and gain_mode not in ("legacy", "paired")
        ):
            raise ValueError("Unknown boundary or gain mode")
        if (
            not 0 < confidence < 1
            or (boundary_mode == "relative" and not 0 < alpha < 1)
            or (boundary_mode == "fixed" and (not np.isfinite(epsilon) or epsilon <= 0))
        ):
            raise ValueError("Invalid confidence, alpha or epsilon")
        tag = f"epsilon_{epsilon:g}" if boundary_mode == "fixed" else f"alpha_{alpha:g}"
        branch = (
            ""
            if boundary_mode == "fixed"
            else ("relative_paired" if gain_mode == "paired" else "relative")
        )
        path = (
            directory / "analysis" / branch / tag / f"confidence_{confidence:g}" / "boundaries.csv"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Нет границ {path}. Подготовьте выгрузку через analyze_relative_run с выбранными параметрами."
            )
        frame = pd.read_csv(path).sort_values("H").reset_index(drop=True)
        manifest = json.loads((directory / "experiment.json").read_text())
        if (
            frame.H.duplicated().any()
            or not np.isfinite(frame.H).all()
            or not np.array_equal(frame.H.to_numpy(), np.sort(manifest["config"]["H_grid"]))
        ):
            raise ValueError("Boundary H grid differs from source experiment")
        if boundary_mode == "fixed":
            obj = cls.from_frame(frame, epsilon, f"{directory.name}; fixed; ε={epsilon:g}")
            return cls(obj.H, obj.mu1, obj.mu2, obj.epsilon, obj.label, source=str(path))
        metadata = json.loads((path.parent / "metadata.json").read_text())
        if (
            metadata.get("experiment_identity") != digest(manifest)
            or metadata.get("gain_mode", "legacy") != gain_mode
            or metadata.get("alpha") != alpha
            or metadata.get("confidence") != confidence
        ):
            raise ValueError("Boundary metadata differs from selected experiment/settings")
        for side in (1, 2):
            values = frame[f"mu{side}_fit"].to_numpy(float)
            allowed = frame[f"mu{side}_fit_status"].isin(
                ["fitted", "fitted_unconfirmed", "analytic"]
            )
            if np.isinf(values).any() or (np.isfinite(values) & ~allowed).any():
                raise ValueError("Invalid fitted boundary status/value")
        return cls(
            frame.H.to_numpy(float),
            frame.mu1_fit.to_numpy(float),
            frame.mu2_fit.to_numpy(float),
            None,
            f"{directory.name}; relative/{gain_mode}; α={alpha:g}",
            boundary_mode,
            gain_mode,
            alpha,
            str(path),
            frame.epsilon.to_numpy(float),
            {s: frame[f"mu{s}_fit_status"].to_numpy() for s in (1, 2)},
        )

    @property
    def analysis_tag(self):
        if self.mode == "fixed":
            return f"fixed_eps{self.epsilon:g}"
        return f"relative_{self.gain_mode}_alpha{self.alpha:g}"

    def epsilon_at(self, H):
        if self.epsilon_values is None:
            return np.full_like(np.asarray(H, dtype=float), self.epsilon, dtype=float)
        return self._interp(self.epsilon_values, H, np.nan)

    def diagnostics(self):
        rows = []
        for side, values in ((1, self.mu1), (2, self.mu2)):
            status = np.asarray(self.statuses.get(side, np.full(len(self.H), "fitted")))
            good = np.flatnonzero(np.isfinite(values) & (status != "interpolated"))
            bridges = [
                (float(self.H[a]), float(self.H[b]))
                for a, b in zip(good[:-1], good[1:])
                if b > a + 1
            ]
            rows.append(
                dict(
                    side=side,
                    missing=int((~np.isfinite(values)).sum()),
                    imputed=int(np.sum(status == "interpolated")),
                    unconfirmed=int(
                        np.sum(np.asarray(self.statuses.get(side, [])) == "fitted_unconfirmed")
                    ),
                    interpolated_gaps=bridges,
                )
            )
        return pd.DataFrame(rows)

    def plot(self, ax, *, colors=("#2a78d6", "#eb6834")):
        """Dashes identify unconfirmed nodes and interpolation across gaps."""
        for side, values, color in ((1, self.mu1, colors[0]), (2, self.mu2, colors[1])):
            ids = np.flatnonzero(np.isfinite(values))
            status = self.statuses.get(side, np.full(len(self.H), "fitted"))
            uncertain = np.isin(status, ["fitted_unconfirmed", "interpolated"])
            strong = ids[~uncertain[ids]]
            weak = ids[uncertain[ids]]
            ax.scatter(self.H[strong], values[strong], color=color, s=10, label=f"μ{side}(H)")
            ax.scatter(self.H[weak], values[weak], facecolors="none", edgecolors=color, s=18)
            for a, b in zip(ids[:-1], ids[1:]):
                dashed = b > a + 1 or uncertain[a] or uncertain[b]
                ax.plot(self.H[[a, b]], values[[a, b]], "--" if dashed else "-", color=color)

    @classmethod
    def from_frame(cls, frame, epsilon, label=""):
        frame = frame.sort_values("H")
        return cls(
            frame["H"].to_numpy(float),
            frame["mu1"].to_numpy(float),
            frame["mu2"].to_numpy(float),
            float(epsilon),
            label,
        )

    @classmethod
    def from_csv(cls, path, epsilon, label=None):
        path = Path(path)
        return cls.from_frame(
            pd.read_csv(path), epsilon, path.parent.name if label is None else label
        )

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
