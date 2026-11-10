"""Prepare complete synthetic boundary grids without extrapolating missing endpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_analysis.curves import Boundaries
from boundary_analysis.monotone import fit_monotone_boundaries
from boundary_analysis.relative import relative_boundaries
from experiments.statistics import load_experiment_statistics
from infrastructure.identity import digest
from infrastructure.storage import atomic_json


def complete_boundary_grid(frame):
    """Internal interpolation only. A missing endpoint makes alpha ineligible."""
    out = frame.sort_values("H").reset_index(drop=True).copy()
    if out.H.duplicated().any() or not np.isfinite(out.H).all() or not out.H.eq(0.5).any():
        raise ValueError("Require unique finite H and analytic H=0.5 anchor")
    for side in (1, 2):
        values = out[f"mu{side}_fit"].to_numpy(float)
        ok = np.isfinite(values)
        if not ok[0] or not ok[-1]:
            raise ValueError(f"mu{side}: missing endpoint; cannot interpolate")
        out[f"mu{side}_imputed"] = ~ok
        out[f"mu{side}"] = np.interp(out.H, out.H[ok], values[ok])
        left = out.loc[out.H.le(0.5), f"mu{side}"].to_numpy()
        right = out.loc[out.H.ge(0.5), f"mu{side}"].to_numpy()
        direction = 1 if side == 1 else -1
        if (
            np.any(direction * np.diff(left) < -1e-7)
            or np.any(direction * np.diff(right) > 1e-7)
            or np.any(out[f"mu{side}"] * direction > 1e-7)
            or abs(out.loc[out.H.eq(0.5), f"mu{side}"].iloc[0]) > 1e-7
        ):
            raise ValueError("Boundary grid violates signs, monotonicity or anchor")
    if (out.mu1 > out.mu2 + 1e-7).any():
        raise ValueError("Crossing boundaries")
    if not np.isfinite(out.epsilon).all() or (out.epsilon < 0).any():
        raise ValueError("Invalid relative tolerance")
    return out


def boundary_object(frame, alpha, label="synthetic frozen grid", source=""):
    return Boundaries(
        frame.H.to_numpy(),
        frame.mu1.to_numpy(),
        frame.mu2.to_numpy(),
        None,
        label,
        "relative",
        "paired",
        alpha,
        source=source,
        epsilon_values=frame.epsilon.to_numpy(),
        statuses={
            s: np.where(frame[f"mu{s}_imputed"], "interpolated", frame[f"mu{s}_fit_status"])
            for s in (1, 2)
        },
    )


def prepare_boundary_grids(run_dir, output, config):
    _, nodes = load_experiment_statistics(run_dir, confidence=config.confidence, gain_mode="paired")
    columns = [
        "H",
        "mu",
        "V",
        "SE",
        "n",
        "complete",
        "paired_gain",
        "paired_SE",
        "paired_n",
        "paired_complete",
    ]
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(nodes[columns], index=False).to_numpy().tobytes()
    ).hexdigest()
    frames, excluded = {}, []
    for alpha in config.alphas:
        try:
            raw = relative_boundaries(
                nodes, T=1.0, alpha=alpha, confidence=config.confidence, gain_mode="paired"
            )
            frame = complete_boundary_grid(fit_monotone_boundaries(raw))
        except ValueError as error:
            excluded.append(dict(alpha=alpha, reason=str(error)))
            continue
        frames[alpha] = frame
    description = dict(
        source_statistics_hash=data_hash,
        frames={str(a): digest(f.to_csv(index=False)) for a, f in frames.items()},
        excluded=excluded,
    )
    manifest = Path(output) / "boundary_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != description:
        raise ValueError("Boundary source/estimates changed; use a new TUNING_ID")
    for alpha, frame in frames.items():
        path = Path(output) / f"boundaries_alpha{alpha:g}.csv"
        if path.exists() and digest(path.read_text()) != description["frames"][str(alpha)]:
            raise ValueError("Cached boundary grid changed")
        if not path.exists():
            path.write_text(frame.to_csv(index=False))
    if not manifest.exists():
        atomic_json(manifest, description)
    pd.DataFrame(excluded, columns=["alpha", "reason"]).to_csv(
        Path(output) / "excluded_alphas.csv", index=False
    )
    if not frames:
        raise ValueError("No alpha has complete supported boundaries; see excluded_alphas.csv")
    return frames
