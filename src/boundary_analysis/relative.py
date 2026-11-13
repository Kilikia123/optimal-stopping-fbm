"""Relative tolerances and left/right boundary localisation from node moments."""

import numpy as np
import pandas as pd
from scipy.stats import t

from boundary_analysis.inversion import invert_band


def relative_boundaries(nodes, *, T, alpha=0.2, confidence=0.95, gain_mode="legacy"):
    """Invert V <= alpha*V(0) and V-mu*T <= alpha*V(0).

    Expected grids are taken from loader-provided DataFrame.attrs; without them
    the union of supplied nodes defines the grid. Intervals enclose grid extrema;
    one adjacent cell is added for a continuous monotone crossing. No covariance
    estimate is needed: all comparisons use one Bonferroni coverage event.
    """
    if (
        not np.isfinite([T, alpha, confidence]).all()
        or T <= 0
        or not 0 < alpha < 1
        or not 0 < confidence < 1
    ):
        raise ValueError("Require T > 0, 0 < alpha < 1, 0 < confidence < 1")
    if gain_mode not in ("legacy", "paired"):
        raise ValueError("gain_mode must be legacy or paired")
    required = ["H", "mu", "V", "SE", "n", "complete"]
    if gain_mode == "paired":
        required += ["paired_gain", "paired_SE", "paired_n", "paired_complete"]
    if not set(required).issubset(nodes.columns):
        raise ValueError("Missing node statistics")
    if nodes.duplicated(["H", "mu"]).any():
        raise ValueError("Duplicate nodes")
    numeric = nodes[["H", "mu", "V", "SE", "n"]].to_numpy(dtype=float)
    if (
        not np.isfinite(numeric).all()
        or (nodes.SE < 0).any()
        or (nodes.n < 2).any()
        or (nodes.n % 1 != 0).any()
        or not nodes.H.between(0, 1, inclusive="neither").all()
    ):
        raise ValueError("Invalid node statistics")
    hs = np.sort(nodes.attrs.get("expected_H", nodes.H.unique()))
    mus = np.sort(nodes.attrs.get("expected_mu", nodes.mu.unique()))
    if (
        len(np.unique(hs)) != len(hs)
        or len(np.unique(mus)) != len(mus)
        or not np.isfinite(hs).all()
        or not np.isfinite(mus).all()
        or not nodes.H.isin(hs).all()
        or not nodes.mu.isin(mus).all()
    ):
        raise ValueError("Invalid expected grids")
    family = max(len(hs) * len(mus), len(nodes), 1)
    rows = []
    for h in hs:
        row = dict(
            H=float(h),
            alpha=alpha,
            confidence=confidence,
            gain_mode=gain_mode,
            epsilon=np.nan,
            epsilon_low=np.nan,
            epsilon_high=np.nan,
            V0=np.nan,
            V0_SE=np.nan,
        )
        for side in (1, 2):
            row.update(
                {
                    f"mu{side}_{key}": np.nan
                    for key in ("raw", "low", "high", "grid_low", "grid_high")
                }
            )
            row[f"mu{side}_status"] = "incomplete_test"
        group = nodes[nodes.H.eq(h)].sort_values("mu")
        if h == 0.5:
            row.update(epsilon=0.0, epsilon_low=0.0, epsilon_high=0.0, V0=0.0, V0_SE=0.0)
            for side in (1, 2):
                for key in ("raw", "low", "high", "grid_low", "grid_high"):
                    row[f"mu{side}_{key}"] = 0.0
                row[f"mu{side}_status"] = "analytic"
        elif not np.any(mus == 0):
            for side in (1, 2):
                row[f"mu{side}_status"] = "missing_zero"
        elif (
            len(group) == len(mus)
            and len(mus) >= 2
            and group.complete.all()
            and np.array_equal(group.mu, mus)
        ):
            means = group.V.to_numpy()
            half = (
                t.isf((1 - confidence) / (2 * family), group.n.to_numpy() - 1) * group.SE.to_numpy()
            )
            low, high = means - half, means + half
            zero = int(np.flatnonzero(mus == 0)[0])
            row.update(
                V0=float(means[zero]),
                V0_SE=float(group.SE.iloc[zero]),
                epsilon=float(alpha * means[zero]),
                epsilon_low=float(alpha * low[zero]),
                epsilon_high=float(alpha * high[zero]),
            )
            for side in (1, 2):
                if means[zero] <= 0:
                    row[f"mu{side}_status"] = "insufficient_precision"
                    continue
                reference_low = low[zero]
                if side == 2 and gain_mode == "paired":
                    if not group.paired_complete.all():
                        row["mu2_status"] = "incomplete_paired_test"
                        continue
                    pg, ps, pn = (
                        group[name].to_numpy() for name in ("paired_gain", "paired_SE", "paired_n")
                    )
                    if (
                        not np.isfinite([pg, ps, pn]).all()
                        or np.any(ps < 0)
                        or np.any(pn != group.n.to_numpy())
                    ):
                        raise ValueError("Invalid paired statistics")
                    # A separate simultaneous event for the right boundary:
                    # paired gains plus reference values. Left analysis is unchanged.
                    paired_half = t.isf((1 - confidence) / (4 * family), pn - 1) * ps
                    ref_half = float(
                        t.isf((1 - confidence) / (4 * family), group.n.iloc[zero] - 1)
                        * group.SE.iloc[zero]
                    )
                    ref_low, ref_high = means[zero] - ref_half, means[zero] + ref_half
                    residual = pg - alpha * means[zero]
                    lower, upper = (
                        pg - paired_half - alpha * ref_high,
                        pg + paired_half - alpha * ref_low,
                    )
                    reference_low = ref_low
                    # At mu=0 the paired gain is V(0)-B_T, NOT V(0).
                    # Do not apply the legacy (1-alpha)*V(0) shortcut here.
                else:
                    shift = 0 if side == 1 else mus * T
                    residual = means - shift - alpha * means[zero]
                    lower, upper = (
                        low - shift - alpha * high[zero],
                        high - shift - alpha * low[zero],
                    )
                    lower[zero], upper[zero] = (1 - alpha) * low[zero], (1 - alpha) * high[zero]
                result = invert_band(mus, residual, lower, upper, 0.0, side)
                row[f"mu{side}_status"] = result["status"]
                # A point crossing can exist even when the band cannot locate it.
                # Require an enclosed single transition, not an edge extremum.
                inside = residual <= 0
                enclosed_point = (
                    (inside[0] and not inside[-1]) if side == 1 else (not inside[0] and inside[-1])
                )
                if result["estimate"] is not None and enclosed_point:
                    row[f"mu{side}_raw"] = result["estimate"]
                if reference_low <= 0 and result["status"] != "multiple_transitions":
                    row[f"mu{side}_status"] = "insufficient_precision"
                if row[f"mu{side}_status"] == "localized_grid":
                    lo, hi = result["mu_low"], result["mu_high"]
                    row[f"mu{side}_grid_low"], row[f"mu{side}_grid_high"] = lo, hi
                    # Add only the adjacent grid cell, not a statistical margin.
                    if side == 1:
                        hi = mus[min(np.searchsorted(mus, hi) + 1, len(mus) - 1)]
                    else:
                        lo = mus[max(np.searchsorted(mus, lo) - 1, 0)]
                    row[f"mu{side}_low"], row[f"mu{side}_high"] = float(lo), float(hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No H grid supplied")
    out.attrs.update(
        T=T,
        confidence=confidence,
        family_size=family,
        gain_mode=gain_mode,
        interpretation="policy_mean; monotone-crossing assumption for cell envelopes",
    )
    return out
