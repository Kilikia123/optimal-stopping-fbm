"""Compare the complete strategy with holding across groups of estimated H."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.storage import atomic_json
from market.statistics import stationary_bootstrap_indices


def load_hurst_blocks(output):
    """Read only factual rewards of the complete strategy from the original backtest."""
    return pd.read_csv(Path(output) / "blocks.csv")


def summarize_hurst_performance(
    rows,
    *,
    distance_edges=(0.0, 0.025, 0.05, 0.10, np.inf),
    side_threshold=0.05,
    contrast_widths=(0.025, 0.05, 0.10),
    confidence=0.95,
    n_boot=10000,
    seed=7,
    mean_block=None,
):
    """Joint calendar bootstrap, followed by subgroup conditioning in every draw.

    Rewards are log returns of the complete strategy (boundaries plus model).
    Contrasts compare its advantage over buy-and-hold farther from / nearer to
    H=0.5; they do not modify the strategy. No threshold is selected here.
    """
    required = {"H", "decision_idx", "end_idx", "r_model", "r_buy_hold"}
    if not required <= set(rows):
        raise ValueError(f"Missing backtest columns: {sorted(required - set(rows))}")
    n = len(rows)
    edges = np.asarray(distance_edges, dtype=float)
    widths = np.asarray(contrast_widths, dtype=float)
    if (
        n < 2
        or not 0 < confidence < 1
        or not isinstance(n_boot, int)
        or isinstance(n_boot, bool)
        or n_boot < 20
    ):
        raise ValueError("Require >=2 blocks, valid confidence and n_boot>=20")
    if (
        edges.ndim != 1
        or len(edges) < 2
        or edges[0] != 0
        or not np.isposinf(edges[-1])
        or not np.isfinite(edges[:-1]).all()
        or not (np.diff(edges) > 0).all()
    ):
        raise ValueError("Distance edges must increase from zero to infinity")
    if (
        widths.ndim != 1
        or not len(widths)
        or not np.isfinite(widths).all()
        or not ((widths > 0) & (widths < 0.5)).all()
        or not (np.diff(widths) > 0).all()
        or not 0 < side_threshold < 0.5
    ):
        raise ValueError("Require increasing contrast widths and side threshold in (0,.5)")
    strategies = ["model"]
    cols = ["H", "decision_idx", "end_idx", "r_buy_hold"] + [f"r_{s}" for s in strategies]
    if not np.isfinite(rows[cols].to_numpy(float)).all():
        raise ValueError("Nonfinite estimates or rewards")
    starts = rows.decision_idx.to_numpy(float)
    ends = rows.end_idx.to_numpy(float)
    if (
        not ((rows.H > 0) & (rows.H < 1)).all()
        or np.any(starts % 1)
        or np.any(ends % 1)
        or np.any(starts < 0)
        or np.any(ends <= starts)
        or not np.all(ends - starts == ends[0] - starts[0])
        or not np.array_equal(starts[1:], ends[:-1])
    ):
        raise ValueError("Require valid H and full consecutive equal-length chronology")
    mean_block = max(2.0, n ** (1 / 3)) if mean_block is None else mean_block
    if not np.isfinite(mean_block) or mean_block < 1:
        raise ValueError("mean_block must be finite and >=1")
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, np.random.default_rng(seed))
    h = rows.H.to_numpy(float)
    distance = np.abs(h - 0.5)
    hold = rows.r_buy_hold.to_numpy(float)
    differences = {s: rows[f"r_{s}"].to_numpy(float) - hold for s in strategies}
    sampled = {s: d[idx] for s, d in differences.items()}
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]

    def interval(values, count):
        valid = values[np.isfinite(values)]
        supported = count >= 2 and len(valid) >= max(20, n_boot // 2)
        low, high = np.quantile(valid, quantiles) if supported else (np.nan, np.nan)
        return dict(
            boot_low=low,
            boot_high=high,
            valid_bootstraps=len(valid),
            status="estimated" if supported else "insufficient_data",
        )

    def conditional(values, mask):
        sm = mask[idx]
        denominator = sm.sum(axis=1)
        return np.divide(
            np.where(sm, values, 0.0).sum(axis=1),
            denominator,
            out=np.full(n_boot, np.nan),
            where=denominator > 0,
        )

    definitions = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"{lo:g} ≤ |H−0.5| < {hi:g}" if np.isfinite(hi) else f"|H−0.5| ≥ {lo:g}"
        definitions.append(("distance", label, (distance >= lo) & (distance < hi)))
    lower, upper = 0.5 - side_threshold, 0.5 + side_threshold
    definitions.extend(
        [
            ("side", f"H < {lower:g}", h < lower),
            ("side", f"{lower:g} ≤ H ≤ {upper:g}", (h >= lower) & (h <= upper)),
            ("side", f"H > {upper:g}", h > upper),
        ]
    )
    groups = []
    for family, label, mask in definitions:
        count = int(mask.sum())
        for strategy, d in differences.items():
            groups.append(
                dict(
                    family=family,
                    group=label,
                    strategy=strategy,
                    n=count,
                    mean_hold=hold[mask].mean() if count else np.nan,
                    mean_strategy=rows.loc[mask, f"r_{strategy}"].mean() if count else np.nan,
                    mean_diff=d[mask].mean() if count else np.nan,
                    gap_contribution=d[mask].sum() / n,
                    share_different=np.mean(np.abs(d[mask]) > 1e-10) if count else np.nan,
                    confidence=confidence,
                    **interval(conditional(sampled[strategy], mask), count),
                )
            )
    contrasts = []
    for width in widths:
        # Equality belongs to the outside, as in the distance bins above.
        outside = distance >= width
        for strategy, delta in differences.items():
            far = conditional(sampled[strategy], outside)
            near = conditional(sampled[strategy], ~outside)
            count = min(int(outside.sum()), int((~outside).sum()))
            contrasts.append(
                dict(
                    distance_threshold=width,
                    strategy=strategy,
                    n_near=int((~outside).sum()),
                    n_far=int(outside.sum()),
                    far_minus_near=(
                        (delta[outside].mean() - delta[~outside].mean()) if count else np.nan
                    ),
                    confidence=confidence,
                    **interval(far - near, count),
                )
            )
    return dict(groups=pd.DataFrame(groups), contrasts=pd.DataFrame(contrasts))


def save_hurst_tables(output, tables, *, settings):
    """Store diagnostics of the complete strategy separately from its backtest."""
    output = Path(output)
    folder = output / "hurst_diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        table.to_csv(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = output / "blocks.csv"
    atomic_json(
        folder / "metadata.json",
        dict(
            schema_version=2,
            settings=settings,
            strategy="boundaries_plus_model",
            backtest_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            outputs=hashes,
            interpretation="Post-hoc estimated-H analysis; no threshold selected",
        ),
    )
    return folder
