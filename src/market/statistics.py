"""HAC errors and paired differences using joint chronological block bootstrap."""

import numpy as np
from scipy.stats import norm


def hac_se(x, lag=None):
    """Newey–West standard error of the mean with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = x - x.mean()
    s = np.sum(d * d) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.sum(d[k:] * d[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def stationary_bootstrap_indices(n, n_boot, mean_block, rng):
    """Politis–Romano stationary bootstrap index matrix (n_boot, n)."""
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    idx = np.empty((n_boot, n), dtype=int)
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
            continue
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def paired_difference(
    result, a="model", b="buy_hold", *, confidence=0.95, n_boot=2000, seed=0, mean_block=None
):
    """Mean of r_a - r_b with HAC t-test and a stationary-bootstrap interval."""
    d = (result[f"r_{a}"] - result[f"r_{b}"]).to_numpy(float)
    n = len(d)
    mean = d.mean()
    se = hac_se(d)
    rng = np.random.default_rng(seed)
    if mean_block is None:
        mean_block = max(2.0, n ** (1 / 3))
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
    boot = d[idx].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    t_stat = mean / se if se > 0 else np.nan
    return {
        "a": a,
        "b": b,
        "months": n,
        "mean_diff": mean,
        "hac_se": se,
        "t_hac": t_stat,
        "p_hac": float(2 * norm.sf(abs(t_stat))) if np.isfinite(t_stat) else np.nan,
        "boot_low": float(low),
        "boot_high": float(high),
        "confidence": confidence,
        "share_better": float(np.mean(d > 0)),
        "share_equal": float(np.mean(d == 0)),
    }
