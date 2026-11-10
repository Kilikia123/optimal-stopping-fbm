"""R/S and DFA estimates of H from centred daily increments."""

import math

import numpy as np

H_MIN, H_MAX = 0.06, 0.98


def _block_sizes(n, min_n, max_n, count):
    max_n = n // 2 if max_n is None else min(max_n, n // 2)
    if max_n < min_n:
        raise ValueError(f"Series of length {n} is too short for block sizes >= {min_n}")
    sizes = np.unique(np.round(np.geomspace(min_n, max_n, count)).astype(int))
    return sizes[(sizes >= min_n) & (sizes <= max_n)]


def expected_rs(n):
    """Anis–Lloyd expectation of R/S for i.i.d. normal data of length n (Peters' form)."""
    n = int(n)
    i = np.arange(1, n)
    total = np.sum(np.sqrt((n - i) / i))
    if n <= 340:
        front = math.exp(math.lgamma((n - 1) / 2) - math.lgamma(n / 2)) / math.sqrt(math.pi)
    else:
        front = 1.0 / math.sqrt(n * math.pi / 2)
    return (n - 0.5) / n * front * total


def rs_statistic(x, size):
    """Mean rescaled range over non-overlapping blocks of the given size."""
    x = np.asarray(x, dtype=float)
    m = len(x) // size
    blocks = x[: m * size].reshape(m, size)
    dev = blocks - blocks.mean(axis=1, keepdims=True)
    cum = np.cumsum(dev, axis=1)
    rng = cum.max(axis=1) - np.minimum(cum.min(axis=1), 0.0)
    std = blocks.std(axis=1, ddof=0)
    keep = std > 0
    if not keep.any():
        return np.nan
    return float(np.mean(rng[keep] / std[keep]))


def rs_hurst(x, *, min_n=8, max_n=None, count=20, corrected=True):
    """Hurst exponent by classical R/S analysis of an increment series.

    The slope of log E[R/S](n) on log n estimates H. With ``corrected=True``
    the Anis–Lloyd expectation under independence is subtracted first and 0.5
    added back, which removes most of the small-sample upward bias.
    """
    x = np.asarray(x, dtype=float)
    sizes = _block_sizes(len(x), min_n, max_n, count)
    rs = np.array([rs_statistic(x, s) for s in sizes])
    ok = np.isfinite(rs) & (rs > 0)
    if ok.sum() < 3:
        raise ValueError("Not enough valid block sizes for the R/S regression")
    log_n = np.log(sizes[ok])
    y = np.log(rs[ok])
    if corrected:
        y = y - np.log([expected_rs(s) for s in sizes[ok]])
    slope = np.polyfit(log_n, y, 1)[0]
    return float(slope + 0.5) if corrected else float(slope)


def dfa_hurst(x, *, min_n=8, max_n=None, count=20, order=1):
    """Detrended fluctuation analysis of an increment series (alpha = H for fGn)."""
    x = np.asarray(x, dtype=float)
    profile = np.cumsum(x - x.mean())
    sizes = _block_sizes(len(x), max(min_n, order + 2), max_n, count)
    fluct = []
    for s in sizes:
        m = len(profile) // s
        segments = profile[: m * s].reshape(m, s)
        k = np.arange(s)
        coef = np.polyfit(k, segments.T, order)
        trend = np.polyval(coef, k[:, None]).T
        fluct.append(np.sqrt(np.mean((segments - trend) ** 2)))
    fluct = np.asarray(fluct)
    ok = fluct > 0
    slope = np.polyfit(np.log(sizes[ok]), np.log(fluct[ok]), 1)[0]
    return float(slope)


HURST_METHODS = {"rs": rs_hurst, "dfa": dfa_hurst}
