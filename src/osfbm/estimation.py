"""Parameter estimation for S_t = S_0 exp(a t + sigma B^H_t) from daily closes.

Time is measured in months of ``D`` trading days, so one day is dt = 1/D.
From a window of W daily log returns r_1..r_W:

    a      = D * mean(r)                      drift per month
    H      = R/S (or DFA) exponent of e_t = r_t - mean(r), the increments of
             the detrended log price
    sigma  = std(r) * D**H                    because std(sigma * (B^H_{t+dt} - B^H_t))
                                              = sigma * dt**H = sigma * D**(-H)
    mu     = a / sigma                        normalised drift of the stopping problem

The sign of the exponent in ``sigma`` follows from self-similarity of fBm and is
checked on simulated data in tests/test_estimation.py.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from osfbm.adapter import silenced
from osfbm.vendor.FBM_package import FBM

H_MIN, H_MAX = 0.06, 0.98


@dataclass(frozen=True)
class ParamEstimate:
    a: float
    sigma: float
    H: float
    mu: float
    H_raw: float
    drift_daily: float
    std_daily: float
    n_returns: int
    D: int
    method: str

    def as_row(self):
        return asdict(self)


def log_returns(logp):
    logp = np.asarray(logp, dtype=float)
    if logp.ndim != 1 or len(logp) < 3 or not np.all(np.isfinite(logp)):
        raise ValueError("logp must be a finite 1-d array with at least three prices")
    return np.diff(logp)


def detrend(logp, drift_daily=None):
    """Log price minus the fitted linear trend ``drift_daily * k``.

    With the default drift (the mean daily return) the result starts and ends
    at the same level, and its increments are exactly ``r_t - mean(r)``.
    """
    logp = np.asarray(logp, dtype=float)
    if drift_daily is None:
        drift_daily = log_returns(logp).mean()
    k = np.arange(len(logp))
    return logp - logp[0] - drift_daily * k


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


def estimate(logp, D=21, method="rs", clip=(H_MIN, H_MAX), **hurst_kwargs):
    """Full parameter estimate from a window of daily log prices."""
    if method not in HURST_METHODS:
        raise ValueError(f"method must be one of {sorted(HURST_METHODS)}")
    r = log_returns(logp)
    drift_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))
    if not std_daily > 0:
        raise ValueError("Zero variance window")
    residual = r - drift_daily  # increments of the detrended log price
    H_raw = HURST_METHODS[method](residual, **hurst_kwargs)
    H = float(np.clip(H_raw, *clip))
    a = D * drift_daily
    sigma = std_daily * D ** H
    return ParamEstimate(a=a, sigma=sigma, H=H, mu=a / sigma, H_raw=float(H_raw),
                         drift_daily=drift_daily, std_daily=std_daily,
                         n_returns=len(r), D=D, method=method)


def rolling_estimates(logp, decision_indices, window, D=21, method="rs", **kw):
    """Estimates on the W returns ending at each decision index (no look-ahead)."""
    logp = np.asarray(logp, dtype=float)
    rows = []
    for idx in decision_indices:
        if idx < window:
            raise ValueError(f"Decision index {idx} has fewer than {window} returns before it")
        est = estimate(logp[idx - window: idx + 1], D, method, **kw)
        rows.append({"decision_idx": int(idx), **est.as_row()})
    return pd.DataFrame(rows)


def simulate_log_prices(a, sigma, H, n_days, D=21, *, seed, n_paths=1, log_s0=0.0):
    """Paths of ln S on a daily grid, time in months: t_k = k / D.

    Returns an array of shape (n_paths, n_days + 1). Davies–Harte is used
    directly from the vendor package: only the marginal law matters here, not
    the causal coupling needed for the dual bound.
    """
    if n_days < 2 or n_paths < 1 or sigma <= 0 or not 0 < H < 1:
        raise ValueError("Need n_days >= 2, n_paths >= 1, sigma > 0, 0 < H < 1")
    np.random.seed(seed)
    with silenced():
        B, _, _ = FBM(n_days, n_paths, H, length=n_days / D, method="daviesharte").fbm()
    t = np.arange(n_days + 1) / D
    return log_s0 + a * t[None, :] + sigma * B.T


def bootstrap_estimates(est, window, *, n_boot=200, seed=0, method=None, **kw):
    """Parametric bootstrap: simulate the fitted model, re-estimate, one row per replicate."""
    method = est.method if method is None else method
    paths = simulate_log_prices(est.a, est.sigma, est.H, window, est.D,
                                seed=seed, n_paths=n_boot)
    rows = [estimate(path, est.D, method, **kw).as_row() for path in paths]
    return pd.DataFrame(rows)


def percentile_ci(samples, confidence=0.95):
    """Percentile interval of a 1-d sample as (low, high)."""
    alpha = 1 - confidence
    low, high = np.quantile(np.asarray(samples, dtype=float), [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
