"""Causal parameter estimation: drift, H, volatility and normalised drift mu."""

import numpy as np
import pandas as pd

from estimation.drift import estimate_daily_drift, scale_drift
from estimation.hurst import H_MAX, H_MIN, HURST_METHODS
from estimation.preprocessing import log_returns
from estimation.types import EstimationWindows, ParamEstimate
from estimation.volatility import estimate_daily_volatility, scale_volatility


def estimate(
    logp, D=21, method="rs", clip=(H_MIN, H_MAX), *, windows=None, **hurst_kwargs
) -> ParamEstimate:
    """Estimate parameters from past daily log prices, using separate windows.

    Drift and volatility use their declared return windows. H is estimated on
    centred increments and clipped to the supported policy range. Daily drift
    and volatility are then scaled to D trading days; mu = a / sigma.
    No observation beyond the end of ``logp`` is used.
    """
    if method not in HURST_METHODS:
        raise ValueError(f"method must be one of {sorted(HURST_METHODS)}")
    r = log_returns(logp)
    if windows is None:
        windows = EstimationWindows(len(r), len(r), len(r))
    if len(r) < windows.warmup:
        raise ValueError("Insufficient history for estimation windows")
    # 1. Estimate daily drift and dispersion on their own historical windows.
    drift_daily = estimate_daily_drift(r[-windows.drift :])
    std_daily = estimate_daily_volatility(r[-windows.sigma :])
    # 2. Estimate memory from centred increments on the H window.
    hr = r[-windows.H :]
    residual = hr - hr.mean()
    H_raw = HURST_METHODS[method](residual, **hurst_kwargs)
    H = float(np.clip(H_raw, *clip))
    # 3. Convert to model-month parameters. The volatility scale depends on H.
    a = scale_drift(drift_daily, D)
    sigma = scale_volatility(std_daily, H, D)
    return ParamEstimate(
        a=a,
        sigma=sigma,
        H=H,
        mu=a / sigma,
        H_raw=float(H_raw),
        drift_daily=drift_daily,
        std_daily=std_daily,
        n_returns=windows.warmup,
        D=D,
        method=method,
        window_H=windows.H,
        window_a=windows.drift,
        window_sigma=windows.sigma,
    )


def rolling_estimates(logp, decision_indices, window, D=21, method="rs", *, windows=None, **kw):
    """Estimates on the W returns ending at each decision index (no look-ahead)."""
    logp = np.asarray(logp, dtype=float)
    rows = []
    window = windows.warmup if windows is not None else window
    for idx in decision_indices:
        if idx < window:
            raise ValueError(f"Decision index {idx} has fewer than {window} returns before it")
        est = estimate(logp[idx - window : idx + 1], D, method, windows=windows, **kw)
        rows.append({"decision_idx": int(idx), **est.as_row()})
    return pd.DataFrame(rows)
