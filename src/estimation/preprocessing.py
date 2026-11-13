"""Daily log returns and removal of a fitted linear trend."""

import numpy as np


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
