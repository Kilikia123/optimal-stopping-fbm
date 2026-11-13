"""Parametric bootstrap and percentile intervals for estimated parameters."""

import numpy as np
import pandas as pd

from estimation.pipeline import estimate
from estimation.simulation import simulate_log_prices


def bootstrap_estimates(est, window, *, n_boot=200, seed=0, method=None, **kw):
    """Parametric bootstrap: simulate the fitted model, re-estimate, one row per replicate."""
    method = est.method if method is None else method
    paths = simulate_log_prices(est.a, est.sigma, est.H, window, est.D, seed=seed, n_paths=n_boot)
    rows = [estimate(path, est.D, method, **kw).as_row() for path in paths]
    return pd.DataFrame(rows)


def percentile_ci(samples, confidence=0.95):
    """Percentile interval of a 1-d sample as (low, high)."""
    alpha = 1 - confidence
    low, high = np.quantile(np.asarray(samples, dtype=float), [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
