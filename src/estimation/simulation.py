"""Synthetic daily log prices under the fitted fractional Brownian model."""

import numpy as np

from adapter import silenced
from vendor.FBM_package import FBM


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
