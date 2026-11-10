"""Validate saved regimes and frozen configuration before analysis."""

import numpy as np


def _classification(result, boundaries):
    low = np.asarray(boundaries.mu1_at(result.H.to_numpy()))
    high = np.asarray(boundaries.mu2_at(result.H.to_numpy()))
    labels = np.where(
        result.mu.to_numpy() <= low,
        "cash",
        np.where(result.mu.to_numpy() >= high, "hold", "policy"),
    )
    if not np.array_equal(labels, result.regime.to_numpy()):
        raise ValueError("Backtest regimes do not match the supplied boundaries")
    return low, high, labels


def validate_saved_results(result, selection):
    """Do not show results of another configuration under the frozen label."""
    c = selection.candidate
    for key, value in [
        ("window_H", c.H),
        ("window_a", c.drift),
        ("window_sigma", c.sigma),
        ("method", c.method),
        ("boundary_alpha", c.alpha),
    ]:
        if key not in result or not result[key].eq(value).all():
            raise ValueError(f"Saved results do not match frozen selection: {key}")
    _classification(result, selection.boundaries)
