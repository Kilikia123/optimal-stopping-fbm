"""Invert simultaneous reward bands into grid-localisation intervals."""

import numpy as np


def invert_band(mus, means, low, high, epsilon, side):
    """Confidence set for a window-restricted GRID extremum, without interpolation.

    On the simultaneous coverage event, definitely-in is a subset of the true
    threshold set, which is a subset of possibly-in. Max/min preserve inclusion
    bounds. We report no bounded interval without certified endpoints.
    """
    mus, means, low, high = map(lambda x: np.asarray(x, dtype=float), (mus, means, low, high))
    if (
        side not in (1, 2)
        or len(mus) < 2
        or any(a.shape != mus.shape for a in (means, low, high))
        or not np.isfinite(np.concatenate((mus, means, low, high))).all()
        or np.any(np.diff(mus) <= 0)
        or np.any(low > high)
    ):
        raise ValueError("Invalid ordered grid/band")
    inside = means <= epsilon
    definite = high <= epsilon
    possible = low <= epsilon
    result = {"estimate": None, "mu_low": None, "mu_high": None, "status": "expand_window"}
    changes = np.diff(inside.astype(int))
    if np.any(changes > 0 if side == 1 else changes < 0):
        result["status"] = "multiple_transitions"
        return result
    if inside.any():
        result["estimate"] = float(mus[np.flatnonzero(inside)[-1 if side == 1 else 0]])
    enclosed = (
        (definite[0] and not possible[-1]) if side == 1 else (not possible[0] and definite[-1])
    )
    if not enclosed:
        return result
    if side == 1:
        lower, upper = mus[definite].max(), mus[possible].max()
    else:
        lower, upper = mus[possible].min(), mus[definite].min()
    result.update(mu_low=float(lower), mu_high=float(upper), status="localized_grid")
    return result
