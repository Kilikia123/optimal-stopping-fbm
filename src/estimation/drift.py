"""Drift estimates: daily mean return and conversion to a D-day model month."""


def estimate_daily_drift(returns):
    """Mean daily log return on the caller's drift window."""
    return float(returns.mean())


def scale_drift(drift_daily, D):
    """Convert daily drift to a model month of D trading days."""
    return D * drift_daily
