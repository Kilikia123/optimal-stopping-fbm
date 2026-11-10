"""Daily volatility and fBm scaling; sigma depends on the estimated H."""


def estimate_daily_volatility(returns):
    """Sample standard deviation on the caller's volatility window."""
    std_daily = float(returns.std(ddof=1))
    if not std_daily > 0:
        raise ValueError("Zero variance window")
    return std_daily


def scale_volatility(std_daily, H, D):
    """Convert daily volatility to the fBm scale: sigma = std_daily * D**H."""
    return std_daily * D**H
