"""Parameter estimates and independent windows for H, drift and volatility."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EstimationWindows:
    H: int
    drift: int
    sigma: int

    def __post_init__(self):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 2
            for v in (self.H, self.drift, self.sigma)
        ):
            raise ValueError("Estimation windows must be integers >= 2")

    @property
    def warmup(self):
        return max(self.H, self.drift, self.sigma)


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
    window_H: int | None = None
    window_a: int | None = None
    window_sigma: int | None = None

    def as_row(self):
        return asdict(self)
