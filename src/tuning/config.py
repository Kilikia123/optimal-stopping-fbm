"""Declared search grids, candidate identifiers and independent evaluation sizes."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from estimation.hurst import H_MAX, H_MIN, HURST_METHODS
from estimation.types import EstimationWindows
from infrastructure.identity import digest


@dataclass(frozen=True)
class TuningConfig:
    H_windows: tuple = (252, 504, 1008)
    drift_windows: tuple = (21, 63, 126, 252)
    sigma_windows: tuple = (63, 126, 252)
    methods: tuple = ("rs", "dfa")
    alphas: tuple = (0.05, 0.1, 0.2, 0.3, 0.5)
    true_H: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)
    stationary_mu: tuple = (-0.5, 0.0, 0.5)
    profiles: tuple = ("drift12", "drift36", "volatility", "joint")
    D: int = 21
    seed: int = 20261108
    confidence: float = 0.95
    coarse_replicates: int = 4
    validation_replicates: int = 12
    test_replicates: int = 24
    coarse_months: int = 60
    validation_months: int = 120
    test_months: int = 120
    top_k: int = 10

    def __post_init__(self):
        for name in ("H_windows", "drift_windows", "sigma_windows"):
            seq = getattr(self, name)
            if not seq or len(set(seq)) != len(seq):
                raise ValueError(f"{name} must be nonempty and unique")
            for w in seq:
                EstimationWindows(w, w, w)
                if name == "H_windows" and w < 32:
                    raise ValueError("H estimation requires windows >= 32 days")
        for name in (
            "coarse_replicates",
            "validation_replicates",
            "test_replicates",
            "coarse_months",
            "validation_months",
            "test_months",
            "top_k",
            "D",
        ):
            v = getattr(self, name)
            if (
                not isinstance(v, int)
                or isinstance(v, bool)
                or v < (2 if "replicates" in name else 1)
            ):
                raise ValueError(f"Invalid {name}")
        if (
            not self.methods
            or not set(self.methods) <= set(HURST_METHODS)
            or len(set(self.methods)) != len(self.methods)
            or not self.alphas
            or len(set(self.alphas)) != len(self.alphas)
            or not all(0 < a < 1 for a in self.alphas)
            or not 0 < self.confidence < 1
            or not self.true_H
            or not all(H_MIN <= h <= H_MAX for h in self.true_H)
            or len(set(self.true_H)) != len(self.true_H)
            or not self.stationary_mu
            or not np.isfinite(self.stationary_mu).all()
            or len(set(self.stationary_mu)) != len(self.stationary_mu)
            or not self.profiles
            or len(set(self.profiles)) != len(self.profiles)
            or not set(self.profiles) <= {"drift12", "drift36", "volatility", "joint"}
        ):
            raise ValueError("Invalid methods, alphas, confidence or scenarios")

    @property
    def warmup(self):
        maximum = max((*self.H_windows, *self.drift_windows, *self.sigma_windows))
        return int(np.ceil(maximum / self.D)) * self.D


@dataclass(frozen=True)
class Candidate:
    H: int
    drift: int
    sigma: int
    method: str
    alpha: float

    @property
    def key(self):
        return digest(asdict(self))[:20]

    @property
    def windows(self):
        return EstimationWindows(self.H, self.drift, self.sigma)


def candidates(config):
    return sorted(
        (
            Candidate(*v)
            for v in product(
                config.H_windows,
                config.drift_windows,
                config.sigma_windows,
                config.methods,
                config.alphas,
            )
        ),
        key=lambda c: (c.H, c.drift, c.sigma, c.method, c.alpha),
    )
