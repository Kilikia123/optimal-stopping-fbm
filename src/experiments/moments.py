"""Online reward moments and pointwise / familywise Student intervals."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t


@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Rewards must be a finite vector")
        if not values.size:
            return
        count = len(values)
        average = float(values.mean())
        delta = average - self.mean
        total = self.n + count
        self.m2 += float(np.sum((values - average) ** 2)) + delta**2 * self.n * count / total
        self.mean += delta * count / total
        self.n = total

    def intervals(self, confidence, family_size):
        if self.n < 2 or family_size < 1:
            raise ValueError("At least two rewards and one family member are required")
        se = np.sqrt(max(0.0, self.m2) / (self.n - 1) / self.n)
        alpha = 1 - confidence
        point = float(t.isf(alpha / 2, self.n - 1) * se)
        band = float(t.isf(alpha / (2 * family_size), self.n - 1) * se)
        return {
            "n": self.n,
            "V": self.mean,
            "SE": se,
            "ci_low": self.mean - point,
            "ci_high": self.mean + point,
            "band_low": self.mean - band,
            "band_high": self.mean + band,
        }
