"""Reusable causal parameter estimates and batched policy decisions."""

import numpy as np

from estimation.hurst import HURST_METHODS
from market.backtest import normalised_path


class EstimatorCache:
    """Causal component estimates shared across candidates on one dataset."""

    def __init__(self, data, config, clip):
        self.data, self.config, self.clip = data, config, clip
        self.r = np.diff(data["logp"])
        self.cache = {}

    def _component(self, kind, window, method=None):
        """Calculate one component once for all decision dates in this market."""
        key = (kind, window, method)
        if key not in self.cache:
            values = []
            for start in self.data["starts"]:
                returns = self.r[start - window : start]
                if len(returns) != window:
                    raise ValueError("Insufficient common warmup")
                if kind == "H":
                    values.append(HURST_METHODS[method](returns - returns.mean()))
                else:
                    values.append(returns.mean() if kind == "a" else returns.std(ddof=1))
            self.cache[key] = np.asarray(values)
        return self.cache[key]

    def get(self, candidate):
        """Assemble causal H, a, sigma and mu for a candidate's three windows."""
        hraw = self._component("H", candidate.H, candidate.method)
        h = np.clip(hraw, *self.clip)
        a = self.config.D * self._component("a", candidate.drift)
        sigma = self._component("sigma", candidate.sigma) * self.config.D**h
        if not np.isfinite([h, a, sigma]).all() or np.any(sigma <= 0):
            raise ValueError("Invalid estimated parameters")
        return dict(H=h, H_raw=hraw, a=a, sigma=sigma, mu=a / sigma)


class DecisionCache:
    """Batched causal policy decisions, reused when only alpha changes."""

    def __init__(self, data, bank, D):
        self.data, self.bank, self.D = data, bank, D
        self.cache = {}

    def rewards(self, est, boundary):
        starts = self.data["starts"]
        cash = est["mu"] <= boundary.mu1_at(est["H"])
        hold = (~cash) & (est["mu"] >= boundary.mu2_at(est["H"]))
        tau = np.where(hold, self.D, 0)
        groups = {}
        keys = {}
        for i in np.flatnonzero(~cash & ~hold):
            node = self.bank.nearest_node(est["H"][i], est["mu"][i])
            key = (int(i), float(est["sigma"][i]), *node)
            keys[i] = key
            if key in self.cache:
                tau[i] = self.cache[key]
            else:
                groups.setdefault(node, []).append(i)
        for node, ids in groups.items():
            policy = self.bank.policy(*node)
            if policy.stopped_at_zero:
                days = np.zeros(len(ids), dtype=int)
            else:
                x = np.concatenate(
                    [
                        normalised_path(
                            self.data["logp"][starts[i] : starts[i] + self.D + 1],
                            est["sigma"][i],
                            self.bank.cfg,
                            self.D,
                        )
                        for i in ids
                    ]
                )
                decisions = policy.decide(x, self.bank.cfg)
                days = np.rint(
                    self.bank.cfg.exercise_times[decisions] / self.bank.cfg.T * self.D
                ).astype(int)
            for i, day in zip(ids, days):
                self.cache[keys[i]] = int(day)
                tau[i] = day
        logp = self.data["logp"]
        bh = logp[starts + self.D] - logp[starts]
        rewards = logp[starts + tau] - logp[starts]
        return dict(
            model=rewards,
            buy_hold=bh,
            regimes_only=np.where(cash, 0, bh),
            sign_drift=np.where(est["a"] > 0, bh, 0),
            cash=cash,
            hold=hold,
            tau=tau,
        )
