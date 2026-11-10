"""Saved stopping rules: exercise decisions, serialisation and legacy reconstruction."""

from dataclasses import dataclass

import numpy as np

import adapter


@dataclass
class FixedPolicy:
    coefficients: np.ndarray
    intercepts: np.ndarray
    stopped_at_zero: bool

    def decide(self, X, cfg):
        """Exercise-date index 0..n_exercise-1 chosen by the rule on each path.

        X is the observed normalised path mu*t + B^H_t on the fine grid, shape
        (M, n_fine+1). The rule stops at the first date where the fitted
        continuation value does not exceed the payoff; the last date is forced.
        The t=0 choice (``stopped_at_zero``) is not applied here.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != cfg.n_fine + 1:
            raise ValueError(f"X must have shape (M, {cfg.n_fine + 1})")
        if self.coefficients.shape != (cfg.n_exercise - 1, cfg.sig_dim):
            raise ValueError("Policy shape does not match the configuration")
        sig = adapter.signatures(X, cfg)
        index = np.full(len(X), cfg.n_exercise - 1, dtype=int)
        active = np.ones(len(X), dtype=bool)
        for j in range(cfg.n_exercise - 1):
            k = cfg.exercise_index[j]
            continuation = sig[:, k, :] @ self.coefficients[j] + self.intercepts[j]
            stop = active & (continuation <= X[:, k])
            index[stop] = j
            active[stop] = False
        return index

    def rewards(self, B, mu, cfg):
        if self.stopped_at_zero:
            return np.zeros(len(B))
        X = adapter.drifted(B, mu, cfg)
        index = self.decide(X, cfg)
        return X[np.arange(len(B)), cfg.exercise_index[index]]

    def to_dict(self):
        return {
            "coefficients": self.coefficients.tolist(),
            "intercepts": self.intercepts.tolist(),
            "stopped_at_zero": self.stopped_at_zero,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            np.asarray(d["coefficients"]), np.asarray(d["intercepts"]), bool(d["stopped_at_zero"])
        )


def restore_policy(row, cfg, paths):
    """Reproduce the legacy fit; freeze its old t=0 choice before a fresh test."""
    mu = row["mu"]
    train = adapter.drifted(paths["B_train"], mu, cfg)
    test = adapter.drifted(paths["B_test"], mu, cfg)
    sig_train, sig_test = adapter.signatures(train, cfg), adapter.signatures(test, cfg)
    # Vendor may mutate the terminal payoff column; do not mutate source paths.
    raw, _, regressions = adapter.longstaff_schwartz(sig_train, train, sig_test, test, cfg)
    if not np.isclose(raw, row["V_raw"], rtol=1e-7, atol=1e-9):
        raise ValueError(
            f"Legacy policy mismatch H={row['H']} mu={mu}: "
            f"saved={row['V_raw']}, reproduced={raw}"
        )
    policy = FixedPolicy(
        np.asarray([r.coef_ for r in regressions]),
        np.asarray([r.intercept_ for r in regressions]),
        bool(row["stopped_at_zero"]),
    )
    # Check our evaluator as well, before storing the artifact.
    replay = policy.rewards(paths["B_test"], mu, cfg).mean()
    expected = 0.0 if policy.stopped_at_zero else raw
    if not np.isclose(replay, expected, rtol=1e-7, atol=1e-9):
        raise ValueError("Serialized policy does not reproduce the original stopping rule")
    return policy
