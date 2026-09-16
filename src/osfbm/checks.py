from dataclasses import dataclass

import numpy as np

from osfbm import adapter
from osfbm.boundaries import boundaries_from_scan, scan
from osfbm.config import Config
from osfbm.core import value_at

# Эталон из notebook авторов (H=0.3, mu=0, N=100, N1=10, K=4, M=M2=50000):
# "Linear Longstaff-Schwartz lower bound: 0.2436758369259125 ± 0.0035850490987833178"
REFERENCE_MU0 = {"H": 0.3, "V": 0.2436758369259125, "SE": 0.0035850490987833178}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def fbm_covariance(cfg: Config, H: float = 0.3, M: int = 20_000) -> Check:
    """Эмпирическая ковариация генератора против теоретической."""
    small = cfg.replace(n_fine=20, n_exercise=10, M_train=M)
    B, _ = adapter.simulate_paths(H, small, seed=cfg.seed)
    t = small.times
    emp = (B.T @ B) / M
    th = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    err = np.abs(emp - th).max()
    # 3 MC-ошибки для ковариации гауссовских величин масштаба ~T^{2H}
    tol = 4.0 * np.sqrt(2.0 / M) * max(th.max(), 1e-12)
    return Check("ковариация fBm", err <= tol,
                 f"max|эмп - теор| = {err:.2e}, допуск {tol:.2e}")


def reference_mu0(cfg: Config) -> Check:
    """Сверка с опубликованным числом Section 4.1 при mu=0."""
    ref = cfg.replace(n_exercise=10, n_fine=100, M_train=50_000, M_test=50_000, K=4)
    r = value_at(0.0, REFERENCE_MU0["H"], ref)
    comb = np.hypot(r.SE, REFERENCE_MU0["SE"])
    z = abs(r.V - REFERENCE_MU0["V"]) / comb
    return Check("mu=0 vs Section 4.1", z <= 3.0,
                 f"наш {r.V:.6f} ± {r.SE:.6f} vs эталон {REFERENCE_MU0['V']:.6f}; "
                 f"расхождение {z:.2f} SE")


def half_hurst(cfg: Config, *, epsilon: float = 0.01) -> Check:
    """H=1/2: V(mu) = max(0, mu*T); практические границы равны ±epsilon/T.

    Критерий PASS относится к стоимости. Границы сообщаются диагностически:
    шум оценки и разрешение сетки могут не позволить их локализовать.
    """
    res = scan(0.5, cfg)
    b = boundaries_from_scan(res, cfg, epsilon=epsilon)
    worst = max(abs(r.V - max(0.0, r.mu * cfg.T)) / max(r.SE, 1e-12) for r in res)
    return Check("H=1/2", bool(worst <= 5.0),
                 f"mu1_epsilon={b.mu1_epsilon}, mu2_epsilon={b.mu2_epsilon}; "
                 f"теоретически {epsilon / cfg.T:g}, {-epsilon / cfg.T:g}; "
                 f"max|V - max(0,mu*T)| = {worst:.1f} SE; {b.note}")


def invariants(cfg: Config, H: float = 0.3) -> Check:
    """V >= max(0, mu*T) и выпуклость V по mu (docs/04.5, п.3)."""
    res = sorted(scan(H, cfg), key=lambda r: r.mu)
    mus = np.array([r.mu for r in res]); vals = np.array([r.V for r in res])
    floor = np.maximum(0.0, mus * cfg.T)
    se = np.array([r.SE for r in res])
    viol = float(np.max((floor - vals) / np.maximum(se, 1e-12)))

    # выпуклость: вторая разность на равномерной сетке
    d2 = vals[2:] - 2 * vals[1:-1] + vals[:-2]
    conv = float(np.min(d2 / np.maximum(se[1:-1], 1e-12)))
    ok = viol <= 3.0 and conv >= -5.0
    return Check("инварианты V", ok,
                 f"max нарушения V>=max(0,mu*T): {viol:.1f} SE; "
                 f"худшая вторая разность: {conv:.1f} SE")


def run_all(cfg: Config) -> list[Check]:
    fast = cfg.replace(M_train=min(cfg.M_train, 20_000), M_test=min(cfg.M_test, 20_000))
    return [fbm_covariance(cfg), reference_mu0(cfg), half_hurst(fast), invariants(fast)]
