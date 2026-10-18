from dataclasses import asdict, dataclass

import numpy as np

from osfbm import adapter
from osfbm.config import Config


@dataclass(frozen=True)
class NodeResult:
    H: float
    mu: float
    V: float
    V_raw: float
    SE: float
    p0: float
    p1: float
    E_tau: float
    stopped_at_zero: bool
    config_hash: str

    def as_row(self) -> dict:
        return asdict(self)


def value_at(mu: float, H: float, cfg: Config, paths: dict | None = None) -> NodeResult:
    """Оценка V(mu, H) и диагностики p0, p1, E[tau].

    Поправка на нулевой момент (B5). Даты остановки в коде авторов начинаются
    с t_1, а наш левый режим — это ровно tau*=0. Так как F_0 тривиальна, то
    continuation value в нуле равен значению задачи на t_1..t_N, а Z_0 = 0
    известен заранее. Поэтому

        V = max(0, V_raw)

    Для истинного ожидаемого продолжения это точное равенство. Здесь V_raw —
    тестовое среднее: выбор его положительной части может вносить смещение.
    Исправление выбора действия в нуле — отдельная задача. Текущий p0 равен
    0 или 1, но не подтверждает оптимальность немедленного выхода (docs/04.2).
    """
    if paths is None:
        paths = adapter.simulate_pair(H, cfg)

    X_tr = adapter.drifted(paths["B_train"], mu, cfg)
    X_te = adapter.drifted(paths["B_test"], mu, cfg)

    sig_tr = adapter.signatures(X_tr, cfg)
    sig_te = adapter.signatures(X_te, cfg)

    v_raw, sample_std, regr = adapter.longstaff_schwartz(sig_tr, X_tr, sig_te, X_te, cfg)
    se = sample_std / np.sqrt(X_te.shape[0])  # B7: vendor отдаёт СКО, не SE

    if v_raw <= 0.0:
        return NodeResult(
            H=H, mu=mu, V=0.0, V_raw=v_raw, SE=se,
            p0=1.0, p1=0.0, E_tau=0.0,
            stopped_at_zero=True, config_hash=cfg.hash,
        )

    idx = adapter.stopping_indices(regr, sig_te, X_te, cfg)
    tau = cfg.exercise_times[idx]
    return NodeResult(
        H=H, mu=mu, V=v_raw, V_raw=v_raw, SE=se,
        p0=0.0,
        p1=float(np.mean(idx == cfg.n_exercise - 1)),
        E_tau=float(np.mean(tau)),
        stopped_at_zero=False, config_hash=cfg.hash,
    )
