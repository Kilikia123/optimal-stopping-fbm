from dataclasses import asdict, dataclass

import numpy as np

from osfbm import adapter
from osfbm.config import Config
from osfbm.core import NodeResult, value_at


@dataclass(frozen=True)
class BoundaryResult:
    H: float
    mu1_epsilon: float | None
    mu2_epsilon: float | None
    boundary_difference: float | None
    mu1_bracket: tuple[float, float] | None
    mu2_bracket: tuple[float, float] | None
    epsilon: float
    mu1_SE: float | None
    mu2_SE: float | None
    note: str
    config_hash: str

    def as_row(self) -> dict:
        d = asdict(self)
        for k in ("mu1_bracket", "mu2_bracket"):
            d[k] = "" if d[k] is None else f"[{d[k][0]!r}, {d[k][1]!r}]"
        return d


def scan(H: float, cfg: Config, progress=None) -> list[NodeResult]:
    """Скан по сетке mu при фиксированном H.

    Траектории генерируются один раз и переиспользуются во всех узлах:
    B^H от mu не зависит. Общие случайные числа снижают шум сравнений,
    но не гарантируют гладкость или выпуклость оценённой кривой.
    """
    paths = adapter.simulate_pair(H, cfg)
    out = []
    for mu in cfg.mu_grid:
        res = value_at(float(mu), H, cfg, paths=paths)
        out.append(res)
        if progress is not None:
            progress(res)
    return out


def boundaries_from_scan(results: list[NodeResult], cfg: Config, *, epsilon: float) -> BoundaryResult:
    """Практические границы на всей оси mu с фиксированным допуском награды.

    mu1_epsilon — последний узел V <= epsilon перед выходом из области;
    mu2_epsilon — первый узел V - mu*T <= epsilon после входа в область.
    Брекеты описывают разрешение сетки, а не доверительный интервал.
    SE не участвует в определении epsilon. Используется оценка обученной
    стратегии: без dual результат не подтверждает близость истинного оптимума.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon должен быть конечным положительным числом")
    if not results:
        raise ValueError("Нужен непустой скан одного H")
    if len({r.H for r in results}) != 1 or any(r.config_hash != cfg.hash for r in results):
        raise ValueError("Узлы должны иметь один H и соответствовать cfg")
    mus = np.array([r.mu for r in results], dtype=float)
    order = np.argsort(mus)
    mus = mus[order]
    vals = np.array([results[i].V for i in order])
    ses = np.array([results[i].SE for i in order])
    if not np.all(np.isfinite([mus, vals, ses])) or np.any(ses < 0):
        raise ValueError("mu, V и SE должны быть конечными, SE >= 0")
    if np.any(np.diff(mus) <= 0):
        raise ValueError("Значения mu должны быть уникальными")

    left = vals <= epsilon
    right = (vals - mus * cfg.T) <= epsilon

    notes = []
    mu1 = mu1_br = mu1_se = None
    if np.any(np.diff(left.astype(int)) > 0):
        notes.append("mu1_epsilon не локализована: повторный вход в область V <= epsilon")
    elif left.any():
        i = int(np.flatnonzero(left)[-1])
        if i == len(mus) - 1:
            notes.append("mu1_epsilon не локализована: расширить сетку вправо")
        else:
            mu1 = float(mus[i])
            mu1_se = float(ses[i])
            mu1_br = (float(mus[i]), float(mus[i + 1]))
    else:
        notes.append("mu1_epsilon не локализована: расширить сетку влево")

    mu2 = mu2_br = mu2_se = None
    if np.any(np.diff(right.astype(int)) < 0):
        notes.append("mu2_epsilon не локализована: повторный выход из области V - mu*T <= epsilon")
    elif right.any():
        j = int(np.flatnonzero(right)[0])
        if j == 0:
            notes.append("mu2_epsilon не локализована: расширить сетку влево")
        else:
            mu2 = float(mus[j])
            mu2_se = float(ses[j])
            mu2_br = (float(mus[j - 1]), float(mus[j]))
    else:
        notes.append("mu2_epsilon не локализована: расширить сетку вправо")

    delta = None if (mu1 is None or mu2 is None) else mu2 - mu1
    if delta is not None and delta < 0:
        notes.append("Отрицательная разность: перекрытие областей близости к двум опорным стратегиям")

    return BoundaryResult(
        H=float(results[0].H), mu1_epsilon=mu1, mu2_epsilon=mu2, boundary_difference=delta,
        mu1_bracket=mu1_br, mu2_bracket=mu2_br,
        epsilon=float(epsilon), mu1_SE=mu1_se, mu2_SE=mu2_se,
        note="; ".join(notes), config_hash=cfg.hash,
    )
