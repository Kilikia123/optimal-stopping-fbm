import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    """Конфигурация одного эксперимента.

    Две шкалы времени различаются намеренно и путать их нельзя:
    ``n_exercise`` — моменты, когда разрешено продать (N=20 из docs/01.1);
    ``n_fine`` — сетка, по которой приближаются итерированные интегралы
    сигнатуры. В notebook авторов это N1=10 и N=100.
    """

    T: float = 1.0
    n_exercise: int = 20
    n_fine: int = 200

    M_train: int = 20_000
    M_test: int = 20_000

    K: int = 4
    ridge: float = 1e-9
    method: str = "cholesky"

    # Поле старых конфигураций: сохранено для совместимости их хэшей.
    # Практические границы используют отдельный epsilon в единицах награды.
    tol_se: float = 3.0

    seed: int = 20260904

    mu_grid: tuple[float, ...] = field(
        # Стартовый диапазон скана. Локализация практических границ зависит
        # от epsilon; при необходимости сетку расширяют отдельным запуском.
        default_factory=lambda: tuple(np.round(np.arange(-6.0, 12.01, 0.5), 6))
    )
    H_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.arange(0.1, 0.901, 0.05), 6))
    )

    def __post_init__(self) -> None:
        if self.n_fine % self.n_exercise != 0:
            raise ValueError(
                f"n_fine={self.n_fine} должно делиться на n_exercise={self.n_exercise}: "
                "этого требует построение subindex в коде авторов"
            )
        if self.method != "cholesky":
            raise ValueError(
                f"method={self.method!r}: допустим только 'cholesky'. У daviesharte "
                "преобразование шума непричинное, dW перестаёт порождать fBm и dual "
                "становится невалидным (см. adapter.simulate_paths)"
            )
        if not 0.0 < self.T:
            raise ValueError("T должно быть положительным")

    @property
    def times(self) -> np.ndarray:
        """Мелкая сетка t_0..t_n_fine."""
        return np.linspace(0.0, self.T, self.n_fine + 1)

    @property
    def exercise_index(self) -> np.ndarray:
        """Индексы дат решения внутри мелкой сетки — как их строит vendor.

        Внимание: начинается с t_1, нулевой момент исключён. Поправка на t_0
        делается в core.value_at (см. docs/02.2 и adapter.B5).
        """
        n1 = self.n_exercise
        return np.array([int((j + 1) * self.n_fine / n1) for j in range(n1)])

    @property
    def exercise_times(self) -> np.ndarray:
        return self.times[self.exercise_index]

    @property
    def sig_dim(self) -> int:
        """Размерность куба сигнатур: 2^(K+1)-1 для двумерного пути (t, X)."""
        return 2 ** (self.K + 1) - 1

    def replace(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mu_grid"] = list(self.mu_grid)
        d["H_grid"] = list(self.H_grid)
        return d

    @property
    def hash(self) -> str:
        """Хэш конфига без сеток — идентифицирует режим расчёта одного узла."""
        d = {k: v for k, v in self.to_dict().items() if k not in ("mu_grid", "H_grid")}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key in ("mu_grid", "H_grid"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(float(x) for x in raw[key])
        return cls(**raw)
