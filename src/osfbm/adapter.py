import contextlib
import io
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from osfbm.vendor.FBM_package import FBM
    from osfbm.vendor.Linear_signature_optimal_stopping import LinearLongstaffSchwartzPricer
    from osfbm.vendor.Signature_computer import SignatureComputer

from osfbm.config import Config


@contextlib.contextmanager
def silenced():
    """B8. Гасит print из vendor.

    В коде авторов шесть print: отладочный ``print(z.shape)`` в _daviesharte,
    сообщение о лифте на каждый вызов сигнатур, score регрессии на каждом шаге
    обратной рекурсии и таймеры LP. На сотнях узлов сетки это тысячи строк.
    """
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        # Ridge жалуется на плохую обусловленность: куб сигнатур содержит
        # константный столбец, а Ridge по умолчанию ещё и подгоняет intercept,
        # то есть колонки строго коллинеарны. Это свойство конструкции авторов,
        # оно воспроизводит их опубликованное число; лечится увеличением ridge.
        warnings.filterwarnings("ignore", category=Warning, module="scipy")
        warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
        yield


def simulate_paths(H: float, cfg: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Траектории fBm и приращения порождающего броуновского движения.

    B3. Метод только ``cholesky``: он умножает шум на нижнетреугольную матрицу,
    то есть преобразование причинное, и фильтрации B^H и W совпадают. У
    daviesharte внутри досоздаётся собственный шум gn2 и всё смешивается через
    FFT непричинно — возвращаемое dW перестаёт порождать fBm, сигнатура не
    адаптирована к фильтрации W, и dual становится невалидным.

    B4. fbm() отдаёт массивы в форме (n+1, M) и (n, M) — транспонируем.

    Воспроизводимость: генератор авторов использует глобальный np.random,
    поэтому seed задаётся только так, а не через default_rng.

    Возвращает B формы (M, n_fine+1) и dW формы (M, n_fine).
    """
    np.random.seed(seed)
    with silenced():
        B, _dB, dW = FBM(
            cfg.n_fine, cfg.M_train, H, length=cfg.T, method=cfg.method
        ).fbm()
    return B.T, dW.T


def simulate_pair(H: float, cfg: Config) -> dict[str, np.ndarray]:
    """Обучающие и тестовые траектории для одного H.

    B^H не зависит от mu — mu входит только в X = mu*t + B^H. Поэтому
    траектории генерируются один раз на H и переиспользуются на всём скане по
    mu: это common random numbers из docs/04.3, здесь они бесплатны и делают
    кривую V(mu) гладкой.
    """
    base = abs(hash((cfg.seed, round(float(H), 6)))) % (2**31 - 1000)
    B_tr, dW_tr = simulate_paths(H, cfg, seed=base)
    B_te, dW_te = simulate_paths(H, cfg.replace(M_train=cfg.M_test), seed=base + 1)
    return {"B_train": B_tr, "dW_train": dW_tr, "B_test": B_te, "dW_test": dW_te}


def drifted(B: np.ndarray, mu: float, cfg: Config) -> np.ndarray:
    """B1. Единственная содержательная правка постановки (docs/04.1).

    X_t = mu * t + B^H_t. Дальше X идёт и в сигнатуры, и в payoff.
    """
    return mu * cfg.times[None, :] + B


def signatures(X: np.ndarray, cfg: Config) -> np.ndarray:
    """B2. Сигнатуры расширенного пути (t, X).

    compute_signature требует 7 позиционных аргументов (X, vol, A, Payoff,
    dW, I, MM), а в notebook авторов вызывается с четырьмя — то есть notebook
    в текущей версии репозитория падает с TypeError. Лифт "normal" использует
    только dX и A, но остальные слоты разыменовываются в начале метода,
    поэтому передаём X во все неиспользуемые.

    Возвращает куб (M, n_fine+1, 2^(K+1)-1).
    """
    m = X.shape[0]
    A = np.zeros((m, cfg.n_fine + 1))
    A[:, 1:] = cfg.times[1:]
    sc = SignatureComputer(
        cfg.T, cfg.n_fine, cfg.K, "linear", signature_lift="normal", poly_degree=0
    )
    with silenced():
        return sc.compute_signature(X, X, A, X, X, X, X)


def longstaff_schwartz(
    sig_train: np.ndarray,
    X_train: np.ndarray,
    sig_test: np.ndarray,
    X_test: np.ndarray,
    cfg: Config,
) -> tuple[float, float, list]:
    """Signature Longstaff-Schwartz авторов, payoff = сам путь X.

    r=0 и mode="Standard" уже есть в их коде, менять алгоритм не требуется.
    Возвращает (V_raw, sample_std, regr) — V_raw ещё БЕЗ поправки на t_0.
    """
    pricer = LinearLongstaffSchwartzPricer(
        N1=cfg.n_exercise, T=cfg.T, r=0.0, mode="Standard", ridge=cfg.ridge
    )
    with silenced():
        v_raw, sample_std, regr = pricer.price(sig_train, X_train, sig_test, X_test)
    return float(v_raw), float(sample_std), regr


def stopping_indices(regr: list, sig_test: np.ndarray, X_test: np.ndarray, cfg: Config) -> np.ndarray:
    """B6. Моменты остановки, которых price не возвращает.

    Повторяет правило остановки авторов, но векторно по траекториям: первый
    индекс, где предсказанный continuation value не превышает payoff; на
    последней дате остановка принудительная.

    Возвращает индексы в шкале дат решения, 0..n_exercise-1. Соответствующее
    время — cfg.exercise_times[idx] = (idx+1)*T/n_exercise: subindex авторов
    сдвинут на единицу, поэтому нулевой даты здесь нет.
    """
    n1 = cfg.n_exercise
    payoff_ex = X_test[:, cfg.exercise_index]

    cont = np.empty((X_test.shape[0], n1))
    for j in range(n1 - 1):
        cont[:, j] = regr[j].predict(sig_test[:, cfg.exercise_index[j], :])
    cont[:, n1 - 1] = -np.inf  # на последней дате останавливаемся всегда

    stop = cont <= payoff_ex
    stop[:, n1 - 1] = True
    return stop.argmax(axis=1)
