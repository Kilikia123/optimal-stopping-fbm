import json
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from osfbm import adapter
from osfbm.config import Config
from osfbm.core import NodeResult, value_at

RESULTS_DIR = Path(__file__).resolve().parents[2] / "result_optimal_stopping"
NODES_CSV = RESULTS_DIR / "nodes.csv"
GRID_CSV = RESULTS_DIR / "grid.csv"


def append(path: Path, rows: list[dict], key: list[str]) -> None:
    """Дозапись с перезаписью строк по ключу: повторный прогон той же точки
    обновляет строку, а не плодит дубликаты."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, float_precision="round_trip")
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=key, keep="last").sort_values(key)
    # Сбой во время записи не должен уничтожать уже сохранённые узлы.
    with NamedTemporaryFile(dir=path.parent, suffix=".csv.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        new.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_run(run_dir: str | Path, cfg: Config) -> Path:
    """Создать запуск или проверить полный конфиг перед продолжением."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    config = cfg.to_dict()
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Конфигурация запуска отличается, включая сетки. Выберите новый run_id")
    else:
        if (run_dir / "nodes.csv").exists():
            raise ValueError("nodes.csv без config.json: нельзя безопасно продолжить запуск")
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir


def load_run(run_dir: str | Path) -> tuple[Config, pd.DataFrame]:
    """Загрузить конфиг и сохранённые узлы без симуляции или обучения."""
    run_dir = Path(run_dir)
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    for key in ("mu_grid", "H_grid"):
        raw[key] = tuple(raw[key])
    cfg = Config(**raw)
    columns = [f.name for f in fields(NodeResult)]
    nodes_path = run_dir / "nodes.csv"
    if not nodes_path.exists():
        return cfg, pd.DataFrame(columns=columns)
    nodes = pd.read_csv(nodes_path, float_precision="round_trip")
    if set(nodes.columns) != set(columns):
        raise ValueError("Неожиданные столбцы nodes.csv")
    if nodes.duplicated(["H", "mu"]).any() or not nodes["config_hash"].eq(cfg.hash).all():
        raise ValueError("Дубликаты узлов или несовместимый config_hash")
    if not nodes["H"].isin(cfg.H_grid).all() or not nodes["mu"].isin(cfg.mu_grid).all():
        raise ValueError("Сохранённые узлы не соответствуют сеткам config.json")
    return cfg, nodes.sort_values(["H", "mu"]).reset_index(drop=True)


def run_grid(cfg: Config, run_dir: str | Path, progress=None) -> pd.DataFrame:
    """Рассчитать недостающие узлы; checkpoint после каждого узла.

    progress получает NodeResult только для вновь рассчитанных узлов.
    Пути генерируются один раз на H; при возобновлении seed сохраняет их.
    """
    run_dir = prepare_run(run_dir, cfg)
    _, saved = load_run(run_dir)
    done = set(zip(saved["H"], saved["mu"]))
    for H in cfg.H_grid:
        missing = [mu for mu in cfg.mu_grid if (H, mu) not in done]
        if not missing:
            continue
        paths = adapter.simulate_pair(float(H), cfg)
        for mu in missing:
            result = value_at(float(mu), float(H), cfg, paths=paths)
            append(run_dir / "nodes.csv", [result.as_row()], key=["config_hash", "H", "mu"])
            done.add((H, mu))
            if progress is not None:
                progress(result)
    return load_run(run_dir)[1]
