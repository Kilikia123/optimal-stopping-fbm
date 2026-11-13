"""Text progress and read-only audits; presentation does not change tuning identity."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from infrastructure.storage import atomic_json
from tuning.config import TuningConfig
from tuning.scenarios import scenarios


class TextTuningProgress:
    """Always emit plain text as well as a non-widget progress bar."""

    def __init__(self, output):
        self.output = Path(output)
        self.cached = {
            stage: sum(1 for _ in (self.output / "scores" / stage).glob("*/*.json"))
            for stage in ("coarse", "validation", "test")
        }
        self.bar = None
        self.stages = {}

    def __enter__(self):
        self.started = time.perf_counter()
        print(
            "Подготовка: проверка банка политик, чтение парных моментов и построение границ.",
            flush=True,
        )
        print(
            "Индикатор coarse появится после подготовки. Уже сохранённые оценки будут прочитаны из кэша.",
            flush=True,
        )
        return self

    def __call__(self, event):
        stage = event["stage"]
        if stage not in self.stages:
            self._finish_stage()
            cached = min(self.cached[stage], event["total"])
            self.stages[stage] = dict(
                total=event["total"],
                cached_at_start=cached,
                done=0,
                started=time.perf_counter(),
                last_milestone=-1,
            )
            print(
                f"{stage}: всего {event['total']:,} оценок; сохранено ранее {cached:,}; новых {event['total']-cached:,}.",
                flush=True,
            )
            self.bar = tqdm(
                total=event["total"],
                desc=stage,
                unit="оценка",
                file=sys.stdout,
                mininterval=1.0,
                dynamic_ncols=False,
                ncols=95,
            )
        info = self.stages[stage]
        self.bar.update(event["done"] - self.bar.n)
        info["done"] = event["done"]
        milestone = int(10 * event["done"] / event["total"])
        if milestone > info["last_milestone"]:
            self.bar.write(
                f"{stage}: {event['done']:,}/{event['total']:,} ({100*event['done']/event['total']:.0f}%)"
            )
            info["last_milestone"] = milestone

    def _finish_stage(self):
        if self.bar is not None:
            self.bar.close()
            for info in self.stages.values():
                if "seconds" not in info:
                    info["seconds"] = time.perf_counter() - info["started"]
            self.bar = None

    def __exit__(self, typ, value, tb):
        self._finish_stage()
        report = dict(
            completed=typ is None,
            elapsed_seconds=time.perf_counter() - self.started,
            stages={
                stage: {k: v for k, v in info.items() if k not in ("started", "last_milestone")}
                for stage, info in self.stages.items()
            },
        )
        if self.output.exists():
            atomic_json(self.output / "last_execution.json", report)
        print(
            f"Сессия: {report['elapsed_seconds']/60:.1f} мин. "
            + ("Расчёт завершён." if typ is None else "Расчёт прерван; checkpoints сохранены."),
            flush=True,
        )


def audit_tuning_run(output):
    """Check the complete Cartesian coverage of saved score tables, without simulation."""
    output = Path(output)
    manifest = json.loads((output / "experiment.json").read_text())
    try:
        config = TuningConfig(**manifest["config"])
    except TypeError as error:
        raise ValueError("Saved tuning configuration is obsolete; use a new TUNING_ID") from error
    worlds = {s["id"] for s in scenarios(config)}
    rows = []
    for stage in ("coarse", "validation", "test"):
        ranks = pd.read_csv(output / f"{stage}_ranking.csv")
        scores = pd.read_csv(output / f"{stage}_realizations.csv")
        reps = getattr(config, f"{stage}_replicates")
        months = getattr(config, f"{stage}_months")
        candidate_ids = set(ranks.candidate)
        expected = len(candidate_ids) * len(worlds) * reps
        complete = (
            len(ranks) == len(candidate_ids)
            and len(scores) == expected
            and not scores.duplicated(["candidate", "scenario", "replicate"]).any()
            and set(scores.candidate) == candidate_ids
            and set(scores.scenario) == worlds
            and set(scores.replicate) == set(range(reps))
        )
        checkpoints = list((output / "scores" / stage).glob("*/*.json"))
        times = [p.stat().st_mtime for p in checkpoints]
        markets = list((output / "markets" / stage).glob("*.npz"))
        dimensions_ok = False
        if markets:
            with np.load(markets[0], allow_pickle=False) as sample:
                dimensions_ok = (
                    len(sample["starts"]) == months
                    and len(sample["logp"]) == config.warmup + months * config.D + 1
                )
        rows.append(
            dict(
                stage=stage,
                candidates=len(candidate_ids),
                scenarios=len(worlds),
                replicates=reps,
                blocks_per_path=months,
                score_rows=len(scores),
                expected_rows=expected,
                checkpoints=len(checkpoints),
                market_paths=len(markets),
                expected_paths=len(worlds) * reps,
                block_evaluations=expected * months,
                complete=bool(
                    complete
                    and len(checkpoints) == expected
                    and dimensions_ok
                    and len(markets) == len(worlds) * reps
                ),
                first_checkpoint=(
                    datetime.fromtimestamp(min(times)).isoformat(timespec="seconds")
                    if times
                    else None
                ),
                last_checkpoint=(
                    datetime.fromtimestamp(max(times)).isoformat(timespec="seconds")
                    if times
                    else None
                ),
                checkpoint_span_seconds=(max(times) - min(times) if times else 0.0),
            )
        )
    return pd.DataFrame(rows)
