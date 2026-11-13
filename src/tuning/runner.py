"""Resumable coarse selection, validation, winner freezing and independent test."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_analysis.preparation import boundary_object, prepare_boundary_grids
from estimation.hurst import H_MAX, H_MIN
from experiments.policy_bank import PolicyBank
from infrastructure.identity import digest
from infrastructure.storage import atomic_json, atomic_npz
from tuning.cache import DecisionCache, EstimatorCache
from tuning.config import TuningConfig, candidates
from tuning.frozen import _experiment_identity, freeze_selection
from tuning.scenarios import scenarios, synthetic_market
from tuning.scoring import realization_row, score_rows


def _stage(stage, selected, frames, config, bank, output, progress):
    count = getattr(config, f"{stage}_replicates")
    months = getattr(config, f"{stage}_months")
    worlds = scenarios(config)
    all_rows = []
    total = count * len(worlds) * len(selected)
    done = 0
    by_key = {c.key: c for c in selected}
    for replicate in range(count):
        for world in worlds:
            cache_path = output / "markets" / stage / f"{world['id']}_r{replicate}.npz"
            data = None
            estimator = None
            decider = None
            truth_rewards = {}
            for c in selected:
                checkpoint = output / "scores" / stage / c.key / f"{world['id']}_r{replicate}.json"
                if checkpoint.exists():
                    row = json.loads(checkpoint.read_text())
                    if row["candidate"] != c.key or row["stage"] != stage:
                        raise ValueError("Unexpected score checkpoint")
                else:
                    if data is None:
                        if cache_path.exists():
                            with np.load(cache_path, allow_pickle=False) as saved:
                                data = {k: saved[k] for k in saved.files}
                        else:
                            data = synthetic_market(config, stage, world, replicate, months)
                            atomic_npz(cache_path, data)
                        clip = (
                            max(H_MIN, float(bank.H_grid[0])),
                            min(H_MAX, float(bank.H_grid[-1])),
                        )
                        estimator = EstimatorCache(data, config, clip)
                        decider = DecisionCache(data, bank, config.D)
                    estimates = estimator.get(c)
                    boundary = boundary_object(frames[c.alpha], c.alpha)
                    result = decider.rewards(estimates, boundary)
                    starts = data["starts"]
                    ta = data["true_a"][starts]
                    ts = data["true_sigma"][starts]
                    if c.alpha not in truth_rewards:
                        truth = dict(H=np.full(len(starts), world["H"]), a=ta, sigma=ts, mu=ta / ts)
                        truth_rewards[c.alpha] = float(
                            decider.rewards(truth, boundary)["model"].mean()
                        )
                    row = realization_row(
                        c,
                        stage,
                        replicate,
                        world,
                        result,
                        estimates,
                        truth_rewards[c.alpha],
                        ta,
                        ts,
                    )
                    atomic_json(checkpoint, row)
                all_rows.append(row)
                done += 1
                if progress:
                    progress(
                        dict(
                            stage=stage,
                            done=done,
                            total=total,
                            candidate=c.key,
                            replicate=replicate,
                            scenario=world["id"],
                        )
                    )
    result = []
    for key, group in pd.DataFrame(all_rows).groupby("candidate"):
        result.append(
            dict(
                candidate=key,
                **asdict(by_key[key]),
                **score_rows(group.to_dict("records"), config.confidence),
            )
        )
    result = (
        pd.DataFrame(result)
        .sort_values(
            ["score", "H", "drift", "sigma", "method", "alpha"],
            ascending=[False, True, True, True, True, True],
        )
        .reset_index(drop=True)
    )
    result.to_csv(output / f"{stage}_ranking.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output / f"{stage}_realizations.csv", index=False)
    return result


def run_synthetic_tuning(run_dir, output, config=TuningConfig(), *, progress=None):
    """Resume a synthetic experiment, freeze on validation, then audit once."""
    output = Path(output)
    bank = PolicyBank(run_dir)
    if bank.D != config.D or bank.cfg.T != 1.0:
        raise ValueError("Require policy horizon T=1 and n_exercise=D")
    if bank.coverage() != 1.0:
        raise ValueError("Complete the policy bank before synthetic tuning")
    if min(config.true_H) < bank.H_grid[0] or max(config.true_H) > bank.H_grid[-1]:
        raise ValueError("Synthetic H values fall outside the trained H grid")
    manifest = json.loads(json.dumps(_experiment_identity(bank, config)))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Synthetic settings/source/code changed; use a new TUNING_ID")
    else:
        atomic_json(path, manifest)
    frames = prepare_boundary_grids(run_dir, output, config)
    available = [c for c in candidates(config) if c.alpha in frames]
    by_key = {c.key: c for c in available}
    coarse = _stage("coarse", available, frames, config, bank, output, progress)
    shortlist = list(coarse.candidate.head(config.top_k))
    validation = _stage(
        "validation", [by_key[k] for k in shortlist], frames, config, bank, output, progress
    )
    winner = by_key[validation.candidate.iloc[0]]
    frozen = freeze_selection(output, bank, manifest, winner, frames[winner.alpha])
    test = _stage("test", [winner], frames, config, bank, output, progress)
    atomic_json(
        output / "test_complete.json",
        dict(
            frozen_identity=digest(frozen),
            test_ranking_hash=digest((output / "test_ranking.csv").read_text()),
        ),
    )
    return dict(coarse=coarse, validation=validation, test=test, frozen=frozen, output=output)
