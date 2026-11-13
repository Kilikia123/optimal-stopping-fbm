"""Fit backward Ridge policies, validate the initial decision and evaluate test paths."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

import adapter
from experiments.configuration import prepare_experiment
from experiments.moments import Moments
from experiments.policy import FixedPolicy
from experiments.statistics import _record, load_experiment
from infrastructure.identity import digest, node_key
from infrastructure.storage import atomic_json


def train_policy(B, mu, cfg, batch_size):
    """Same backward Ridge rule, retaining signatures only at exercise dates.

    Fine signatures are temporary per batch, not a 100000 x 201 x 31 tensor.
    """
    features = np.empty((len(B), cfg.n_exercise, cfg.sig_dim))
    payoff = B[:, cfg.exercise_index] + mu * cfg.exercise_times
    for start in range(0, len(B), batch_size):
        end = min(start + batch_size, len(B))
        X = adapter.drifted(B[start:end], mu, cfg)
        features[start:end] = adapter.signatures(X, cfg)[:, cfg.exercise_index, :]
    value = payoff[:, -1].copy()
    coefficients = np.empty((cfg.n_exercise - 1, cfg.sig_dim))
    intercepts = np.empty(cfg.n_exercise - 1)
    with adapter.silenced():
        for j in reversed(range(cfg.n_exercise - 1)):
            model = Ridge(alpha=cfg.ridge).fit(features[:, j, :], value)
            coefficients[j], intercepts[j] = model.coef_, model.intercept_
            stop = model.predict(features[:, j, :]) <= payoff[:, j]
            value[stop] = payoff[stop, j]
    return FixedPolicy(coefficients, intercepts, False)


def _moments_for_paths(policy, B, mu, cfg, batch_size):
    moments = Moments()
    for start in range(0, len(B), batch_size):
        moments.add(policy.rewards(B[start : start + batch_size], mu, cfg))
    return moments


def run_experiment(
    cfg, run_dir, *, batch_size=2_000, M_validation=10_000, progress=None, collect_paired=True
):
    if not isinstance(collect_paired, bool):
        raise ValueError("collect_paired must be boolean")
    directory = Path(run_dir)
    manifest = prepare_experiment(directory, cfg, batch_size=batch_size, M_validation=M_validation)
    identity = digest(manifest)
    for h in cfg.H_grid:
        pending = []
        for mu in cfg.mu_grid:
            key = node_key(h, mu)
            path = directory / "states" / f"{key}.json"
            state = json.loads(path.read_text()) if path.exists() else None
            if state is not None and state["identity"] != identity:
                raise ValueError("Checkpoint belongs to a different experiment")
            if (
                state is None
                or state["moments"]["n"] != cfg.M_test
                or not (directory / "nodes" / f"{key}.json").exists()
            ):
                pending.append((mu, key, state))
        if not pending:
            continue
        B_train = B_validation = B_test = None
        seeds = manifest["seeds"][str(float(h))]
        for mu, key, state in pending:
            policy_path = directory / "policies" / f"{key}.json"
            if policy_path.exists():
                artifact = json.loads(policy_path.read_text())
                if artifact["identity"] != identity:
                    raise ValueError("Policy belongs to a different experiment")
                policy = FixedPolicy.from_dict(artifact["policy"])
            else:
                if B_train is None:
                    B_train, _ = adapter.simulate_paths(float(h), cfg, seeds["train"])
                    B_validation, _ = adapter.simulate_paths(
                        float(h), cfg.replace(M_train=M_validation), seeds["validation"]
                    )
                if progress:
                    progress({"stage": "training", "H": h, "mu": mu})
                policy = train_policy(B_train, mu, cfg, batch_size)
                validation = _moments_for_paths(policy, B_validation, mu, cfg, batch_size)
                policy.stopped_at_zero = validation.mean <= 0
                artifact = {
                    "identity": identity,
                    "H": h,
                    "mu": mu,
                    "policy": policy.to_dict(),
                    "validation_mean": validation.mean,
                }
                atomic_json(policy_path, artifact)
            moments = Moments(**state["moments"]) if state else Moments()
            # Legacy partial checkpoints cannot acquire missing paired history
            # without replay. Finish those with their original statistics.
            paired = collect_paired and (
                state is None or "paired_moments" in state or moments.n == 0
            )
            if state and "paired_moments" in state and not collect_paired:
                raise ValueError("Resume this paired test with collect_paired=True")
            gains = (
                Moments(**state["paired_moments"])
                if state and "paired_moments" in state
                else Moments()
            )
            if paired and gains.n != moments.n:
                raise ValueError("Paired and reward checkpoint counts differ")
            if moments.n > cfg.M_test or (moments.n != cfg.M_test and moments.n % batch_size):
                raise ValueError("Invalid test checkpoint")
            if B_test is None and (paired or not policy.stopped_at_zero):
                B_test, _ = adapter.simulate_paths(
                    float(h), cfg.replace(M_train=cfg.M_test), seeds["test"]
                )
            for start in range(moments.n, cfg.M_test, batch_size):
                end = min(start + batch_size, cfg.M_test)
                rewards = (
                    np.zeros(end - start)
                    if policy.stopped_at_zero
                    else policy.rewards(B_test[start:end], mu, cfg)
                )
                moments.add(rewards)
                if paired:
                    gains.add(rewards - (B_test[start:end, -1] + mu * cfg.T))
                paired_fields = {"paired_moments": asdict(gains)} if paired else {}
                atomic_json(
                    directory / "states" / f"{key}.json",
                    {
                        "identity": identity,
                        "H": h,
                        "mu": mu,
                        "moments": asdict(moments),
                        "stopped_at_zero": policy.stopped_at_zero,
                        "validation_mean": artifact["validation_mean"],
                        **paired_fields,
                    },
                )
                if progress:
                    progress({"stage": "test", "H": h, "mu": mu, "n": moments.n})
            record = _record(
                h, mu, moments, policy.stopped_at_zero, artifact["validation_mean"], cfg, manifest
            )
            atomic_json(directory / "nodes" / f"{key}.json", record)
            if progress:
                progress({"stage": "node", **record})
        del B_train, B_validation, B_test
    return load_experiment(directory, epsilon=None)
