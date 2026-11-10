"""Equal-group synthetic scores and bootstrap differences across independent replicas."""

import numpy as np
import pandas as pd

from market.metrics import max_drawdown


def score_rows(rows, confidence=0.95):
    """Equal weights for stationary/changing groups, then scenarios and replicas."""
    frame = pd.DataFrame(rows)
    values = [
        "model",
        "buy_hold",
        "regimes_only",
        "sign_drift",
        "true_parameters",
        "H_bias",
        "a_bias",
        "sigma_bias",
        "H_mse",
        "a_mse",
        "sigma_mse",
        "cash_share",
        "policy_share",
        "drawdown",
        "model_second_moment",
        "buy_hold_second_moment",
    ]
    scenario = frame.groupby(["replicate", "group", "scenario"])[values].mean()
    grouped = scenario.groupby(["replicate", "group"]).mean().groupby("replicate").mean()
    score = grouped.mean().to_dict()
    score["score"] = score["model"]
    score["replicate_se"] = float(grouped.model.std(ddof=1) / np.sqrt(len(grouped)))
    score["mean_diff"] = score["model"] - score["buy_hold"]
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(grouped), size=(2000, len(grouped)))
    differences = (grouped.model - grouped.buy_hold).to_numpy()[indices].mean(axis=1)
    delta = 1 - confidence
    score["diff_low"], score["diff_high"] = map(
        float, np.quantile(differences, [delta / 2, 1 - delta / 2])
    )
    for name in ("model", "buy_hold"):
        std = float(np.sqrt(max(0.0, score[f"{name}_second_moment"] - score[name] ** 2)))
        score[f"{name}_std"] = std
        score[f"{name}_sharpe"] = score[name] / std * np.sqrt(12) if std > 0 else np.nan
    score["replicates"] = len(grouped)
    return score


def realization_row(
    candidate, stage, replicate, world, rewards, estimates, truth_reward, true_a, true_sigma
):
    """Summarise one candidate on one market without checkpoint or RNG side effects."""
    return dict(
        candidate=candidate.key,
        stage=stage,
        replicate=replicate,
        scenario=world["id"],
        group=world["group"],
        **{
            k: float(rewards[k].mean()) for k in ("model", "buy_hold", "regimes_only", "sign_drift")
        },
        model_second_moment=float(np.mean(rewards["model"] ** 2)),
        buy_hold_second_moment=float(np.mean(rewards["buy_hold"] ** 2)),
        true_parameters=truth_reward,
        H_bias=float(np.mean(estimates["H"] - world["H"])),
        a_bias=float(np.mean(estimates["a"] - true_a)),
        sigma_bias=float(np.mean(estimates["sigma"] - true_sigma)),
        H_mse=float(np.mean((estimates["H"] - world["H"]) ** 2)),
        a_mse=float(np.mean((estimates["a"] - true_a) ** 2)),
        sigma_mse=float(np.mean((estimates["sigma"] - true_sigma) ** 2)),
        cash_share=float(rewards["cash"].mean()),
        policy_share=float((~rewards["cash"] & ~rewards["hold"]).mean()),
        drawdown=float(-np.expm1(-max_drawdown(rewards["model"]))),
    )
