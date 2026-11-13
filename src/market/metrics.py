"""Descriptive reward, risk, drawdown and calibration metrics."""

import numpy as np
import pandas as pd

from boundary_analysis.curves import CASH, HOLD, POLICY

STRATEGIES = ("model", "regimes_only", "sign_drift", "buy_hold", "oracle")


PER_YEAR = 12  # one block is one model month


def max_drawdown(log_returns):
    wealth = np.cumsum(np.asarray(log_returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], wealth]))[1:]
    return float(np.max(peak - wealth)) if len(wealth) else 0.0


def downside_std(log_returns, target=0.0):
    """Root mean square of the shortfalls below ``target`` (all months in the denominator)."""
    r = np.asarray(log_returns, dtype=float)
    short = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(short**2))) if len(r) else np.nan


def sortino(log_returns, target=0.0, per_year=PER_YEAR):
    """Annualised Sortino ratio; NaN when no month falls below the target."""
    r = np.asarray(log_returns, dtype=float)
    down = downside_std(r, target)
    return float((r.mean() - target) / down * np.sqrt(per_year)) if down > 0 else np.nan


def summarize(result, strategies=STRATEGIES):
    """Per-strategy monthly statistics; mean log reward is the model's objective.

    ``cagr``, ``sharpe_annual``, ``sortino_annual`` and the drawdowns are the
    second layer for a practitioner's reading. Months in cash earn zero, not the
    risk-free rate, so the ratios are conservative for the model.
    """
    rows = []
    for name in strategies:
        r = result[f"r_{name}"].to_numpy(float)
        n = len(r)
        mean, std = r.mean(), r.std(ddof=1) if n > 1 else np.nan
        mdd = max_drawdown(r)
        invested = {
            "model": result["tau_day"] > 0,
            "regimes_only": result["regime"] != CASH,
            "sign_drift": result["a"] > 0,
            "buy_hold": pd.Series(True, index=result.index),
            "oracle": result["oracle_day"] > 0,
        }[name]
        rows.append(
            {
                "strategy": name,
                "months": n,
                "mean": mean,
                "std": std,
                "se": std / np.sqrt(n) if n > 1 else np.nan,
                "cagr": float(np.expm1(mean * PER_YEAR)),
                "sharpe_annual": mean / std * np.sqrt(PER_YEAR) if std and std > 0 else np.nan,
                "sortino_annual": sortino(r),
                "cum_log": r.sum(),
                "max_drawdown": mdd,
                "max_drawdown_pct": float(-np.expm1(-mdd)),
                "frac_invested": float(invested.mean()),
                "mean_tau_day": float(result["tau_day"].mean()) if name == "model" else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def regime_shares(result):
    counts = result["regime"].value_counts().reindex([CASH, POLICY, HOLD], fill_value=0)
    return (counts / len(result)).rename("share")


def by_period(result, strategies=STRATEGIES, freq="10YS"):
    """Mean monthly log reward of each strategy per calendar period."""
    grouped = result.set_index("decision_date").groupby(pd.Grouper(freq=freq))
    table = grouped[[f"r_{s}" for s in strategies]].mean()
    table.columns = list(strategies)
    table["months"] = grouped.size()
    return table[table["months"] > 0]


def calibration(result, pred="pred_model", real="r_model", bins=10):
    """Realised against predicted reward, binned by the prediction."""
    frame = result[[pred, real]].dropna()
    if frame.empty:
        return pd.DataFrame(columns=["n", "pred_mean", "real_mean", "real_se"])
    q = pd.qcut(frame[pred].rank(method="first"), bins, labels=False)
    grouped = frame.groupby(q)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "pred_mean": grouped[pred].mean(),
            "real_mean": grouped[real].mean(),
            "real_se": grouped[real].std(ddof=1) / np.sqrt(grouped.size()),
        }
    )
    out.index.name = "bin"
    return out
