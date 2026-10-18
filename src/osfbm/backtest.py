"""Walk-forward evaluation of the fBm stopping model on daily prices.

For every block of D trading days the parameters are estimated on the W daily
returns ending at the decision close P_0 (nothing after P_0 is used), the
regime is read from the practical boundaries, and the reward is the realised
log return ln(P_tau / P_0) of the chosen selling day (0 when not buying).

Strategies reported for each block:

* ``model``        — cash below mu_1, hold above mu_2, trained rule in between;
* ``regimes_only`` — cash below mu_1, otherwise hold to the end;
* ``sign_drift``   — hold iff a_hat > 0 (the H = 1/2 theory);
* ``buy_hold``     — always hold the whole block;
* ``oracle``       — best close of the block in hindsight (an upper envelope).

``cost`` is a proportional cost per side in log units; a round trip costs 2*cost.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from osfbm.data import block_frame, month_blocks
from osfbm.estimation import estimate
from osfbm.policy_bank import CASH, HOLD, POLICY

STRATEGIES = ("model", "regimes_only", "sign_drift", "buy_hold", "oracle")
PER_YEAR = 12  # one block is one model month


@dataclass(frozen=True)
class BlockDecision:
    regime: str
    action: str
    tau_day: int
    node_H: float
    node_mu: float
    V_hat: float
    stopped_at_zero: bool


def normalised_path(logp_block, sigma, cfg, D):
    """(ln P_k - ln P_0) / sigma, linearly interpolated to the fine grid of cfg."""
    x = (np.asarray(logp_block, dtype=float) - logp_block[0]) / sigma
    block_times = np.arange(D + 1) / D * cfg.T
    if cfg.n_fine == D:
        return x[None, :]
    return np.interp(cfg.times, block_times, x)[None, :]


def decide_block(logp_block, est, bank, boundaries, values=None):
    """Regime, action and selling day for one block under the model."""
    D = len(logp_block) - 1
    regime = boundaries.regime(est.H, est.mu)
    node_H, node_mu = (bank.nearest_node(est.H, est.mu) if bank is not None
                       else (np.nan, np.nan))
    V_hat = values.value(est.H, est.mu) if values is not None else np.nan
    if regime == CASH:
        return BlockDecision(regime, CASH, 0, node_H, node_mu, V_hat, False)
    if regime == HOLD:
        return BlockDecision(regime, HOLD, D, node_H, node_mu, V_hat, False)
    if bank is None:
        raise ValueError("A PolicyBank is required for the middle regime")
    if bank.cfg.n_exercise != D:
        raise ValueError(f"Policy bank has n_exercise={bank.cfg.n_exercise}, block has D={D}")
    node_H, node_mu, policy = bank.nearest(est.H, est.mu)
    if policy.stopped_at_zero:
        return BlockDecision(regime, CASH, 0, node_H, node_mu, V_hat, True)
    X = normalised_path(logp_block, est.sigma, bank.cfg, D)
    j = int(policy.decide(X, bank.cfg)[0])
    tau_day = int(round(bank.cfg.exercise_times[j] / bank.cfg.T * D))
    return BlockDecision(regime, POLICY, tau_day, node_H, node_mu, V_hat, False)


def _reward(logp_block, tau_day, cost):
    if tau_day == 0:
        return 0.0
    return float(logp_block[tau_day] - logp_block[0] - 2 * cost)


def run_backtest(logp, dates, *, window, D, bank, boundaries, values=None,
                 cost=0.0, hurst_method="rs", blocks=None, progress=None, **hurst_kwargs):
    """One row per block with estimates, decisions and rewards of all strategies."""
    logp = np.asarray(logp, dtype=float)
    if blocks is None:
        blocks = month_blocks(len(logp), D, warmup=window)
    if not len(blocks):
        raise ValueError("No complete blocks after the warm-up window")
    frame = block_frame(dates, blocks)
    rows = []
    for n, (start, end) in enumerate(blocks):
        est = estimate(logp[start - window: start + 1], D, hurst_method, **hurst_kwargs)
        block = logp[start: end + 1]
        decision = decide_block(block, est, bank, boundaries, values)
        bh = float(block[-1] - block[0])
        best = int(np.argmax(block - block[0]))
        row = {
            **est.as_row(),
            "regime": decision.regime, "action": decision.action,
            "tau_day": decision.tau_day, "node_H": decision.node_H, "node_mu": decision.node_mu,
            "stopped_at_zero": decision.stopped_at_zero, "V_hat": decision.V_hat,
            "r_model": _reward(block, decision.tau_day, cost),
            "r_regimes_only": _reward(block, 0 if decision.regime == CASH else D, cost),
            "r_sign_drift": _reward(block, D if est.a > 0 else 0, cost),
            "r_buy_hold": bh - 2 * cost,
            "r_oracle": _reward(block, best, cost),
            "oracle_day": best,
            "pred_model": _predicted(decision, est),
            "pred_buy_hold": est.a,
        }
        rows.append(row)
        if progress is not None:
            progress(n + 1, len(blocks))
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result.attrs.update({"window": window, "D": D, "cost": cost, "hurst_method": hurst_method,
                         "boundaries": getattr(boundaries, "label", ""),
                         "epsilon": getattr(boundaries, "epsilon", np.nan)})
    return result


def _predicted(decision, est):
    """Expected log reward implied by the fitted model for the chosen regime."""
    if decision.action == CASH:
        return 0.0
    if decision.action == HOLD:
        return est.a
    return est.sigma * decision.V_hat if np.isfinite(decision.V_hat) else np.nan


def max_drawdown(log_returns):
    wealth = np.cumsum(np.asarray(log_returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], wealth]))[1:]
    return float(np.max(peak - wealth)) if len(wealth) else 0.0


def summarize(result, strategies=STRATEGIES):
    """Per-strategy monthly statistics; mean log reward is the model's objective."""
    rows = []
    for name in strategies:
        r = result[f"r_{name}"].to_numpy(float)
        n = len(r)
        mean, std = r.mean(), r.std(ddof=1) if n > 1 else np.nan
        invested = {"model": result["tau_day"] > 0,
                    "regimes_only": result["regime"] != CASH,
                    "sign_drift": result["a"] > 0,
                    "buy_hold": pd.Series(True, index=result.index),
                    "oracle": result["oracle_day"] > 0}[name]
        rows.append({
            "strategy": name, "months": n, "mean": mean, "std": std,
            "se": std / np.sqrt(n) if n > 1 else np.nan,
            "sharpe_annual": mean / std * np.sqrt(PER_YEAR) if std and std > 0 else np.nan,
            "cum_log": r.sum(), "max_drawdown": max_drawdown(r),
            "frac_invested": float(invested.mean()),
            "mean_tau_day": float(result["tau_day"].mean()) if name == "model" else np.nan,
        })
    return pd.DataFrame(rows).set_index("strategy")


def regime_shares(result):
    counts = result["regime"].value_counts().reindex([CASH, POLICY, HOLD], fill_value=0)
    return (counts / len(result)).rename("share")


def hac_se(x, lag=None):
    """Newey–West standard error of the mean with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = x - x.mean()
    s = np.sum(d * d) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.sum(d[k:] * d[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def stationary_bootstrap_indices(n, n_boot, mean_block, rng):
    """Politis–Romano stationary bootstrap index matrix (n_boot, n)."""
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    idx = np.empty((n_boot, n), dtype=int)
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
            continue
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def paired_difference(result, a="model", b="buy_hold", *, confidence=0.95,
                      n_boot=2000, seed=0, mean_block=None):
    """Mean of r_a - r_b with HAC t-test and a stationary-bootstrap interval."""
    d = (result[f"r_{a}"] - result[f"r_{b}"]).to_numpy(float)
    n = len(d)
    mean = d.mean()
    se = hac_se(d)
    rng = np.random.default_rng(seed)
    if mean_block is None:
        mean_block = max(2.0, n ** (1 / 3))
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
    boot = d[idx].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    t_stat = mean / se if se > 0 else np.nan
    return {"a": a, "b": b, "months": n, "mean_diff": mean, "hac_se": se, "t_hac": t_stat,
            "p_hac": float(2 * norm.sf(abs(t_stat))) if np.isfinite(t_stat) else np.nan,
            "boot_low": float(low), "boot_high": float(high), "confidence": confidence,
            "share_better": float(np.mean(d > 0)), "share_equal": float(np.mean(d == 0))}


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
    out = pd.DataFrame({"n": grouped.size(), "pred_mean": grouped[pred].mean(),
                        "real_mean": grouped[real].mean(),
                        "real_se": grouped[real].std(ddof=1) / np.sqrt(grouped.size())})
    out.index.name = "bin"
    return out


def window_search(logp, dates, *, windows, split_date, D, bank, boundaries, values=None,
                  objective="model", **kw):
    """Metrics per estimation window, split into in-sample and out-of-sample halves.

    The window is chosen by the in-sample mean log reward of ``objective``;
    the out-of-sample column of that row is the honest estimate.
    """
    split = pd.Timestamp(split_date)
    rows, results = [], {}
    for w in windows:
        res = run_backtest(logp, dates, window=w, D=D, bank=bank, boundaries=boundaries,
                           values=values, **kw)
        results[w] = res
        ins, outs = res[res.decision_date < split], res[res.decision_date >= split]
        row = {"window": w}
        for part, name in ((ins, "in"), (outs, "out"), (res, "all")):
            if len(part) < 2:
                continue
            s = summarize(part)
            for strat in ("model", "regimes_only", "buy_hold"):
                row[f"{name}_mean_{strat}"] = s.loc[strat, "mean"]
                row[f"{name}_sharpe_{strat}"] = s.loc[strat, "sharpe_annual"]
            row[f"{name}_months"] = len(part)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("window")
    key = f"in_mean_{objective}"
    chosen = int(table[key].idxmax()) if key in table and table[key].notna().any() else None
    table.attrs["chosen_window"] = chosen
    return table, results
