"""Causal block decisions and realised rewards of the complete boundary-plus-policy strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boundary_analysis.curves import CASH, HOLD, POLICY
from estimation.pipeline import estimate
from market.data import block_frame, month_blocks
from market.metrics import summarize


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
    node_H, node_mu = bank.nearest_node(est.H, est.mu) if bank is not None else (np.nan, np.nan)
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


def run_backtest(
    logp,
    dates,
    *,
    window,
    D,
    bank,
    boundaries,
    values=None,
    cost=0.0,
    hurst_method="rs",
    blocks=None,
    progress=None,
    estimation_windows=None,
    **hurst_kwargs,
):
    """One row per block with estimates, decisions and rewards of all strategies."""
    logp = np.asarray(logp, dtype=float)
    if estimation_windows is not None:
        window = estimation_windows.warmup
    if blocks is None:
        blocks = month_blocks(len(logp), D, warmup=window)
    if not len(blocks):
        raise ValueError("No complete blocks after the warm-up window")
    frame = block_frame(dates, blocks)
    rows = []
    for n, (start, end) in enumerate(blocks):
        if start < window or end != start + D or end >= len(logp):
            raise ValueError("Invalid blocks or insufficient estimation history")
        est = estimate(
            logp[start - window : start + 1],
            D,
            hurst_method,
            windows=estimation_windows,
            **hurst_kwargs,
        )
        block = logp[start : end + 1]
        decision = decide_block(block, est, bank, boundaries, values)
        bh = float(block[-1] - block[0])
        best = int(np.argmax(block - block[0]))
        row = {
            **est.as_row(),
            "regime": decision.regime,
            "action": decision.action,
            "tau_day": decision.tau_day,
            "node_H": decision.node_H,
            "node_mu": decision.node_mu,
            "stopped_at_zero": decision.stopped_at_zero,
            "V_hat": decision.V_hat,
            "r_model": _reward(block, decision.tau_day, cost),
            "r_regimes_only": _reward(block, 0 if decision.regime == CASH else D, cost),
            "r_sign_drift": _reward(block, D if est.a > 0 else 0, cost),
            "r_buy_hold": bh - 2 * cost,
            "r_oracle": _reward(block, best, cost),
            "oracle_day": best,
            "pred_model": _predicted(decision, est),
            "pred_buy_hold": est.a,
        }
        row.update(
            boundary_mode=getattr(boundaries, "mode", "fixed"),
            boundary_gain_mode=getattr(boundaries, "gain_mode", "legacy"),
            boundary_alpha=getattr(boundaries, "alpha", None),
            boundary_source=getattr(boundaries, "source", ""),
            boundary_epsilon=float(boundaries.epsilon_at(est.H)),
        )
        rows.append(row)
        if progress is not None:
            progress(n + 1, len(blocks))
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result.attrs.update(
        {
            "window": window,
            "D": D,
            "cost": cost,
            "hurst_method": hurst_method,
            "estimation_windows": (
                None
                if estimation_windows is None
                else {
                    "H": estimation_windows.H,
                    "drift": estimation_windows.drift,
                    "sigma": estimation_windows.sigma,
                }
            ),
            "boundaries": getattr(boundaries, "label", ""),
            "epsilon": getattr(boundaries, "epsilon", np.nan),
            "boundary_mode": getattr(boundaries, "mode", "fixed"),
            "boundary_gain_mode": getattr(boundaries, "gain_mode", "legacy"),
            "boundary_alpha": getattr(boundaries, "alpha", None),
            "boundary_source": getattr(boundaries, "source", ""),
        }
    )
    return result


def _predicted(decision, est):
    """Expected log reward implied by the fitted model for the chosen regime."""
    if decision.action == CASH:
        return 0.0
    if decision.action == HOLD:
        return est.a
    return est.sigma * decision.V_hat if np.isfinite(decision.V_hat) else np.nan


def window_search(
    logp, dates, *, windows, split_date, D, bank, boundaries, values=None, objective="model", **kw
):
    """Metrics per estimation window, split into in-sample and out-of-sample halves.

    The window is chosen by the in-sample mean log reward of ``objective``;
    the out-of-sample column of that row is the honest estimate.
    """
    split = pd.Timestamp(split_date)
    rows, results = [], {}
    for w in windows:
        res = run_backtest(
            logp, dates, window=w, D=D, bank=bank, boundaries=boundaries, values=values, **kw
        )
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
