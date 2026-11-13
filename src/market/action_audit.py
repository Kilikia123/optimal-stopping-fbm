"""Archived fixed-action diagnostics; not used by the current notebook workflow."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.identity import dependency_versions, digest, node_key
from infrastructure.provenance import implementation_fingerprint
from infrastructure.storage import atomic_json
from market.backtest import normalised_path
from market.data import load_prices
from market.statistics import stationary_bootstrap_indices

REGIMES = ("cash", "policy", "hold")


REWARD_ATOL = 1e-7  # permits the six-decimal local price cache, not different observations


def _validate_configuration(rows, selection, bank):
    if bank.identity != selection.identity or bank.D != selection.D:
        raise ValueError("Action audit requires the frozen policy bank")
    c = selection.candidate
    for name, value in (
        ("window_H", c.H),
        ("window_a", c.drift),
        ("window_sigma", c.sigma),
        ("method", c.method),
        ("boundary_alpha", c.alpha),
    ):
        if name not in rows or not rows[name].eq(value).all():
            raise ValueError(f"Saved backtest differs from frozen configuration: {name}")
    regimes = [selection.boundaries.regime(h, m) for h, m in zip(rows.H, rows.mu)]
    if not np.array_equal(regimes, rows.regime):
        raise ValueError("Saved regimes do not match the frozen boundaries")


def evaluate_actions(logp, dates, result, *, bank, selection, progress=None):
    """Apply the saved policy in every region, including its fixed t=0 choice."""
    required = {
        "decision_idx",
        "end_idx",
        "decision_date",
        "end_date",
        "H",
        "mu",
        "sigma",
        "regime",
        "r_model",
        "r_buy_hold",
        "tau_day",
        "node_H",
        "node_mu",
    }
    if not required <= set(result):
        raise ValueError(f"Missing backtest columns: {sorted(required-set(result))}")
    rows = result.reset_index(drop=True).copy()
    if rows.empty:
        raise ValueError("No saved blocks to audit")
    logp = np.asarray(logp, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if (
        logp.ndim != 1
        or len(logp) != len(dates)
        or not np.isfinite(logp).all()
        or not dates.is_monotonic_increasing
        or dates.has_duplicates
        or dates.hasnans
    ):
        raise ValueError("Invalid local price history")
    numeric = rows[
        [
            "decision_idx",
            "end_idx",
            "H",
            "mu",
            "sigma",
            "r_model",
            "r_buy_hold",
            "tau_day",
            "node_H",
            "node_mu",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all() or (rows.sigma <= 0).any():
        raise ValueError("Nonfinite backtest values or invalid sigma")
    for name in ("decision_idx", "end_idx", "tau_day"):
        if (rows[name] % 1 != 0).any():
            raise ValueError(f"Noninteger {name}")
    starts = rows.decision_idx.to_numpy(int)
    ends = rows.end_idx.to_numpy(int)
    D = selection.D
    if (
        np.any(starts < 0)
        or np.any(ends >= len(logp))
        or np.any(ends - starts != D)
        or np.any(starts[1:] != ends[:-1])
        or np.any(rows.tau_day < 0)
        or np.any(rows.tau_day > D)
    ):
        raise ValueError("Require original consecutive nonoverlapping D-day blocks")
    if not np.array_equal(
        dates[starts].to_numpy(), pd.to_datetime(rows.decision_date).to_numpy()
    ) or not np.array_equal(dates[ends].to_numpy(), pd.to_datetime(rows.end_date).to_numpy()):
        raise ValueError("Local price dates differ from the saved backtest")
    _validate_configuration(rows, selection, bank)
    hold = logp[ends] - logp[starts]
    factual = logp[starts + rows.tau_day.to_numpy(int)] - logp[starts]
    if not np.allclose(hold, rows.r_buy_hold, atol=REWARD_ATOL, rtol=1e-8) or not np.allclose(
        factual, rows.r_model, atol=REWARD_ATOL, rtol=1e-8
    ):
        raise ValueError(
            "Local rewards differ from saved zero-cost backtest; check prices and costs"
        )
    policy_rewards = []
    policy_days = []
    stopped_zero = []
    for i, row in rows.iterrows():
        h, mu, policy = bank.nearest(float(row.H), float(row.mu))
        if not np.allclose([h, mu], [row.node_H, row.node_mu], atol=1e-12, rtol=0):
            raise ValueError("Saved nearest policy node differs from the frozen bank")
        if policy.stopped_at_zero:
            day = 0
        else:
            block = logp[starts[i] : ends[i] + 1]
            x = normalised_path(block, float(row.sigma), bank.cfg, D)
            j = int(policy.decide(x, bank.cfg)[0])
            day = int(round(bank.cfg.exercise_times[j] / bank.cfg.T * D))
        reward = float(logp[starts[i] + day] - logp[starts[i]])
        if row.regime == "policy" and (
            day != int(row.tau_day)
            or not np.isclose(reward, row.r_model, atol=REWARD_ATOL, rtol=1e-8)
        ):
            raise ValueError("Policy-region decision does not reproduce the original backtest")
        policy_rewards.append(reward)
        policy_days.append(day)
        stopped_zero.append(bool(policy.stopped_at_zero))
        if progress:
            progress(i + 1, len(rows))
    rows["r_cash"] = 0.0
    rows["r_hold"] = hold
    rows["r_policy"] = policy_rewards
    rows["tau_policy_day"] = policy_days
    rows["policy_stopped_at_zero"] = stopped_zero
    return rows


def summarize_actions(rows, *, confidence=0.95, n_boot=2000, seed=7, mean_block=None):
    """Bootstrap whole chronological sequences jointly with their regime labels.

    Conditional means are recomputed in each draw. Filtering a regime before
    resampling would compress calendar gaps and misrepresent dependence.
    Intervals are exploratory, not corrected for multiple comparisons/selection.
    """
    if (
        not 0 < confidence < 1
        or not isinstance(n_boot, int)
        or isinstance(n_boot, bool)
        or n_boot < 20
        or len(rows) < 2
        or not rows.regime.isin(REGIMES).all()
    ):
        raise ValueError("Require >=2 rows, valid regimes, confidence and n_boot>=20")
    if not {"decision_idx", "end_idx"} <= set(rows) or not np.array_equal(
        rows.decision_idx.to_numpy()[1:], rows.end_idx.to_numpy()[:-1]
    ):
        raise ValueError("Bootstrap requires the full consecutive chronology, not filtered regimes")
    rewards = rows[["r_cash", "r_hold", "r_policy", "r_model"]].to_numpy(float)
    if not np.isfinite(rewards).all():
        raise ValueError("Nonfinite audit rewards")
    n = len(rows)
    mean_block = max(2.0, n ** (1 / 3)) if mean_block is None else mean_block
    if not np.isfinite(mean_block) or mean_block < 1:
        raise ValueError("mean_block must be finite and >=1")
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, np.random.default_rng(seed))
    codes = pd.Categorical(rows.regime, categories=REGIMES).codes
    sampled_codes = codes[idx]
    summary = []
    comparisons = []
    delta = 1 - confidence
    for code, region in enumerate(REGIMES):
        mask = codes == code
        count = int(mask.sum())
        difference = rewards[:, 3] - rewards[:, 1]
        row = dict(
            regime=region,
            n=count,
            model_minus_hold_sum=float(difference[mask].sum()),
            model_minus_hold_contribution=float(difference[mask].sum() / n),
            policy_stopped_zero_share=(
                float(rows.loc[mask, "policy_stopped_at_zero"].mean()) if count else np.nan
            ),
        )
        for j, name in enumerate(("cash", "hold", "policy", "model")):
            row[f"mean_{name}"] = float(rewards[mask, j].mean()) if count else np.nan
        summary.append(row)
        sampled_mask = sampled_codes == code
        denominator = sampled_mask.sum(axis=1)
        for a, b, ai, bi in (
            ("policy", "hold", 2, 1),
            ("policy", "cash", 2, 0),
            ("hold", "cash", 1, 0),
        ):
            d = rewards[:, ai] - rewards[:, bi]
            means = np.divide(
                np.where(sampled_mask, d[idx], 0).sum(axis=1),
                denominator,
                out=np.full(n_boot, np.nan),
                where=denominator > 0,
            )
            valid = means[np.isfinite(means)]
            supported = count >= 2 and len(valid) >= max(20, n_boot // 2)
            low, high = (
                np.quantile(valid, [delta / 2, 1 - delta / 2]) if supported else (np.nan, np.nan)
            )
            comparisons.append(
                dict(
                    regime=region,
                    a=a,
                    b=b,
                    n=count,
                    mean_diff=float(d[mask].mean()) if count else np.nan,
                    boot_low=low,
                    boot_high=high,
                    confidence=confidence,
                    share_better=float(np.mean(d[mask] > 0)) if count else np.nan,
                    share_equal=float(np.mean(d[mask] == 0)) if count else np.nan,
                    valid_bootstraps=len(valid),
                    status="estimated" if supported else "insufficient_data",
                )
            )
    return pd.DataFrame(summary), pd.DataFrame(comparisons)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_action_audit(
    output,
    selection,
    *,
    bank,
    prices_path,
    recompute=False,
    confidence=0.95,
    n_boot=2000,
    seed=7,
    mean_block=None,
    progress=None,
):
    """Explicit one-time replay, or read validated cached diagnostics. No network."""
    output = Path(output)
    prices_path = Path(prices_path)
    if not prices_path.exists():
        raise FileNotFoundError(
            f"No local price cache: {prices_path}; load the original data first"
        )
    source = output / "blocks.csv"
    result = pd.read_csv(source, parse_dates=["decision_date", "end_date"])
    _validate_configuration(result, selection, bank)
    nodes = sorted({bank.nearest_node(h, m) for h, m in zip(result.H, result.mu)})
    policy_hash = digest(
        [(h, m, _sha(bank.run_dir / "policies" / f"{node_key(h,m)}.json")) for h, m in nodes]
    )
    identity = dict(
        schema_version=1,
        source_backtest_hash=_sha(source),
        prices_hash=_sha(prices_path),
        source_backtest=str(source.resolve()),
        prices_source=str(prices_path.resolve()),
        source_identity=bank.identity,
        policy_hash=policy_hash,
        candidate=selection.candidate.key,
        boundary_hash=digest(selection.boundaries.frame().to_dict("list")),
        implementation=implementation_fingerprint("action"),
        reward_atol=REWARD_ATOL,
        versions=dependency_versions(),
        cost=0.0,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
        mean_block=mean_block,
    )
    folder = output / "action_audit"
    meta = folder / "metadata.json"
    names = ("blocks", "summary", "comparisons")
    if not recompute:
        if not meta.exists():
            raise FileNotFoundError(
                "No action audit yet. Set RECOMPUTE_ACTION_AUDIT=True in 07 for its first calculation."
            )
        saved = json.loads(meta.read_text())
        if saved["inputs"] != identity:
            raise ValueError(
                "Action audit inputs/settings changed; explicitly recompute diagnostics"
            )
        for name in names:
            if _sha(folder / f"{name}.csv") != saved["outputs"][name]:
                raise ValueError(f"Saved action audit changed: {name}")
        tables = {name: pd.read_csv(folder / f"{name}.csv") for name in names}
    else:
        prices = load_prices(prices_path)
        rows = evaluate_actions(
            np.log(prices.to_numpy()),
            prices.index,
            result,
            bank=bank,
            selection=selection,
            progress=progress,
        )
        summary, comparisons = summarize_actions(
            rows, confidence=confidence, n_boot=n_boot, seed=seed, mean_block=mean_block
        )
        tables = dict(blocks=rows, summary=summary, comparisons=comparisons)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
        atomic_json(
            meta,
            dict(
                inputs=identity,
                outputs={name: _sha(folder / f"{name}.csv") for name in names},
                interpretation="Post-hoc fixed-action diagnostic; not a hindsight strategy",
            ),
        )
    tables["output"] = folder
    return tables
