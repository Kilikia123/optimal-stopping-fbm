"""Synthetic market scenarios, deterministic seeds and causal decision blocks."""

import numpy as np

from estimation.simulation import simulate_log_prices
from infrastructure.identity import digest


def scenarios(config):
    out = []
    for h in config.true_H:
        for mu in config.stationary_mu:
            out.append(
                dict(
                    id=f"constant_H{h:g}_mu{mu:g}", group="constant", H=h, mu=mu, profile="constant"
                )
            )
        for profile in config.profiles:
            out.append(
                dict(id=f"{profile}_H{h:g}", group="changing", H=h, mu=None, profile=profile)
            )
    return out


def simulation_seed(config, stage, scenario, replicate, role="noise"):
    # Sharing the latent fBm across profiles is deliberate. Replicates remain
    # independent, and confidence intervals resample complete replicate groups.
    source = scenario["H"] if role == "noise" else scenario["id"]
    return int(digest(["synthetic-tuning-v1", config.seed, stage, source, replicate, role])[:8], 16)


def synthetic_market(config, stage, scenario, replicate, months):
    n = config.warmup + months * config.D
    base = simulate_log_prices(
        0.0,
        1.0,
        scenario["H"],
        n,
        config.D,
        seed=simulation_seed(config, stage, scenario, replicate),
    )[0]
    rng = np.random.default_rng(simulation_seed(config, stage, scenario, replicate, "phase"))
    period = 36 if scenario["profile"] == "drift36" else 12
    phase = int(rng.integers(0, 2 * period))
    regime = ((np.arange(n) // config.D + phase) // period) % 2
    sigma = np.full(n, 0.05)
    drift = np.full(n, 0.05 * scenario["mu"] if scenario["profile"] == "constant" else 0.005)
    if scenario["profile"] in ("drift12", "drift36", "joint"):
        drift = np.where(regime == 0, 0.01, -0.01)
    if scenario["profile"] in ("volatility", "joint"):
        sigma = np.where(regime == 0, 0.025, 0.075)
    returns = drift / config.D + sigma * np.diff(base)
    return dict(
        logp=np.r_[0.0, np.cumsum(returns)],
        true_a=drift,
        true_sigma=sigma,
        starts=np.arange(config.warmup, n, config.D),
        H=np.array(scenario["H"]),
    )
