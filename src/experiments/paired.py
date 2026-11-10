"""Replay policies on shared test paths and verify paired strategy-minus-hold moments."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import adapter
from experiments.configuration import _cfg
from experiments.moments import Moments
from experiments.policy import FixedPolicy
from infrastructure.identity import dependency_versions, digest, node_key
from infrastructure.provenance import implementation_fingerprint
from infrastructure.storage import atomic_json


def _paired_source(directory, h, mu, identity, cfg):
    key = node_key(h, mu)
    policy = json.loads((directory / "policies" / f"{key}.json").read_text())
    state = json.loads((directory / "states" / f"{key}.json").read_text())
    for item in (policy, state):
        if item["identity"] != identity or item["H"] != h or item["mu"] != mu:
            raise ValueError("Paired evaluation source identity mismatch")
    if state["moments"]["n"] != cfg.M_test:
        raise ValueError(f"Complete the original test first: H={h}, mu={mu}")
    if policy["policy"]["stopped_at_zero"] != state["stopped_at_zero"]:
        raise ValueError("Saved policy and test disagree about stopping at zero")
    return policy, state, digest([policy, state])


def paired_rewards(policy, B, mu, cfg):
    """Strategy reward and its pathwise difference from holding to T."""
    reward = policy.rewards(B, mu, cfg)
    return reward, reward - (B[:, -1] + mu * cfg.T)


def run_paired_evaluation(run_dir, *, H_values=None, batch_size=None, progress=None):
    """Replay saved policies on the original test paths; never fit or choose a policy.

    Checkpoints are separate from the source experiment. Full-size simulation
    with the original seed preserves the original random-number layout; only
    signature evaluation is batched. Resume is supported after each batch.
    """
    directory = Path(run_dir)
    manifest = json.loads((directory / "experiment.json").read_text())
    cfg, identity = _cfg(manifest), digest(manifest)
    batch_size = manifest["execution"]["batch_size"] if batch_size is None else batch_size
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    hs = list(cfg.H_grid) if H_values is None else list(H_values)
    if not hs or len(set(hs)) != len(hs) or any(h not in cfg.H_grid for h in hs):
        raise ValueError("H_values must be a nonempty unique subset of the saved H grid")
    output = directory / "paired_evaluation"
    # Record replay environment, independent of the source experiment's code hash.
    implementation = implementation_fingerprint("paired")
    replay = dict(
        schema_version=1,
        source_identity=identity,
        implementation=implementation,
        versions=dependency_versions(),
        target_n=cfg.M_test,
        kind="paired_strategy_minus_hold",
    )
    replay_path = output / "manifest.json"
    if replay_path.exists():
        if json.loads(replay_path.read_text()) != replay:
            raise ValueError(
                "Paired replay environment changed; use a separate source run or archive paired_evaluation"
            )
    else:
        atomic_json(replay_path, replay)
    replay_id = digest(replay)
    for h in hs:
        B = None
        for mu in cfg.mu_grid:
            artifact, source, source_hash = _paired_source(directory, h, mu, identity, cfg)
            if "paired_moments" in source and source["paired_moments"]["n"] == cfg.M_test:
                if progress:
                    progress(dict(stage="cached", H=h, mu=mu, n=cfg.M_test))
                continue
            path = output / "states" / f"{node_key(h, mu)}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved is not None and (
                saved["identity"] != replay_id or saved["source_hash"] != source_hash
            ):
                raise ValueError("Paired checkpoint does not match the source policy/test")
            rewards = Moments(**saved["reward_moments"]) if saved else Moments()
            gains = Moments(**saved["gain_moments"]) if saved else Moments()
            if rewards.n != gains.n or not 0 <= rewards.n <= cfg.M_test:
                raise ValueError("Invalid paired checkpoint count")
            if saved and saved["verified"] and rewards.n == cfg.M_test:
                if progress:
                    progress(dict(stage="cached", H=h, mu=mu, n=rewards.n))
                continue
            if B is None:
                B, _ = adapter.simulate_paths(
                    float(h),
                    cfg.replace(M_train=cfg.M_test),
                    manifest["seeds"][str(float(h))]["test"],
                )
            policy = FixedPolicy.from_dict(artifact["policy"])
            for start in range(rewards.n, cfg.M_test, batch_size):
                end = min(start + batch_size, cfg.M_test)
                reward, gain = paired_rewards(policy, B[start:end], mu, cfg)
                rewards.add(reward)
                gains.add(gain)
                atomic_json(
                    path,
                    dict(
                        identity=replay_id,
                        source_hash=source_hash,
                        H=h,
                        mu=mu,
                        reward_moments=asdict(rewards),
                        gain_moments=asdict(gains),
                        verified=False,
                    ),
                )
                if progress:
                    progress(dict(stage="test", H=h, mu=mu, n=rewards.n))
            # Guard against changed simulators/environment/policy implementations.
            original = Moments(**source["moments"])
            if not np.allclose(
                [rewards.mean, rewards.m2], [original.mean, original.m2], rtol=1e-8, atol=1e-10
            ):
                raise ValueError(f"Test replay differs from saved rewards: H={h}, mu={mu}")
            atomic_json(
                path,
                dict(
                    identity=replay_id,
                    source_hash=source_hash,
                    H=h,
                    mu=mu,
                    reward_moments=asdict(rewards),
                    gain_moments=asdict(gains),
                    verified=True,
                ),
            )
            if progress:
                progress(dict(stage="node", H=h, mu=mu, n=rewards.n, paired_gain=gains.mean))
    return output


def _attach_paired_statistics(directory, manifest, nodes, confidence):
    # New 03 runs store both moments in one atomic test checkpoint. This needs
    # neither a replay manifest nor a second evaluation of policy rewards.
    inline = []
    inline_seen = False
    source_identity = digest(manifest)
    cfg = _cfg(manifest)
    for file in (directory / "states").glob("*.json"):
        state = json.loads(file.read_text())
        if state["identity"] != source_identity:
            raise ValueError("Inline paired source identity mismatch")
        if "paired_moments" not in state:
            continue
        inline_seen = True
        moments = Moments(**state["paired_moments"])
        if moments.n != state["moments"]["n"]:
            raise ValueError("Paired and reward checkpoint counts differ")
        if moments.n != cfg.M_test:
            continue
        stats = moments.intervals(confidence, 2 * manifest["family_size"])
        inline.append(
            dict(
                H=state["H"],
                mu=state["mu"],
                paired_gain=stats["V"],
                paired_SE=stats["SE"],
                paired_n=stats["n"],
                paired_complete=True,
            )
        )
    path = directory / "paired_evaluation" / "manifest.json"
    if not path.exists():
        if inline_seen:
            return _merge_paired(nodes, inline, digest([source_identity, "inline_paired"]))
        raise FileNotFoundError(
            "No paired statistics. Run run_paired_evaluation in notebook 04 first; notebook 03 is not needed."
        )
    replay = json.loads(path.read_text())
    if replay["source_identity"] != digest(manifest) or replay["schema_version"] != 1:
        raise ValueError("Paired statistics belong to another experiment")
    cfg, replay_id = _cfg(manifest), digest(replay)
    source_identity = digest(manifest)
    rows = []
    for file in (path.parent / "states").glob("*.json"):
        state = json.loads(file.read_text())
        if state["identity"] != replay_id:
            raise ValueError("Paired checkpoint identity mismatch")
        if state["H"] not in cfg.H_grid or state["mu"] not in cfg.mu_grid:
            raise ValueError("Unexpected paired node")
        _, _, source_hash = _paired_source(directory, state["H"], state["mu"], source_identity, cfg)
        if state["source_hash"] != source_hash:
            raise ValueError("Paired statistics are stale: source policy/test changed")
        moments = Moments(**state["gain_moments"])
        if not state["verified"] or moments.n != cfg.M_test:
            continue
        stats = moments.intervals(confidence, 2 * manifest["family_size"])
        rows.append(
            dict(
                H=state["H"],
                mu=state["mu"],
                paired_gain=stats["V"],
                paired_SE=stats["SE"],
                paired_n=stats["n"],
                paired_complete=True,
            )
        )
    # Inline checkpoints are authoritative if both evaluation formats exist.
    inline_keys = {(r["H"], r["mu"]) for r in inline}
    rows = [r for r in rows if (r["H"], r["mu"]) not in inline_keys] + inline
    return _merge_paired(nodes, rows, digest([replay_id, "inline_paired"]) if inline else replay_id)


def _merge_paired(nodes, rows, identity):
    paired = pd.DataFrame(
        rows, columns=["H", "mu", "paired_gain", "paired_SE", "paired_n", "paired_complete"]
    )
    out = nodes.merge(paired, on=["H", "mu"], how="left", validate="one_to_one")
    out["paired_complete"] = out.paired_complete.eq(True)
    out.attrs.update(nodes.attrs, gain_mode="paired", paired_identity=identity)
    return out
