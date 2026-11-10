"""Independent Monte Carlo audit of fixed policies with resumable checkpoints."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import adapter
from boundary_analysis.fixed import boundaries_from_scan
from boundary_analysis.inversion import invert_band
from config import Config
from core import NodeResult
from experiments.legacy_grid import load_run
from experiments.moments import Moments
from experiments.policy import FixedPolicy, restore_policy
from infrastructure.identity import dependency_versions, digest, node_key
from infrastructure.storage import atomic_json


@dataclass(frozen=True)
class AuditConfig:
    epsilon: float = 0.03
    confidence: float = 0.95
    n_test: int = 100_000
    batch_size: int = 2_000
    window_radius: float = 0.10
    seed: int = 20260917

    def __post_init__(self):
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0, 1)")
        if not np.isfinite(self.window_radius) or self.window_radius < 0:
            raise ValueError("window_radius must be finite and nonnegative")
        if (
            any(
                not isinstance(v, int) or isinstance(v, bool) or v <= 0
                for v in (self.n_test, self.batch_size)
            )
            or self.n_test < 2
        ):
            raise ValueError("n_test >= 2 and batch_size >= 1 must be integers")


def _implementation_hash():
    """Identity of the audit algorithm, including verified layout compatibility."""
    from infrastructure.provenance import implementation_fingerprint

    return implementation_fingerprint("audit")


def prepare_audit(source_dir, audit_dir, settings=AuditConfig()):
    """Freeze node selection before drawing the independent audit sample."""
    source_dir, audit_dir = Path(source_dir).resolve(), Path(audit_dir).resolve()
    cfg, nodes = load_run(source_dir)
    selected, windows = {}, []
    for h in cfg.H_grid:
        group = nodes[nodes.H.eq(h)].sort_values("mu")
        complete = len(group) == len(cfg.mu_grid)
        boundary = None
        if complete:
            boundary = boundaries_from_scan(
                [NodeResult(**r) for r in group.to_dict("records")],
                cfg,
                epsilon=settings.epsilon,
            )
        for side in (1, 2):
            bracket = getattr(boundary, f"mu{side}_bracket", None)
            window = {
                "H": h,
                "side": side,
                "mus": [],
                "old_mu": None,
                "old_bracket": bracket,
                "status": "source_not_localized",
                "source_note": boundary.note if boundary is not None else "Исходная сетка неполна",
            }
            if bracket is not None:
                view = group[
                    group.mu.between(
                        bracket[0] - settings.window_radius - 1e-10,
                        bracket[1] + settings.window_radius + 1e-10,
                    )
                ]
                window.update(
                    mus=view.mu.tolist(),
                    status="selected",
                    old_mu=getattr(boundary, f"mu{side}_epsilon"),
                )
                for row in view.to_dict("records"):
                    selected[node_key(h, row["mu"])] = row
            elif not complete:
                window["status"] = "source_incomplete"
            windows.append(window)
    # Reserve original training/test seeds across the whole source run. All audit
    # blocks have distinct seeds, and all mu at a given H share each block.
    used = set()
    for h in cfg.H_grid:
        base = abs(hash((cfg.seed, round(float(h), 6)))) % (2**31 - 1000)
        used.update((base, base + 1))
    seeds = {}
    n_blocks = (settings.n_test + settings.batch_size - 1) // settings.batch_size
    for h in sorted({r["H"] for r in selected.values()}):
        seeds[str(h)] = []
        for block in range(n_blocks):
            seed = int(digest(["audit", settings.seed, h, block])[:8], 16)
            while seed in used:
                seed = (seed + 1) % 2**32
            used.add(seed)
            seeds[str(h)].append(seed)
    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "source_sha256": hashlib.sha256((source_dir / "nodes.csv").read_bytes()).hexdigest(),
        "source_config": cfg.to_dict(),
        "settings": asdict(settings),
        "nodes": selected,
        "windows": windows,
        "family_size": len(selected),
        "seeds": seeds,
        "versions": dependency_versions(),
        "implementation": _implementation_hash(),
    }
    path = audit_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Audit configuration/source/environment changed: use a new audit ID")
    else:
        atomic_json(path, manifest)
    return manifest


def _source_config(manifest):
    raw = dict(manifest["source_config"])
    for name in ("mu_grid", "H_grid"):
        raw[name] = tuple(raw[name])
    return Config(**raw)


def run_audit(audit_dir, progress=None):
    """Restore policies, then evaluate fixed-size independent blocks with checkpoints.

    progress receives events after durable policy/block writes. Raising from the
    callback is safe; the next call resumes exactly at the saved block.
    """
    directory = Path(audit_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["versions"] != dependency_versions()
        or manifest["implementation"] != _implementation_hash()
    ):
        raise ValueError("Implementation/environment changed: create a new audit")
    cfg, settings = _source_config(manifest), AuditConfig(**manifest["settings"])
    identity = digest(manifest)
    for h in sorted({r["H"] for r in manifest["nodes"].values()}):
        rows = {k: r for k, r in manifest["nodes"].items() if r["H"] == h}
        policies, states = {}, {}
        paths = None
        for key, row in rows.items():
            file = directory / "policies" / f"{key}.json"
            if file.exists():
                artifact = json.loads(file.read_text())
                if artifact["audit_identity"] != identity:
                    raise ValueError("Policy belongs to a different audit")
                policy = FixedPolicy.from_dict(artifact["policy"])
            else:
                if paths is None:
                    paths = adapter.simulate_pair(h, cfg)
                policy = restore_policy(row, cfg, paths)
                atomic_json(
                    file,
                    {
                        "audit_identity": identity,
                        "H": h,
                        "mu": row["mu"],
                        "policy": policy.to_dict(),
                    },
                )
                if progress:
                    progress({"stage": "policy", "H": h, "mu": row["mu"]})
            policies[key] = policy
            state_path = directory / "states" / f"{key}.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["audit_identity"] != identity:
                    raise ValueError("Checkpoint belongs to a different audit")
            else:
                state = {"audit_identity": identity, "next_block": 0, "moments": asdict(Moments())}
            expected = min(state["next_block"] * settings.batch_size, settings.n_test)
            if state["moments"]["n"] != expected:
                raise ValueError("Invalid block checkpoint")
            states[key] = state
        del paths
        for block, seed in enumerate(manifest["seeds"][str(h)]):
            pending = [k for k in rows if states[k]["next_block"] == block]
            if not pending:
                continue
            count = min(settings.batch_size, settings.n_test - block * settings.batch_size)
            B = None
            if any(not policies[k].stopped_at_zero for k in pending):
                B, _ = adapter.simulate_paths(h, cfg.replace(M_train=count), seed)
            for key in pending:
                rewards = (
                    np.zeros(count)
                    if policies[key].stopped_at_zero
                    else policies[key].rewards(B, rows[key]["mu"], cfg)
                )
                moments = Moments(**states[key]["moments"])
                moments.add(rewards)
                state = {
                    "audit_identity": identity,
                    "next_block": block + 1,
                    "moments": asdict(moments),
                }
                atomic_json(directory / "states" / f"{key}.json", state)
                states[key] = state
                if progress:
                    progress({"stage": "test", "H": h, "mu": rows[key]["mu"], "n": moments.n})
    return load_audit(directory)


NODE_COLUMNS = [
    "H",
    "mu",
    "n",
    "V",
    "SE",
    "ci_low",
    "ci_high",
    "band_low",
    "band_high",
    "excess",
    "excess_ci_low",
    "excess_ci_high",
    "excess_band_low",
    "excess_band_high",
    "complete",
    "old_V",
    "old_SE",
    "stopped_at_zero",
]


def load_audit(audit_dir):
    """Read-only analysis; K is always the full, preselected family, even if partial."""
    directory = Path(audit_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    settings, cfg = AuditConfig(**manifest["settings"]), _source_config(manifest)
    identity = digest(manifest)
    records = []
    for key, row in manifest["nodes"].items():
        path = directory / "states" / f"{key}.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state["audit_identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        moments = Moments(**state["moments"])
        if moments.n < 2:
            continue
        record = {
            "H": row["H"],
            "mu": row["mu"],
            **moments.intervals(settings.confidence, manifest["family_size"]),
            "complete": moments.n == settings.n_test,
            "old_V": row["V"],
            "old_SE": row["SE"],
            "stopped_at_zero": row["stopped_at_zero"],
        }
        record["excess"] = record["V"] - row["mu"] * cfg.T
        for field in ("ci_low", "ci_high", "band_low", "band_high"):
            record["excess_" + field] = record[field] - row["mu"] * cfg.T
        records.append(record)
    nodes = pd.DataFrame(records, columns=NODE_COLUMNS).sort_values(["H", "mu"])
    boundaries = []
    for window in manifest["windows"]:
        row = {
            "H": window["H"],
            "side": window["side"],
            "epsilon": settings.epsilon,
            "source_note": window["source_note"],
            "old_mu": window["old_mu"],
            "estimate": None,
            "mu_low": None,
            "mu_high": None,
            "window_low": min(window["mus"]) if window["mus"] else None,
            "window_high": max(window["mus"]) if window["mus"] else None,
            "grid_step_min": None,
            "grid_step_max": None,
            "status": window["status"],
        }
        if window["mus"]:
            steps = np.diff(window["mus"])
            if len(steps):
                row.update(grid_step_min=float(steps.min()), grid_step_max=float(steps.max()))
            view = nodes[nodes.H.eq(window["H"]) & nodes.mu.isin(window["mus"])]
            if len(view) != len(window["mus"]) or not view.complete.all():
                row["status"] = "incomplete_test"
            else:
                prefix = "" if window["side"] == 1 else "excess_"
                row.update(
                    invert_band(
                        view.mu.to_numpy(),
                        view["V" if window["side"] == 1 else "excess"].to_numpy(),
                        view[prefix + "band_low"].to_numpy(),
                        view[prefix + "band_high"].to_numpy(),
                        settings.epsilon,
                        window["side"],
                    )
                )
        boundaries.append(row)
    return manifest, nodes.reset_index(drop=True), pd.DataFrame(boundaries)
