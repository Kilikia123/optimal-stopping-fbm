"""Independent, conditional Monte Carlo audit of saved stopping policies.

No dual bound: confidence intervals describe fixed policies on selected grid windows.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.stats import t

from osfbm import adapter
from osfbm.boundaries import boundaries_from_scan
from osfbm.config import Config
from osfbm.core import NodeResult
from osfbm.results import load_run


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
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
               for v in (self.n_test, self.batch_size)) or self.n_test < 2:
            raise ValueError("n_test >= 2 and batch_size >= 1 must be integers")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _key(h, mu):
    return _digest([float(h), float(mu)])[:20]


def _versions():
    return {name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "iisignature")}


def _implementation_hash():
    names = ["confidence.py", "adapter.py", "config.py",
             "vendor/Linear_signature_optimal_stopping.py", "vendor/FBM_package.py",
             "vendor/Signature_computer.py"]
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


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
                [NodeResult(**r) for r in group.to_dict("records")], cfg,
                epsilon=settings.epsilon,
            )
        for side in (1, 2):
            bracket = getattr(boundary, f"mu{side}_bracket", None)
            window = {"H": h, "side": side, "mus": [], "old_mu": None,
                      "old_bracket": bracket, "status": "source_not_localized",
                      "source_note": boundary.note if boundary is not None else "Исходная сетка неполна"}
            if bracket is not None:
                view = group[group.mu.between(bracket[0] - settings.window_radius - 1e-10,
                                              bracket[1] + settings.window_radius + 1e-10)]
                window.update(mus=view.mu.tolist(), status="selected",
                              old_mu=getattr(boundary, f"mu{side}_epsilon"))
                for row in view.to_dict("records"):
                    selected[_key(h, row["mu"])] = row
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
            seed = int(_digest(["audit", settings.seed, h, block])[:8], 16)
            while seed in used:
                seed = (seed + 1) % 2**32
            used.add(seed)
            seeds[str(h)].append(seed)
    manifest = {
        "schema_version": 1, "source_dir": str(source_dir),
        "source_sha256": hashlib.sha256((source_dir / "nodes.csv").read_bytes()).hexdigest(),
        "source_config": cfg.to_dict(), "settings": asdict(settings),
        "nodes": selected, "windows": windows, "family_size": len(selected),
        "seeds": seeds, "versions": _versions(), "implementation": _implementation_hash(),
    }
    path = audit_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Audit configuration/source/environment changed: use a new audit ID")
    else:
        _atomic_json(path, manifest)
    return manifest


@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Rewards must be a finite vector")
        if not values.size:
            return
        count = len(values)
        average = float(values.mean())
        delta = average - self.mean
        total = self.n + count
        self.m2 += float(np.sum((values - average)**2)) + delta**2 * self.n * count / total
        self.mean += delta * count / total
        self.n = total

    def intervals(self, confidence, family_size):
        if self.n < 2 or family_size < 1:
            raise ValueError("At least two rewards and one family member are required")
        se = np.sqrt(max(0.0, self.m2) / (self.n - 1) / self.n)
        alpha = 1 - confidence
        point = float(t.isf(alpha / 2, self.n - 1) * se)
        band = float(t.isf(alpha / (2 * family_size), self.n - 1) * se)
        return {"n": self.n, "V": self.mean, "SE": se,
                "ci_low": self.mean - point, "ci_high": self.mean + point,
                "band_low": self.mean - band, "band_high": self.mean + band}


@dataclass
class FixedPolicy:
    coefficients: np.ndarray
    intercepts: np.ndarray
    stopped_at_zero: bool

    def rewards(self, B, mu, cfg):
        if self.stopped_at_zero:
            return np.zeros(len(B))
        X = adapter.drifted(B, mu, cfg)
        sig = adapter.signatures(X, cfg)
        index = np.full(len(B), cfg.n_exercise - 1, dtype=int)
        active = np.ones(len(B), dtype=bool)
        for j in range(cfg.n_exercise - 1):
            k = cfg.exercise_index[j]
            continuation = sig[:, k, :] @ self.coefficients[j] + self.intercepts[j]
            stop = active & (continuation <= X[:, k])
            index[stop] = j
            active[stop] = False
        return X[np.arange(len(B)), cfg.exercise_index[index]]

    def to_dict(self):
        return {"coefficients": self.coefficients.tolist(),
                "intercepts": self.intercepts.tolist(),
                "stopped_at_zero": self.stopped_at_zero}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["coefficients"]), np.asarray(d["intercepts"]),
                   bool(d["stopped_at_zero"]))


def restore_policy(row, cfg, paths):
    """Reproduce the legacy fit; freeze its old t=0 choice before a fresh test."""
    mu = row["mu"]
    train = adapter.drifted(paths["B_train"], mu, cfg)
    test = adapter.drifted(paths["B_test"], mu, cfg)
    sig_train, sig_test = adapter.signatures(train, cfg), adapter.signatures(test, cfg)
    # Vendor may mutate the terminal payoff column; do not mutate source paths.
    raw, _, regressions = adapter.longstaff_schwartz(sig_train, train, sig_test, test, cfg)
    if not np.isclose(raw, row["V_raw"], rtol=1e-7, atol=1e-9):
        raise ValueError(f"Legacy policy mismatch H={row['H']} mu={mu}: "
                         f"saved={row['V_raw']}, reproduced={raw}")
    policy = FixedPolicy(np.asarray([r.coef_ for r in regressions]),
                         np.asarray([r.intercept_ for r in regressions]),
                         bool(row["stopped_at_zero"]))
    # Check our evaluator as well, before storing the artifact.
    replay = policy.rewards(paths["B_test"], mu, cfg).mean()
    expected = 0.0 if policy.stopped_at_zero else raw
    if not np.isclose(replay, expected, rtol=1e-7, atol=1e-9):
        raise ValueError("Serialized policy does not reproduce the original stopping rule")
    return policy


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
    if manifest["versions"] != _versions() or manifest["implementation"] != _implementation_hash():
        raise ValueError("Implementation/environment changed: create a new audit")
    cfg, settings = _source_config(manifest), AuditConfig(**manifest["settings"])
    identity = _digest(manifest)
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
                _atomic_json(file, {"audit_identity": identity, "H": h, "mu": row["mu"],
                                    "policy": policy.to_dict()})
                if progress:
                    progress({"stage": "policy", "H": h, "mu": row["mu"]})
            policies[key] = policy
            state_path = directory / "states" / f"{key}.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["audit_identity"] != identity:
                    raise ValueError("Checkpoint belongs to a different audit")
            else:
                state = {"audit_identity": identity, "next_block": 0,
                         "moments": asdict(Moments())}
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
                rewards = (np.zeros(count) if policies[key].stopped_at_zero
                           else policies[key].rewards(B, rows[key]["mu"], cfg))
                moments = Moments(**states[key]["moments"])
                moments.add(rewards)
                state = {"audit_identity": identity, "next_block": block + 1,
                         "moments": asdict(moments)}
                _atomic_json(directory / "states" / f"{key}.json", state)
                states[key] = state
                if progress:
                    progress({"stage": "test", "H": h, "mu": rows[key]["mu"], "n": moments.n})
    return load_audit(directory)


NODE_COLUMNS = ["H", "mu", "n", "V", "SE", "ci_low", "ci_high", "band_low", "band_high",
                "excess", "excess_ci_low", "excess_ci_high", "excess_band_low", "excess_band_high",
                "complete", "old_V", "old_SE", "stopped_at_zero"]


def load_audit(audit_dir):
    """Read-only analysis; K is always the full, preselected family, even if partial."""
    directory = Path(audit_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    settings, cfg = AuditConfig(**manifest["settings"]), _source_config(manifest)
    identity = _digest(manifest)
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
        record = {"H": row["H"], "mu": row["mu"],
                  **moments.intervals(settings.confidence, manifest["family_size"]),
                  "complete": moments.n == settings.n_test, "old_V": row["V"],
                  "old_SE": row["SE"], "stopped_at_zero": row["stopped_at_zero"]}
        record["excess"] = record["V"] - row["mu"] * cfg.T
        for field in ("ci_low", "ci_high", "band_low", "band_high"):
            record["excess_" + field] = record[field] - row["mu"] * cfg.T
        records.append(record)
    nodes = pd.DataFrame(records, columns=NODE_COLUMNS).sort_values(["H", "mu"])
    boundaries = []
    for window in manifest["windows"]:
        row = {"H": window["H"], "side": window["side"], "epsilon": settings.epsilon,
               "source_note": window["source_note"],
               "old_mu": window["old_mu"], "estimate": None, "mu_low": None, "mu_high": None,
               "window_low": min(window["mus"]) if window["mus"] else None,
               "window_high": max(window["mus"]) if window["mus"] else None,
               "grid_step_min": None, "grid_step_max": None, "status": window["status"]}
        if window["mus"]:
            steps = np.diff(window["mus"])
            if len(steps):
                row.update(grid_step_min=float(steps.min()), grid_step_max=float(steps.max()))
            view = nodes[nodes.H.eq(window["H"]) & nodes.mu.isin(window["mus"])]
            if len(view) != len(window["mus"]) or not view.complete.all():
                row["status"] = "incomplete_test"
            else:
                prefix = "" if window["side"] == 1 else "excess_"
                row.update(invert_band(view.mu.to_numpy(),
                                       view["V" if window["side"] == 1 else "excess"].to_numpy(),
                                       view[prefix + "band_low"].to_numpy(),
                                       view[prefix + "band_high"].to_numpy(),
                                       settings.epsilon, window["side"]))
        boundaries.append(row)
    return manifest, nodes.reset_index(drop=True), pd.DataFrame(boundaries)


def invert_band(mus, means, low, high, epsilon, side):
    """Confidence set for a window-restricted GRID extremum, without interpolation.

    On the simultaneous coverage event, definitely-in is a subset of the true
    threshold set, which is a subset of possibly-in. Max/min preserve inclusion
    bounds. We report no bounded interval without certified endpoints.
    """
    mus, means, low, high = map(lambda x: np.asarray(x, dtype=float), (mus, means, low, high))
    if (side not in (1, 2) or len(mus) < 2 or
            any(a.shape != mus.shape for a in (means, low, high)) or
            not np.isfinite(np.concatenate((mus, means, low, high))).all() or
            np.any(np.diff(mus) <= 0) or np.any(low > high)):
        raise ValueError("Invalid ordered grid/band")
    inside = means <= epsilon
    definite = high <= epsilon
    possible = low <= epsilon
    result = {"estimate": None, "mu_low": None, "mu_high": None, "status": "expand_window"}
    changes = np.diff(inside.astype(int))
    if np.any(changes > 0 if side == 1 else changes < 0):
        result["status"] = "multiple_transitions"
        return result
    if inside.any():
        result["estimate"] = float(mus[np.flatnonzero(inside)[-1 if side == 1 else 0]])
    enclosed = (definite[0] and not possible[-1]) if side == 1 else (not possible[0] and definite[-1])
    if not enclosed:
        return result
    if side == 1:
        lower, upper = mus[definite].max(), mus[possible].max()
    else:
        lower, upper = mus[possible].min(), mus[definite].min()
    result.update(mu_low=float(lower), mu_high=float(upper), status="localized_grid")
    return result
