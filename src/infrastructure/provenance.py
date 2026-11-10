"""Exact implementation identities across the verified source-tree refactor.

A registered new source digest represents the same algorithm as its pre-refactor
digest. Only an exact match is translated: future source changes obtain a new
identity and cannot silently resume checkpoints made by another implementation.
The registry is data, not an override flag or a blanket schema-version exemption.
"""

import hashlib
import json

from infrastructure.paths import SOURCE_ROOT

SHARED = (
    "adapter.py",
    "config.py",
    "core.py",
    "infrastructure/identity.py",
    "infrastructure/storage.py",
    "infrastructure/paths.py",
    "infrastructure/provenance.py",
    "vendor/FBM_package.py",
    "vendor/Signature_computer.py",
    "vendor/Linear_signature_optimal_stopping.py",
)
POLICY = (
    "experiments/configuration.py",
    "experiments/moments.py",
    "experiments/policy.py",
    "experiments/paired.py",
    "boundary_analysis/inversion.py",
)
TRAINING = (*POLICY, "experiments/training.py", "experiments/statistics.py")
PROFILES = {
    "training": (*SHARED, *TRAINING),
    "paired": (*SHARED, *POLICY),
    "audit": (
        *SHARED,
        "experiments/audit.py",
        "experiments/policy.py",
        "experiments/moments.py",
        "experiments/legacy_grid.py",
        "boundary_analysis/fixed.py",
        "boundary_analysis/inversion.py",
    ),
    "tuning": (
        *SHARED,
        *TRAINING,
        "experiments/policy_bank.py",
        "boundary_analysis/relative.py",
        "boundary_analysis/monotone.py",
        "boundary_analysis/preparation.py",
        "boundary_analysis/curves.py",
        "estimation/__init__.py",
        "estimation/types.py",
        "estimation/preprocessing.py",
        "estimation/hurst.py",
        "estimation/drift.py",
        "estimation/volatility.py",
        "estimation/pipeline.py",
        "estimation/simulation.py",
        "estimation/uncertainty.py",
        "market/backtest.py",
        "market/metrics.py",
        "market/data.py",
        "tuning/config.py",
        "tuning/scenarios.py",
        "tuning/cache.py",
        "tuning/scoring.py",
        "tuning/runner.py",
        "tuning/frozen.py",
    ),
    "action": (
        *SHARED,
        *POLICY,
        "experiments/policy_bank.py",
        "experiments/statistics.py",
        "boundary_analysis/curves.py",
        "market/action_audit.py",
        "market/backtest.py",
        "market/statistics.py",
        "market/data.py",
        "market/metrics.py",
    ),
}


def source_digest(kind):
    """Hash declared implementation files, including their relative paths."""
    result = hashlib.sha256()
    for name in sorted(set(PROFILES[kind])):
        result.update(name.encode())
        result.update(b"\0")
        result.update((SOURCE_ROOT / name).read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def implementation_fingerprint(kind):
    """Return the legacy identity only for the exact verified refactor bundle."""
    current = source_digest(kind)
    registry_path = SOURCE_ROOT / "infrastructure" / "verified_refactor.json"
    if registry_path.exists():
        entry = json.loads(registry_path.read_text()).get(kind, {})
        if entry.get("refactored") == current:
            return entry["original"]
    return current
