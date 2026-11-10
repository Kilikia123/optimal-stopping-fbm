"""Freeze and validate the selected configuration before deployment."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from boundary_analysis.curves import Boundaries
from boundary_analysis.preparation import boundary_object, complete_boundary_grid
from estimation.hurst import H_MAX, H_MIN
from experiments.policy_bank import PolicyBank
from infrastructure.identity import dependency_versions, digest
from infrastructure.provenance import implementation_fingerprint
from infrastructure.storage import atomic_json
from tuning.config import Candidate, TuningConfig


def _policy_hash(bank):
    digest = hashlib.sha256()
    for path in sorted((bank.run_dir / "policies").glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _experiment_identity(bank, config):
    return dict(
        schema_version=1,
        source_identity=bank.identity,
        policy_hash=_policy_hash(bank),
        python=platform.python_version(),
        config=asdict(config),
        versions=dependency_versions(),
        implementation=implementation_fingerprint("tuning"),
    )


def freeze_selection(output, bank, manifest, candidate, frame):
    """Freeze before the held-out test. Refuse to replace another winner."""
    output = Path(output)
    frozen = output / "frozen.json"
    frame = complete_boundary_grid(frame)
    content = frame.to_csv(index=False)
    payload = dict(
        schema_version=1,
        source_identity=bank.identity,
        run_dir=str(bank.run_dir.resolve()),
        tuning_identity=digest(manifest),
        candidate=asdict(candidate),
        candidate_id=candidate.key,
        D=bank.D,
        confidence=manifest["config"]["confidence"],
        cost=0.0,
        boundary_file="frozen_boundaries.csv",
        boundary_hash=digest(content),
        selection_stage="validation",
        objective="equal_group_mean_log_reward",
    )
    if frozen.exists():
        if json.loads(frozen.read_text()) != payload:
            raise ValueError("Winner is already frozen; use a new TUNING_ID")
        if (
            not (output / payload["boundary_file"]).exists()
            or digest((output / payload["boundary_file"]).read_text()) != payload["boundary_hash"]
        ):
            raise ValueError("Frozen boundary file changed")
    else:
        (output / payload["boundary_file"]).write_text(content)
        atomic_json(frozen, payload)
    return payload


@dataclass(frozen=True)
class FrozenSelection:
    candidate: Candidate
    boundaries: Boundaries
    run_dir: Path
    D: int
    identity: str

    @property
    def windows(self):
        return self.candidate.windows

    def backtest(self, logp, dates, *, bank, values=None, **kwargs):
        """Apply locked settings; reject attempts to override selected parameters."""
        from market.backtest import run_backtest

        if bank.identity != self.identity or bank.D != self.D:
            raise ValueError("Frozen selection belongs to another policy bank")
        if set(kwargs) - {"progress", "blocks"}:
            raise ValueError("Cannot override frozen settings")
        return run_backtest(
            logp,
            dates,
            window=self.windows.warmup,
            D=self.D,
            bank=bank,
            boundaries=self.boundaries,
            values=values,
            estimation_windows=self.windows,
            hurst_method=self.candidate.method,
            cost=0.0,
            clip=(max(H_MIN, float(bank.H_grid[0])), min(H_MAX, float(bank.H_grid[-1]))),
            **kwargs,
        )


def load_frozen_selection(path, *, run_dir=None):
    path = Path(path)
    if path.is_dir():
        path = path / "frozen.json"
    frozen = json.loads(path.read_text())
    audit_path = path.parent / "test_complete.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            "Finish the independent synthetic test in notebook 06 before using 07"
        )
    audit = json.loads(audit_path.read_text())
    if audit["frozen_identity"] != digest(frozen) or audit["test_ranking_hash"] != digest(
        (path.parent / "test_ranking.csv").read_text()
    ):
        raise ValueError("Synthetic test audit changed")
    manifest = json.loads((path.parent / "experiment.json").read_text())
    if frozen["schema_version"] != 1 or digest(manifest) != frozen["tuning_identity"]:
        raise ValueError("Frozen experiment manifest changed")
    directory = Path(frozen["run_dir"]) if run_dir is None else Path(run_dir)
    bank = PolicyBank(directory)
    if bank.identity != frozen["source_identity"] or bank.D != frozen["D"]:
        raise ValueError("Frozen selection belongs to another policy bank")
    try:
        config = TuningConfig(**manifest["config"])
    except TypeError as error:
        raise ValueError(
            "Saved tuning configuration is obsolete; rerun 06 with a new TUNING_ID"
        ) from error
    current = json.loads(json.dumps(_experiment_identity(bank, config)))
    if current != manifest:
        raise ValueError("Frozen policy bank, implementation or environment changed")
    text = (path.parent / frozen["boundary_file"]).read_text()
    if digest(text) != frozen["boundary_hash"]:
        raise ValueError("Frozen boundary file changed")
    from io import StringIO

    frame = complete_boundary_grid(pd.read_csv(StringIO(text)))
    c = Candidate(**frozen["candidate"])
    if c.key != frozen["candidate_id"]:
        raise ValueError("Frozen candidate identity mismatch")
    return FrozenSelection(
        c,
        boundary_object(
            frame,
            c.alpha,
            f"frozen {path.parent.name}; α={c.alpha:g}",
            str((path.parent / frozen["boundary_file"]).resolve()),
        ),
        directory,
        bank.D,
        bank.identity,
    )
