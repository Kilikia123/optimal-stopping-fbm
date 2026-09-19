"""Fresh training and independent testing over the full parameter grid."""
from dataclasses import asdict
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from osfbm import adapter
from osfbm.config import Config
from osfbm.confidence import (
    FixedPolicy, Moments, _atomic_json, _digest, _key, _versions, invert_band,
)


def prepare_experiment(run_dir, cfg, *, batch_size=2_000, M_validation=10_000):
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
           for v in (batch_size, M_validation)):
        raise ValueError('batch_size and M_validation must be positive integers')
    directory = Path(run_dir)
    if cfg.M_train < 2 or cfg.M_test < 2:
        raise ValueError('Training and test sizes must be at least two')
    if (not cfg.H_grid or not cfg.mu_grid or len(set(cfg.H_grid)) != len(cfg.H_grid)
            or len(set(cfg.mu_grid)) != len(cfg.mu_grid)
            or not all(0 < h < 1 for h in cfg.H_grid)
            or not np.isfinite(cfg.mu_grid).all()):
        raise ValueError('Require unique finite grids and 0 < H < 1')
    if (directory / 'config.json').exists() and not (directory / 'experiment.json').exists():
        raise ValueError('Existing legacy run: choose a new RUN_ID')
    used, seeds = set(), {}
    for h in cfg.H_grid:
        seeds[str(float(h))] = {}
        for role in ('train', 'validation', 'test'):
            seed = int(_digest(['fresh-experiment', cfg.seed, h, role])[:8], 16)
            while seed in used:
                seed = (seed + 1) % 2**32
            used.add(seed)
            seeds[str(float(h))][role] = seed
    manifest = {'schema_version': 2, 'kind': 'fresh_grid', 'config': cfg.to_dict(),
                'execution': {'batch_size': batch_size, 'M_validation': M_validation}, 'seeds': seeds, 'versions': _versions(),
                'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'family_size': len(cfg.H_grid) * len(cfg.mu_grid)}
    path = directory / 'experiment.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Experiment parameters/environment changed: use a new RUN_ID')
    else:
        _atomic_json(path, manifest)
        _atomic_json(directory / 'config.json', cfg.to_dict())
    return manifest


def _cfg(manifest):
    raw = dict(manifest['config'])
    for name in ('H_grid', 'mu_grid'):
        raw[name] = tuple(raw[name])
    return Config(**raw)


def train_policy(B, mu, cfg, batch_size):
    """Same backward Ridge rule, retaining signatures only at exercise dates.

    Fine signatures are temporary per batch, not a 100000 x 201 x 31 tensor.
    """
    features = np.empty((len(B), cfg.n_exercise, cfg.sig_dim))
    payoff = B[:, cfg.exercise_index] + mu * cfg.exercise_times
    for start in range(0, len(B), batch_size):
        end = min(start + batch_size, len(B))
        X = adapter.drifted(B[start:end], mu, cfg)
        features[start:end] = adapter.signatures(X, cfg)[:, cfg.exercise_index, :]
    value = payoff[:, -1].copy()
    coefficients = np.empty((cfg.n_exercise - 1, cfg.sig_dim))
    intercepts = np.empty(cfg.n_exercise - 1)
    with adapter.silenced():
        for j in reversed(range(cfg.n_exercise - 1)):
            model = Ridge(alpha=cfg.ridge).fit(features[:, j, :], value)
            coefficients[j], intercepts[j] = model.coef_, model.intercept_
            stop = model.predict(features[:, j, :]) <= payoff[:, j]
            value[stop] = payoff[stop, j]
    return FixedPolicy(coefficients, intercepts, False)


def _moments_for_paths(policy, B, mu, cfg, batch_size):
    moments = Moments()
    for start in range(0, len(B), batch_size):
        moments.add(policy.rewards(B[start:start + batch_size], mu, cfg))
    return moments


def run_experiment(cfg, run_dir, *, batch_size=2_000, M_validation=10_000, progress=None):
    directory = Path(run_dir)
    manifest = prepare_experiment(directory, cfg, batch_size=batch_size, M_validation=M_validation)
    identity = _digest(manifest)
    for h in cfg.H_grid:
        pending = []
        for mu in cfg.mu_grid:
            key = _key(h, mu)
            path = directory / 'states' / f'{key}.json'
            state = json.loads(path.read_text()) if path.exists() else None
            if state is not None and state['identity'] != identity:
                raise ValueError('Checkpoint belongs to a different experiment')
            if (state is None or state['moments']['n'] != cfg.M_test
                    or not (directory / 'nodes' / f'{key}.json').exists()):
                pending.append((mu, key, state))
        if not pending:
            continue
        B_train = B_validation = B_test = None
        seeds = manifest['seeds'][str(float(h))]
        for mu, key, state in pending:
            policy_path = directory / 'policies' / f'{key}.json'
            if policy_path.exists():
                artifact = json.loads(policy_path.read_text())
                if artifact['identity'] != identity:
                    raise ValueError('Policy belongs to a different experiment')
                policy = FixedPolicy.from_dict(artifact['policy'])
            else:
                if B_train is None:
                    B_train, _ = adapter.simulate_paths(float(h), cfg, seeds['train'])
                    B_validation, _ = adapter.simulate_paths(
                        float(h), cfg.replace(M_train=M_validation), seeds['validation'])
                if progress:
                    progress({'stage': 'training', 'H': h, 'mu': mu})
                policy = train_policy(B_train, mu, cfg, batch_size)
                validation = _moments_for_paths(policy, B_validation, mu, cfg, batch_size)
                policy.stopped_at_zero = validation.mean <= 0
                artifact = {'identity': identity, 'H': h, 'mu': mu, 'policy': policy.to_dict(),
                            'validation_mean': validation.mean}
                _atomic_json(policy_path, artifact)
            moments = Moments(**state['moments']) if state else Moments()
            if moments.n > cfg.M_test or (moments.n != cfg.M_test and moments.n % batch_size):
                raise ValueError('Invalid test checkpoint')
            if B_test is None and not policy.stopped_at_zero:
                B_test, _ = adapter.simulate_paths(
                    float(h), cfg.replace(M_train=cfg.M_test), seeds['test'])
            for start in range(moments.n, cfg.M_test, batch_size):
                end = min(start + batch_size, cfg.M_test)
                rewards = (np.zeros(end-start) if policy.stopped_at_zero else
                           policy.rewards(B_test[start:end], mu, cfg))
                moments.add(rewards)
                _atomic_json(directory / 'states' / f'{key}.json',
                             {'identity': identity, 'H': h, 'mu': mu, 'moments': asdict(moments),
                              'stopped_at_zero': policy.stopped_at_zero,
                              'validation_mean': artifact['validation_mean']})
                if progress:
                    progress({'stage': 'test', 'H': h, 'mu': mu, 'n': moments.n})
            record = _record(h, mu, moments, policy.stopped_at_zero,
                             artifact['validation_mean'], cfg, manifest)
            _atomic_json(directory / 'nodes' / f'{key}.json', record)
            if progress:
                progress({'stage': 'node', **record})
        del B_train, B_validation, B_test
    return load_experiment(directory, epsilon=None)


def _record(h, mu, moments, stop0, validation, cfg, manifest, confidence=None):
    statistics = {'n': moments.n, 'V': moments.mean,
                  'SE': float(np.sqrt(max(0.0, moments.m2) / (moments.n - 1) / moments.n))}
    if confidence is not None:
        statistics = moments.intervals(confidence, manifest['family_size'])
    row = {'H': h, 'mu': mu, **statistics,
           'complete': moments.n == cfg.M_test, 'stopped_at_zero': stop0, 'validation_mean': validation}
    row['excess'] = row['V'] - mu * cfg.T
    for name in ('ci_low', 'ci_high', 'band_low', 'band_high'):
        if name in row:
            row['excess_' + name] = row[name] - mu * cfg.T
    return row


def load_experiment(run_dir, epsilon=0.03, *, confidence=0.95):
    """Analyze saved moments; epsilon=None loads raw estimates without intervals."""
    directory = Path(run_dir)
    manifest = json.loads((directory / 'experiment.json').read_text())
    cfg = _cfg(manifest)
    threshold = epsilon
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1')
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('epsilon must be positive and finite')
    identity = _digest(manifest)
    rows = []
    for path in (directory / 'states').glob('*.json'):
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        moments = Moments(**state['moments'])
        if moments.n >= 2:
            rows.append(_record(state['H'], state['mu'], moments, state['stopped_at_zero'],
                                state['validation_mean'], cfg, manifest,
                                confidence if threshold is not None else None))
    columns = ['H', 'mu', 'n', 'V', 'SE', 'ci_low', 'ci_high', 'band_low', 'band_high',
               'complete', 'stopped_at_zero', 'validation_mean', 'excess',
               'excess_ci_low', 'excess_ci_high', 'excess_band_low', 'excess_band_high']
    if threshold is None:
        columns = ['H', 'mu', 'n', 'V', 'SE', 'complete', 'stopped_at_zero',
                   'validation_mean', 'excess']
    nodes = pd.DataFrame(rows, columns=columns).sort_values(['H', 'mu']).reset_index(drop=True)
    if threshold is None:
        return manifest, nodes, pd.DataFrame()
    boundaries = []
    for h in cfg.H_grid:
        view = nodes[nodes.H.eq(h)].sort_values('mu')
        for side in (1, 2):
            row = {'H': h, 'side': side, 'epsilon': threshold, 'old_mu': None,
                   'estimate': None, 'mu_low': None, 'mu_high': None,
                   'window_low': min(cfg.mu_grid), 'window_high': max(cfg.mu_grid),
                   'status': 'incomplete_test'}
            if len(view) == len(cfg.mu_grid) and view.complete.all() and len(view) >= 2:
                prefix = '' if side == 1 else 'excess_'
                row.update(invert_band(view.mu, view['V' if side == 1 else 'excess'],
                                       view[prefix+'band_low'], view[prefix+'band_high'], threshold, side))
            boundaries.append(row)
    return manifest, nodes, pd.DataFrame(boundaries)
