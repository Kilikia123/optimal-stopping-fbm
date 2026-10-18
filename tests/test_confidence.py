from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from osfbm import adapter
from osfbm.config import Config
from osfbm.results import run_grid
from osfbm.confidence import (AuditConfig, Moments, FixedPolicy, prepare_audit,
                              run_audit, load_audit, invert_band)


class IntervalTests(unittest.TestCase):
    def test_moments_and_bonferroni(self):
        y = np.random.default_rng(42).normal(3, 2, 123)
        a = Moments()
        for block in np.array_split(y, 7):
            a.add(block)
        b = Moments(); b.add(y)
        self.assertAlmostEqual(a.mean, y.mean())
        self.assertAlmostEqual(a.m2, np.var(y, ddof=1) * (len(y) - 1))
        self.assertAlmostEqual(a.m2, b.m2)
        ci = a.intervals(.95, 20)
        self.assertAlmostEqual(ci['SE'], np.std(y, ddof=1) / np.sqrt(len(y)))
        self.assertAlmostEqual(ci['band_high'] - ci['V'], t.isf(.05 / 40, 122) * ci['SE'])
        self.assertGreater(ci['band_high'], ci['ci_high'])

    def test_exact_zero_policy(self):
        p = FixedPolicy(np.zeros((1, 3)), np.zeros(1), True)
        with patch.object(adapter, 'signatures', side_effect=AssertionError('not needed')):
            rewards = p.rewards(np.ones((20, 5)), 4, Config(n_fine=4, n_exercise=2))
        m = Moments(); m.add(rewards)
        self.assertEqual(m.intervals(.95, 100)['band_high'], 0)

    def test_grid_inversion_and_threshold_equality(self):
        mu = [-2, -1, 0, 1]
        result = invert_band(mu, [0, .5, 1, 2], [0, .3, .8, 1.8], [0, .7, 1.2, 2.2], 1, 1)
        self.assertEqual((result['mu_low'], result['mu_high']), (-1, 0))
        # A band exactly touching epsilon belongs to the <= set.
        exact = invert_band(mu, [0, .5, 1, 2], [0, .5, 1, 2], [0, .5, 1, 2], 1, 1)
        self.assertEqual((exact['mu_low'], exact['mu_high']), (0, 0))
        right = invert_band(mu, [2, 1, .5, 0], [1.8, .8, .3, 0], [2.2, 1.2, .7, 0], 1, 2)
        self.assertEqual((right['mu_low'], right['mu_high']), (-1, 0))

    def test_missing_multiple_and_unbounded(self):
        cases = [([0, 0, 0], [0, 0, 0], [0, 0, 0], 'expand_window'),
                 ([0, 2, 0], [0, 2, 0], [0, 2, 0], 'multiple_transitions'),
                 ([2, 2, 2], [2, 2, 2], [2, 2, 2], 'expand_window'),
                 ([0, 1, 2], [-2, -1, 0], [2, 3, 4], 'expand_window')]
        for mean, low, high, status in cases:
            r = invert_band([-1, 0, 1], mean, low, high, 1, 1)
            self.assertEqual(r['status'], status)
            self.assertIsNone(r['mu_low'])

    def test_brownian_fixed_hold(self):
        cfg = Config(n_fine=4, n_exercise=2, M_train=4000, K=1)
        B, _ = adapter.simulate_paths(.5, cfg, 827)
        p = FixedPolicy(np.zeros((1, cfg.sig_dim)), np.array([1e8]), False)
        rewards = p.rewards(B, .3, cfg)
        np.testing.assert_allclose(rewards, .3 + B[:, -1])
        self.assertLess(abs(rewards.mean() - .3), 4 * rewards.std() / np.sqrt(len(B)))


class AuditWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source'
        self.cfg = Config(n_fine=4, n_exercise=2, M_train=128, M_test=160, K=1,
                          mu_grid=tuple(np.arange(-2., 2.01, .25)), H_grid=(.5,))
        run_grid(self.cfg, self.source)
        self.settings = AuditConfig(epsilon=.3, n_test=240, batch_size=80, window_radius=.5)

    def test_replay_resume_and_readonly_analysis(self):
        a, b = Path(self.tmp.name) / 'a', Path(self.tmp.name) / 'b'
        manifest = prepare_audit(self.source, a, self.settings)
        self.assertGreater(manifest['family_size'], 0)
        seeds = [s for values in manifest['seeds'].values() for s in values]
        self.assertEqual(len(seeds), len(set(seeds)))
        with self.assertRaisesRegex(RuntimeError, 'interrupt'):
            def interrupt(event):
                if event['stage'] == 'test': raise RuntimeError('interrupt')
            run_audit(a, interrupt)
        partial = load_audit(a)
        self.assertTrue(partial[2].status.eq('incomplete_test').any())
        _, first, boundaries = run_audit(a)
        self.assertTrue(first.complete.all())
        prepare_audit(self.source, b, self.settings)
        _, second, _ = run_audit(b)
        pd.testing.assert_frame_equal(first, second)
        with patch.object(adapter, 'simulate_paths', side_effect=AssertionError('simulation')):
            _, read, _ = load_audit(a)
            _, repeat, _ = run_audit(a)
        pd.testing.assert_frame_equal(first, read)
        pd.testing.assert_frame_equal(first, repeat)
        np.testing.assert_allclose(first.excess_ci_low, first.ci_low - first.mu * self.cfg.T)
        self.assertEqual(set(first.n), {240})
        # Configuration is immutable after preparation.
        with self.assertRaises(ValueError):
            prepare_audit(self.source, a, AuditConfig(n_test=400))

    def test_reproduction_failure(self):
        directory = Path(self.tmp.name) / 'bad'
        prepare_audit(self.source, directory, self.settings)
        with patch.object(adapter, 'longstaff_schwartz', return_value=(12345., 1., [])):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                run_audit(directory)
        self.assertFalse((directory / 'states').exists())

    def test_incomplete_source_and_no_nodes(self):
        source = pd.read_csv(self.source / 'nodes.csv')
        source.iloc[:1].to_csv(self.source / 'nodes.csv', index=False)
        directory = Path(self.tmp.name) / 'partial'
        m = prepare_audit(self.source, directory, self.settings)
        self.assertEqual(m['family_size'], 0)
        _, values, bounds = run_audit(directory)
        self.assertTrue(values.empty)
        self.assertTrue(bounds.status.eq('source_incomplete').all())


if __name__ == '__main__':
    unittest.main()
