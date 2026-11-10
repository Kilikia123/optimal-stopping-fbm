"""Relative-level inversion, shape feasibility, and analysis-only integration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from boundary_analysis.monotone import fit_monotone_boundaries
from boundary_analysis.relative import relative_boundaries
from config import Config
from experiments.statistics import load_experiment_statistics, load_experiment
from experiments.training import run_experiment


def synthetic(se=0.):
    rows = []
    for h, c in ((.3, .5), (.4, .25), (.5, 0.), (.6, .25), (.7, .5)):
        for mu in np.round(np.arange(-2, 2.01, .1), 8):
            rows.append(dict(H=h, mu=mu, V=max(0., .5*mu+c, mu), SE=se, n=10000, complete=True))
    return pd.DataFrame(rows)


class AdaptiveTests(unittest.TestCase):
    def test_analytic_crossings_and_shape(self):
        table = relative_boundaries(synthetic(), T=1)
        for _, row in table.iterrows():
            c = .5 if row.H in (.3, .7) else .25 if row.H != .5 else 0
            self.assertAlmostEqual(row.epsilon, .2*c)
            for side, exact in ((1, -1.6*c), (2, 1.6*c)):
                self.assertLessEqual(row[f'mu{side}_low'], exact+1e-12)
                self.assertGreaterEqual(row[f'mu{side}_high'], exact-1e-12)
                self.assertLessEqual(abs(row[f'mu{side}_raw']-exact), .100001)
        fit = fit_monotone_boundaries(table)
        self.assertTrue((fit.mu1_fit <= 1e-7).all())
        self.assertTrue((fit.mu2_fit >= -1e-7).all())
        for side in (1, 2):
            for left in (True, False):
                v = fit.loc[fit.H.le(.5) if left else fit.H.ge(.5), f'mu{side}_fit']
                direction = 1 if left == (side == 1) else -1
                self.assertTrue((direction*np.diff(v) >= -1e-7).all())

    def test_uncertain_reference_not_silently_clipped(self):
        table = relative_boundaries(synthetic(se=1), T=1)
        unresolved = table[table.H.ne(.5)]
        self.assertTrue(unresolved.mu1_status.eq('insufficient_precision').all())
        self.assertTrue(unresolved.mu1_raw.notna().all())
        strict = fit_monotone_boundaries(table, include_unconfirmed=False)
        self.assertTrue(strict.loc[strict.H.ne(.5), 'mu1_fit'].isna().all())
        exploratory = fit_monotone_boundaries(table)
        self.assertTrue(exploratory.loc[exploratory.H.ne(.5), 'mu1_fit_status'].eq('fitted_unconfirmed').all())
        self.assertTrue(exploratory.loc[exploratory.H.ne(.5), 'mu1_low'].isna().all())
        anchor = table[table.H.eq(.5)].iloc[0]
        self.assertEqual((anchor.mu1_raw, anchor.mu2_raw, anchor.epsilon), (0, 0, 0))

    def test_reference_uncertainty_widens_intervals(self):
        exact = relative_boundaries(synthetic(), T=1)
        noisy = relative_boundaries(synthetic(se=.005), T=1)
        for side in (1, 2):
            self.assertTrue((noisy[f'mu{side}_low'] <= exact[f'mu{side}_low']).all())
            self.assertTrue((noisy[f'mu{side}_high'] >= exact[f'mu{side}_high']).all())

    def test_missing_and_partial_grids(self):
        data = synthetic()
        without = data[data.mu.ne(0)]
        table = relative_boundaries(without, T=1)
        self.assertTrue(table.loc[table.H.ne(.5), 'mu1_status'].eq('missing_zero').all())
        data.attrs.update(expected_H=[.3,.4,.5,.6,.7,.8], expected_mu=sorted(data.mu.unique()))
        data = data.drop(data.index[0])
        table = relative_boundaries(data, T=1).set_index('H')
        self.assertEqual(table.loc[.3, 'mu1_status'], 'incomplete_test')
        self.assertEqual(table.loc[.8, 'mu1_status'], 'incomplete_test')

    def test_multiple_transitions_and_window(self):
        data = synthetic()
        data.loc[data.H.eq(.3) & data.mu.eq(-1.5), 'V'] = 10
        table = relative_boundaries(data, T=1).set_index('H')
        self.assertEqual(table.loc[.3, 'mu1_status'], 'multiple_transitions')
        small = synthetic().query('-.2 <= mu <= .2')
        table = relative_boundaries(small, T=1)
        self.assertTrue(table.loc[table.H.ne(.5), 'mu2_status'].eq('expand_window').all())

    def test_infeasible_branch_is_not_drawn(self):
        table = relative_boundaries(synthetic(), T=1)
        table.loc[table.H.eq(.3), ['mu1_raw','mu1_low','mu1_high']] = [-.1,-.1,-.1]
        table.loc[table.H.eq(.4), ['mu1_raw','mu1_low','mu1_high']] = [-1.,-1.,-1.]
        fit = fit_monotone_boundaries(table)
        self.assertTrue(fit.loc[fit.H.lt(.5), 'mu1_fit_status'].eq('incompatible_shape').all())
        self.assertTrue(fit.loc[fit.H.lt(.5), 'mu1_fit'].isna().all())
        self.assertTrue(fit.loc[fit.H.gt(.5), 'mu1_fit'].notna().all())
        self.assertEqual(fit.loc[fit.H.eq(.3), 'mu1_raw'].iloc[0], -.1)

    def test_gaps_remain_and_asymmetry_is_allowed(self):
        table = relative_boundaries(synthetic(), T=1)
        table.loc[table.H.eq(.4), 'mu1_status'] = 'incomplete_test'
        table.loc[table.H.eq(.4), 'mu1_raw'] = np.nan
        fit = fit_monotone_boundaries(table)
        self.assertTrue(fit.loc[fit.H.eq(.4), 'mu1_fit'].isna().all())
        self.assertTrue(fit.loc[fit.H.eq(.6), 'mu1_fit'].notna().all())

    def test_edge_extremum_is_not_a_point_boundary(self):
        table = relative_boundaries(synthetic().query('-.2 <= mu <= .2'), T=1)
        fit = fit_monotone_boundaries(table)
        for side in (1, 2):
            self.assertTrue(fit.loc[fit.H.ne(.5), f'mu{side}_raw'].isna().all())
            self.assertTrue(fit.loc[fit.H.ne(.5), f'mu{side}_fit'].isna().all())

    def test_main_plot_has_only_one_value_per_boundary(self):
        from visualization.relative import plot_relative_boundaries
        import matplotlib.pyplot as plt
        table = relative_boundaries(synthetic(), T=1)
        table.loc[table.H.eq(.3), ['mu2_raw','mu2_low','mu2_high']] = [.2, .1, 1.]
        fit = fit_monotone_boundaries(table)
        fig = plot_relative_boundaries(fit, alpha=.2)
        # Two disjoint marker collections per boundary: supported and unconfirmed.
        for side in (1, 2):
            offsets = np.concatenate([c.get_offsets() for c in fig.axes[0].collections[2*(side-1):2*side]])
            self.assertEqual(len(offsets), len(fit))
            np.testing.assert_allclose(offsets[:, 1], fit[f'mu{side}_fit'])
        plt.close(fig)

    def test_invalid_parameters(self):
        for kw in ({'alpha':0}, {'alpha':1}, {'alpha':np.nan}, {'confidence':1}, {'T':0}):
            with self.assertRaises(ValueError):
                relative_boundaries(synthetic(), **({'T':1} | kw))
        with self.assertRaises(ValueError):
            relative_boundaries(pd.concat([synthetic(), synthetic()]), T=1)

    def test_saved_statistics_and_artifacts_without_training(self):
        from boundary_analysis.reporting import analyze_relative_run
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(n_fine=4, n_exercise=2, M_train=32, M_test=64, K=1,
                         H_grid=(.3,.5), mu_grid=(-1.,0.,1.))
            run_experiment(cfg, tmp, batch_size=32, M_validation=32)
            before = {p: p.read_bytes() for p in Path(tmp).rglob('*.json')}
            with patch('adapter.simulate_paths', side_effect=AssertionError('simulation')),\
                 patch('experiments.training.train_policy', side_effect=AssertionError('training')):
                _, nodes = load_experiment_statistics(tmp, confidence=.9)
                _, legacy, _ = load_experiment(tmp, epsilon=.1, confidence=.9)
                pd.testing.assert_frame_equal(nodes, legacy)
                _, bounds, _, output = analyze_relative_run(tmp, confidence=.9, sensitivity=(.1,))
            self.assertEqual(before, {p:p.read_bytes() for p in before})
            for filename in ('boundaries.csv','diagnostics.csv','metadata.json','sensitivity.csv',
                             'boundaries.png','epsilon.png','sensitivity.png','boundary_comparison.png'):
                self.assertTrue((output/filename).exists())
            self.assertEqual(bounds.loc[bounds.H.eq(.5),'mu1_fit'].iloc[0], 0)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
