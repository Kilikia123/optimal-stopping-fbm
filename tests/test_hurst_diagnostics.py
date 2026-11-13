import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market.hurst_diagnostics import load_hurst_blocks, summarize_hurst_performance, save_hurst_tables
from visualization.hurst import plot_hurst_performance


def fixture():
    return pd.DataFrame(dict(
        H=[.30, .42, .46, .49, .51, .54, .58, .70],
        decision_idx=np.arange(8) * 21, end_idx=np.arange(1, 9) * 21,
        r_buy_hold=np.full(8, .01),
        r_model=[.00, .02, .01, -.01, -.02, .03, .02, .00],
        r_policy=[.01, .03, .00, .00, -.01, .02, .01, .01],
    ))


class HurstDiagnosticTests(unittest.TestCase):
    def test_conditional_intervals_use_joint_chronology(self):
        rows = fixture()
        before = rows.copy()
        idx = np.tile(np.arange(8), (100, 1))
        idx[::2] = [0, 0, 1, 3, 4, 5, 6, 7]
        with patch('market.hurst_diagnostics.stationary_bootstrap_indices', return_value=idx) as boot:
            tables = summarize_hurst_performance(rows, n_boot=100)
        self.assertEqual(boot.call_args.args[:2], (8, 100))
        model = tables['groups'].query("family == 'distance' and strategy == 'model'")
        self.assertEqual(model.n.sum(), 8)
        self.assertAlmostEqual(model.gap_contribution.sum(), (rows.r_model - rows.r_buy_hold).mean())
        near = np.abs(rows.H.to_numpy() - .5) < .025
        sampled_mask = near[idx]
        delta = (rows.r_model - rows.r_buy_hold).to_numpy()[idx]
        means = np.where(sampled_mask, delta, 0).sum(1) / sampled_mask.sum(1)
        self.assertAlmostEqual(model.iloc[0].boot_low, np.quantile(means, .025))
        self.assertAlmostEqual(model.iloc[0].boot_high, np.quantile(means, .975))
        pd.testing.assert_frame_equal(before, rows)

    def test_only_complete_strategy_and_direct_contrast(self):
        rows = fixture()
        idx = np.tile(np.arange(8), (100, 1))
        with patch('market.hurst_diagnostics.stationary_bootstrap_indices', return_value=idx):
            tables = summarize_hurst_performance(rows, contrast_widths=(.05,), n_boot=100)
        self.assertEqual(set(tables), {'groups', 'contrasts'})
        self.assertEqual(set(tables['groups'].strategy), {'model'})
        delta = (rows.r_model - rows.r_buy_hold).to_numpy()
        far = np.abs(rows.H.to_numpy() - .5) >= .05
        contrast = tables['contrasts'].query("strategy == 'model'").iloc[0]
        expected = delta[far].mean() - delta[~far].mean()
        self.assertAlmostEqual(contrast.far_minus_near, expected)
        self.assertAlmostEqual(contrast.boot_low, expected)

    def test_missing_policy_empty_groups_and_plot(self):
        rows = fixture().drop(columns='r_policy')
        rows['H'] = .49
        tables = summarize_hurst_performance(rows, n_boot=100)
        self.assertEqual(set(tables['groups'].strategy), {'model'})
        empty = tables['groups'].query('n == 0')
        self.assertTrue(empty.boot_low.isna().all())
        self.assertTrue(empty.status.eq('insufficient_data').all())
        self.assertTrue(tables['contrasts'].status.eq('insufficient_data').all())
        fig = plot_hurst_performance(rows, tables)
        self.assertEqual(len(fig.axes), 3)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_rejects_filtered_chronology_and_invalid_settings(self):
        with self.assertRaisesRegex(ValueError, 'chronology'):
            summarize_hurst_performance(fixture().iloc[[0, 2, 4]], n_boot=100)
        for kwargs in (dict(distance_edges=(0, .1, .05, np.inf)),
                       dict(contrast_widths=(.1, .05)), dict(n_boot=2), dict(mean_block=0)):
            with self.assertRaises(ValueError):
                summarize_hurst_performance(fixture(), **kwargs)

    def test_loads_only_original_backtest_and_ignores_unrelated_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = output / 'blocks.csv'
            fixture().drop(columns='r_policy').to_csv(source, index=False)
            rows = load_hurst_blocks(output)
            self.assertNotIn('r_policy', rows)
            audit = output / 'action_audit'
            audit.mkdir()
            fixture().to_csv(audit / 'blocks.csv', index=False)
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            meta = dict(inputs=dict(source_backtest_hash=sha(source)),
                        outputs=dict(blocks=sha(audit / 'blocks.csv')))
            (audit / 'metadata.json').write_text(json.dumps(meta))
            rows = load_hurst_blocks(output)
            self.assertNotIn('r_policy', rows)
            tables = summarize_hurst_performance(rows, n_boot=100)
            folder = save_hurst_tables(output, tables, settings=dict(n_boot=100))
            self.assertTrue((folder / 'contrasts.csv').exists())
            self.assertEqual(json.loads((folder / 'metadata.json').read_text())['backtest_hash'], sha(source))
            source.write_text(source.read_text() + '\n')
            pd.testing.assert_frame_equal(load_hurst_blocks(output), rows)


if __name__ == '__main__':
    unittest.main()
