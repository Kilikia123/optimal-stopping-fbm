"""Read-only loading of 04 exports for empirical notebooks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from infrastructure.identity import digest
from boundary_analysis.curves import Boundaries, CASH, HOLD, POLICY


class AnalysisLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {'config': {'H_grid': [.3,.4,.5,.6,.7]}}
        (self.root/'experiment.json').write_text(json.dumps(self.manifest))

    def export(self, gain='paired', fixed=False):
        branch = '' if fixed else ('relative_paired' if gain=='paired' else 'relative')
        out = self.root/'analysis'/branch/('epsilon_0.06' if fixed else 'alpha_0.2')/'confidence_0.95'
        out.mkdir(parents=True)
        f = pd.DataFrame(dict(H=[.3,.4,.5,.6,.7], mu1=[-1,np.nan,0,-.2,-.4], mu2=[1,np.nan,0,.2,.4]))
        if not fixed:
            f = f.rename(columns={'mu1':'mu1_fit','mu2':'mu2_fit'})
            f['mu1_raw'] = 99.; f['mu2_raw'] = 99.
            f['epsilon'] = [.1,.05,0,.03,.04]
            for s in (1,2):
                f[f'mu{s}_fit_status'] = ['fitted','unavailable','analytic','fitted_unconfirmed','fitted']
            (out/'metadata.json').write_text(json.dumps(dict(experiment_identity=digest(self.manifest),
                                                        gain_mode=gain, alpha=.2, confidence=.95)))
        f.to_csv(out/'boundaries.csv', index=False)
        return out

    def test_modes_and_interpolation_use_fit_not_raw(self):
        for gain in ('legacy','paired'):
            self.export(gain)
            with patch('adapter.simulate_paths', side_effect=AssertionError('simulation')),\
                 patch('experiments.training.train_policy', side_effect=AssertionError('training')):
                b = Boundaries.from_analysis(self.root,gain_mode=gain)
            self.assertEqual(b.gain_mode,gain)
            self.assertIsNone(b.epsilon)
            self.assertAlmostEqual(float(b.mu1_at(.4)),-.5)
            self.assertAlmostEqual(float(b.mu2_at(.4)),.5)
            self.assertAlmostEqual(float(b.epsilon_at(.65)),.035)
            self.assertEqual(b.regime(.4,-.6),CASH)
            self.assertEqual(b.regime(.4,.6),HOLD)
            self.assertEqual(b.regime(.4,0),POLICY)
            d = b.diagnostics()
            self.assertEqual(d.unconfirmed.tolist(),[1,1])
            self.assertEqual(d.iloc[0].interpolated_gaps,[(.3,.5)])
            self.assertAlmostEqual(float(b.mu2_at(.9)),.4)  # existing clamping rule

    def test_fixed_remains_compatible(self):
        out=self.export(fixed=True)
        b=Boundaries.from_analysis(self.root,boundary_mode='fixed')
        old=Boundaries.from_csv(out/'boundaries.csv',.06)
        np.testing.assert_equal(b.frame().to_numpy(),old.frame().to_numpy())
        self.assertEqual(b.epsilon_at(.3),.06)
        self.assertEqual(b.mode,'fixed')
        self.assertEqual(b.diagnostics().unconfirmed.tolist(),[0,0])

    def test_missing_and_stale_metadata(self):
        with self.assertRaisesRegex(FileNotFoundError,'analyze_relative_run'):
            Boundaries.from_analysis(self.root)
        out=self.export()
        p=out/'metadata.json'; m=json.loads(p.read_text()); m['experiment_identity']='stale'
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'metadata'):
            Boundaries.from_analysis(self.root)

    def test_reject_status_and_do_not_extrapolate_missing_edges(self):
        out=self.export(); p=out/'boundaries.csv'; f=pd.read_csv(p)
        f.loc[0,'mu2_fit_status']='unavailable'; f.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'status'):
            Boundaries.from_analysis(self.root)
        f.loc[0,'mu2_fit']=np.nan; f.to_csv(p,index=False)
        b=Boundaries.from_analysis(self.root)
        self.assertTrue(np.isinf(b.mu2_at(.3)))


if __name__=='__main__': unittest.main()
