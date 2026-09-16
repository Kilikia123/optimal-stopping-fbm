from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from osfbm import adapter
from osfbm.config import Config
from osfbm.experiment import ExperimentSettings, train_policy, run_experiment, load_experiment


class FreshExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(n_fine=4, n_exercise=2, M_train=128, M_test=240,
                          H_grid=(.3,.5), mu_grid=(-1.,0.,1.), K=2)
        self.settings = ExperimentSettings(epsilon=.1, batch_size=80, M_validation=96)

    def test_same_training_rule_as_vendor(self):
        cfg = self.cfg.replace(n_exercise=4)
        paths = adapter.simulate_pair(.3, cfg)
        X = adapter.drifted(paths['B_train'], .2, cfg)
        Y = adapter.drifted(paths['B_test'], .2, cfg)
        raw, _, regr = adapter.longstaff_schwartz(adapter.signatures(X,cfg), X,
                                                 adapter.signatures(Y,cfg), Y, cfg)
        p = train_policy(paths['B_train'], .2, cfg, 37)
        np.testing.assert_allclose(p.coefficients, [r.coef_ for r in regr], rtol=1e-7, atol=1e-8)
        self.assertAlmostEqual(p.rewards(paths['B_test'], .2, cfg).mean(), raw, places=8)

    def test_independent_roles_resume_and_intervals(self):
        first, second = Path(self.tmp.name)/'first',Path(self.tmp.name)/'second'
        def interrupt(e):
            if e['stage']=='test': raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            run_experiment(self.cfg, first, self.settings, interrupt)
        m, nodes, bounds = run_experiment(self.cfg, first, self.settings)
        _, other, _ = run_experiment(self.cfg, second, self.settings)
        pd.testing.assert_frame_equal(nodes,other)
        self.assertEqual(len(nodes), 6)
        self.assertTrue(nodes.complete.all())
        self.assertTrue((nodes.ci_low<=nodes.V).all())
        self.assertTrue((nodes.band_high>=nodes.ci_high).all())
        seeds=[s for roles in m['seeds'].values() for s in roles.values()]
        self.assertEqual(len(seeds),len(set(seeds)))
        with patch.object(adapter,'simulate_paths',side_effect=AssertionError('simulation')):
            _, saved, _ = load_experiment(first)
            run_experiment(self.cfg, first, self.settings)
        pd.testing.assert_frame_equal(nodes,saved)
        self.assertEqual(len(list((first/'nodes').glob('*.json'))),6)
        with self.assertRaises(ValueError):
            run_experiment(self.cfg.replace(M_train=256),first,self.settings)

    def test_test_seed_cannot_change_policy(self):
        # Policy fitting receives training paths only, independent of test size.
        B,_=adapter.simulate_paths(.5,self.cfg,19)
        p=train_policy(B,.1,self.cfg,32)
        q=train_policy(B,.1,self.cfg.replace(M_test=1000),32)
        np.testing.assert_array_equal(p.coefficients,q.coefficients)


if __name__=='__main__': unittest.main()
