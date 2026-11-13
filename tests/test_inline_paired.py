"""New experiments collect paired moments in the original test pass."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import adapter
from config import Config
from experiments.policy import FixedPolicy
from infrastructure.identity import node_key
from experiments.paired import run_paired_evaluation, paired_rewards
from experiments.statistics import load_experiment_statistics
from experiments.training import run_experiment, train_policy


class InlinePairedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = Config(n_fine=4,n_exercise=2,M_train=32,M_test=64,K=1,
                          H_grid=(.3,),mu_grid=(-1.,0.,1.))
        self.execution = dict(batch_size=16,M_validation=32)

    def test_single_pass_matches_direct_pairs_and_legacy_rewards(self):
        manifest, _, _ = run_experiment(self.cfg,self.root,**self.execution)
        _, inline = load_experiment_statistics(self.root,gain_mode='paired')
        self.assertTrue(inline.paired_complete.all())
        self.assertFalse((self.root/'paired_evaluation').exists())
        B,_ = adapter.simulate_paths(.3,self.cfg.replace(M_train=self.cfg.M_test),
                                      manifest['seeds']['0.3']['test'])
        for _,row in inline.iterrows():
            p=json.loads((self.root/'policies'/f'{node_key(.3,row.mu)}.json').read_text())
            reward,gain=paired_rewards(FixedPolicy.from_dict(p['policy']),B,row.mu,self.cfg)
            self.assertAlmostEqual(row.V,reward.mean(),places=12)
            self.assertAlmostEqual(row.paired_gain,gain.mean(),places=12)
            self.assertAlmostEqual(row.paired_SE,gain.std(ddof=1)/np.sqrt(len(gain)),places=12)
        with patch('adapter.simulate_paths',side_effect=AssertionError('replay')):
            run_experiment(self.cfg,self.root,**self.execution)
            run_paired_evaluation(self.root)
            _,again=load_experiment_statistics(self.root,gain_mode='paired')
        pd.testing.assert_frame_equal(inline,again)
        legacy=self.root/'legacy'
        run_experiment(self.cfg,legacy,collect_paired=False,**self.execution)
        _,old=load_experiment_statistics(legacy)
        pd.testing.assert_frame_equal(old[['V','SE']],inline[['V','SE']])

    def test_atomic_resume_and_partial_statistics(self):
        def interrupt(e):
            if e['stage']=='test': raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            run_experiment(self.cfg,self.root,progress=interrupt,**self.execution)
        _,partial=load_experiment_statistics(self.root,gain_mode='paired')
        self.assertFalse(partial.paired_complete.any())
        for p in (self.root/'states').glob('*.json'):
            state=json.loads(p.read_text())
            self.assertEqual(state['moments']['n'],state['paired_moments']['n'])
        with patch('experiments.training.train_policy',wraps=train_policy) as trained:
            run_experiment(self.cfg,self.root,**self.execution)
        self.assertEqual(trained.call_count,2)  # the interrupted policy is reused
        other=self.root/'uninterrupted'
        run_experiment(self.cfg,other,**self.execution)
        _,a=load_experiment_statistics(self.root,gain_mode='paired')
        _,b=load_experiment_statistics(other,gain_mode='paired')
        pd.testing.assert_frame_equal(a,b)

    def test_stop_at_zero_still_generates_hold_rewards(self):
        cfg=self.cfg.replace(mu_grid=(-100.,))
        run_experiment(cfg,self.root,**self.execution)
        _,f=load_experiment_statistics(self.root,gain_mode='paired')
        self.assertEqual(f.V.iloc[0],0.)
        self.assertGreater(f.paired_SE.iloc[0],0.)
        self.assertGreater(f.paired_gain.iloc[0],90.)

    def test_hold_policy_has_exact_zero_gain_in_original_test(self):
        cfg=self.cfg.replace(mu_grid=(10.,))  # validation must also choose continuation at t=0
        hold=FixedPolicy(np.zeros((1,cfg.sig_dim)),np.array([1e9]),False)
        with patch('experiments.training.train_policy',return_value=hold):
            run_experiment(cfg,self.root,**self.execution)
        _,f=load_experiment_statistics(self.root,gain_mode='paired')
        self.assertEqual(f.paired_gain.iloc[0],0.)
        self.assertEqual(f.paired_SE.iloc[0],0.)

    def test_legacy_partial_does_not_fabricate_paired_history(self):
        def interrupt(e):
            if e['stage']=='test': raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):
            run_experiment(self.cfg,self.root,collect_paired=False,progress=interrupt,**self.execution)
        run_experiment(self.cfg,self.root,**self.execution)
        _,f=load_experiment_statistics(self.root,gain_mode='paired')
        self.assertFalse(f.loc[f.mu.eq(-1.),'paired_complete'].iloc[0])
        self.assertTrue(f.loc[f.mu.ne(-1.),'paired_complete'].all())


if __name__=='__main__': unittest.main()
