"""Paired replay uses saved policies and the original test paths only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import adapter
from config import Config
from experiments.moments import Moments
from experiments.policy import FixedPolicy
from infrastructure.identity import node_key
from experiments.paired import paired_rewards, run_paired_evaluation
from experiments.statistics import load_experiment_statistics
from experiments.training import run_experiment
from boundary_analysis.relative import relative_boundaries


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cfg = Config(n_fine=4, n_exercise=2, M_train=32, M_test=64, K=1,
                          H_grid=(.3,.5), mu_grid=(-1.,0.,1.))

    def prepare(self):
        # Legacy sources exercise the separate replay path used by old runs.
        run_experiment(self.cfg, self.path, batch_size=16, M_validation=32, collect_paired=False)

    def test_hold_is_exactly_zero_and_stop_zero_uses_terminal_noise(self):
        cfg = self.cfg
        B, _ = adapter.simulate_paths(.3, cfg, 123)
        hold = FixedPolicy(np.zeros((1,cfg.sig_dim)), np.array([1e9]), False)
        reward, gains = paired_rewards(hold, B, .2, cfg)
        np.testing.assert_array_equal(gains, np.zeros(len(B)))
        m = Moments(); m.add(gains)
        self.assertEqual(m.intervals(.95,100)['SE'],0)
        hold.stopped_at_zero = True
        reward, gains = paired_rewards(hold, B, .2, cfg)
        np.testing.assert_array_equal(reward, np.zeros(len(B)))
        np.testing.assert_array_equal(gains, -B[:,-1]-.2*cfg.T)
        self.assertGreater(np.std(gains),0)

    def test_replay_resume_is_identical_and_does_not_train(self):
        self.prepare()
        before = {p:p.read_bytes() for p in self.path.rglob('*.json')}
        def interrupt(e):
            if e['stage']=='test': raise RuntimeError('stop')
        with patch('experiments.training.train_policy',side_effect=AssertionError('training')):
            with self.assertRaisesRegex(RuntimeError,'stop'):
                run_paired_evaluation(self.path,batch_size=13,progress=interrupt)
            run_paired_evaluation(self.path,batch_size=17)
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        manifest, paired = load_experiment_statistics(self.path,gain_mode='paired')
        self.assertTrue(paired.paired_complete.all())
        _, legacy = load_experiment_statistics(self.path)
        pd.testing.assert_frame_equal(legacy[['V','SE','n']],paired[['V','SE','n']])
        # Compare each pair's moments to direct evaluation on precisely the saved seed.
        for h in self.cfg.H_grid:
            B,_=adapter.simulate_paths(h,self.cfg.replace(M_train=self.cfg.M_test),
                                       manifest['seeds'][str(float(h))]['test'])
            for mu in self.cfg.mu_grid:
                artifact=json.loads((self.path/'policies'/f'{node_key(h,mu)}.json').read_text())
                _, gains=paired_rewards(FixedPolicy.from_dict(artifact['policy']),B,mu,self.cfg)
                row=paired[paired.H.eq(h)&paired.mu.eq(mu)].iloc[0]
                self.assertAlmostEqual(row.paired_gain,float(gains.mean()),places=12)
                self.assertAlmostEqual(row.paired_SE,float(gains.std(ddof=1)/np.sqrt(len(gains))),places=12)
        with patch('adapter.simulate_paths',side_effect=AssertionError('simulation')):
            run_paired_evaluation(self.path)
            load_experiment_statistics(self.path,gain_mode='paired')
            from boundary_analysis.reporting import analyze_relative_run
            import matplotlib.pyplot as plt
            _, _, _, output = analyze_relative_run(self.path, gain_mode='paired', sensitivity=())
            self.assertIn('relative_paired', output.parts)
            metadata = json.loads((output/'metadata.json').read_text())
            self.assertEqual(metadata['gain_mode'],'paired')
            self.assertTrue(metadata['paired_identity'])
            plt.close('all')

    def test_missing_partial_and_changed_sources_are_not_silently_used(self):
        self.prepare()
        with self.assertRaisesRegex(FileNotFoundError,'notebook 03'):
            load_experiment_statistics(self.path,gain_mode='paired')
        run_paired_evaluation(self.path,H_values=[.5])
        _, nodes=load_experiment_statistics(self.path,gain_mode='paired')
        self.assertFalse(nodes.loc[nodes.H.eq(.3),'paired_complete'].any())
        file=self.path/'policies'/f'{node_key(.5,0.)}.json'
        d=json.loads(file.read_text()); d['validation_mean']+=1; file.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'stale'):
            load_experiment_statistics(self.path,gain_mode='paired')
        with self.assertRaisesRegex(ValueError,'source policy/test'):
            run_paired_evaluation(self.path,H_values=[.5])

    def test_replay_disagreement_is_not_marked_verified(self):
        self.prepare()
        original=adapter.simulate_paths
        def changed(*args,**kwargs):
            b,w=original(*args,**kwargs)
            return b+10,w
        with patch('adapter.simulate_paths',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'differs'):
                run_paired_evaluation(self.path,H_values=[.3])
        for p in (self.path/'paired_evaluation'/'states').glob('*.json'):
            d=json.loads(p.read_text())
            # Some policies can stop at zero, whose rewards still reproduce.
            if d['mu']==1.:
                self.assertFalse(d['verified'])

    def test_right_boundary_uses_paired_mean_and_se_left_is_unchanged(self):
        mu=np.array([-2.,-1.,0.,1.,2.])
        nodes=pd.DataFrame(dict(H=.3,mu=mu,V=np.maximum(0,mu)+.1,SE=.001,n=10000,complete=True,
                                paired_gain=[2.,1.,.1,0.,0.],paired_SE=.0,paired_n=10000,paired_complete=True))
        old=relative_boundaries(nodes,T=1)
        new=relative_boundaries(nodes,T=1,gain_mode='paired')
        self.assertTrue(np.isnan(old.mu2_raw.iloc[0]))
        self.assertEqual(new.mu2_raw.iloc[0],1.)
        for key in ['raw','low','high','status']:
            pd.testing.assert_series_equal(old[f'mu1_{key}'],new[f'mu1_{key}'])
        nodes.loc[nodes.mu.eq(1),'paired_complete']=False
        partial=relative_boundaries(nodes,T=1,gain_mode='paired')
        self.assertEqual(partial.mu2_status.iloc[0],'incomplete_paired_test')
        self.assertTrue(np.isnan(partial.mu2_raw.iloc[0]))


if __name__=='__main__': unittest.main()
