"""Causality, synthetic scenarios, independent selection and frozen deployment."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market.backtest import decide_block
from config import Config
from estimation.hurst import HURST_METHODS
from estimation.pipeline import estimate
from estimation.simulation import simulate_log_prices
from estimation.types import EstimationWindows
from experiments.training import run_experiment
from experiments.policy_bank import PolicyBank
from boundary_analysis.preparation import complete_boundary_grid, boundary_object
from tuning.cache import EstimatorCache, DecisionCache
from tuning.config import TuningConfig, Candidate, candidates
from tuning.frozen import load_frozen_selection
from tuning.runner import run_synthetic_tuning
from tuning.scenarios import scenarios, simulation_seed, synthetic_market
from tuning.scoring import score_rows


def ideal_nodes():
    rows=[]
    for h in (.3,.5,.7):
        c=0 if h==.5 else .5
        for mu in np.round(np.arange(-2,2.01,.1),8):
            v=max(0,.5*mu+c,mu)
            rows.append(dict(H=h,mu=mu,V=v,SE=0.,n=10000,complete=True,
                             paired_gain=v-mu,paired_SE=0.,paired_n=10000,paired_complete=True))
    return pd.DataFrame(rows)


def full_grid():
    return pd.DataFrame(dict(H=[.3,.4,.5,.6,.7],mu1_fit=[-.5,np.nan,0,-.2,-.4],
                             mu2_fit=[.5,np.nan,0,.2,.4],epsilon=[.1,.05,0,.03,.04],
                             mu1_fit_status=['fitted','unavailable','analytic','fitted','fitted'],
                             mu2_fit_status=['fitted','unavailable','analytic','fitted','fitted']))


class SeparateWindowTests(unittest.TestCase):
    def test_candidates_stay_strictly_inside_declared_grid(self):
        config=TuningConfig()
        grid=candidates(config)
        expected=(len(config.H_windows)*len(config.drift_windows)*len(config.sigma_windows)
                  *len(config.methods)*len(config.alphas))
        self.assertEqual(len(grid),expected)
        self.assertEqual(len(grid),360)
        for c in grid:
            self.assertIn(c.H,config.H_windows)
            self.assertIn(c.drift,config.drift_windows)
            self.assertIn(c.sigma,config.sigma_windows)
        self.assertEqual(config.warmup,1008)

    def test_manual_components_and_common_window_compatibility(self):
        logp=simulate_log_prices(.01,.05,.5,160,seed=17)[0]
        w=EstimationWindows(96,21,63)
        e=estimate(logp,windows=w)
        r=np.diff(logp)
        self.assertAlmostEqual(e.a,21*r[-21:].mean())
        h=HURST_METHODS['rs'](r[-96:]-r[-96:].mean())
        self.assertAlmostEqual(e.H_raw,h)
        self.assertAlmostEqual(e.sigma,r[-63:].std(ddof=1)*21**e.H)
        self.assertEqual((e.window_H,e.window_a,e.window_sigma),(96,21,63))
        a=estimate(logp)
        b=estimate(logp,windows=EstimationWindows(160,160,160))
        self.assertEqual(asdict(a),asdict(b))

    def test_cached_estimates_match_public_estimator_and_are_causal(self):
        config=TuningConfig(H_windows=(64,),drift_windows=(8,),sigma_windows=(16,))
        world=scenarios(config)[0]
        data=synthetic_market(config,'coarse',world,0,3)
        c=Candidate(64,8,16,'rs',.2)
        actual=EstimatorCache(data,config,(.06,.98)).get(c)
        for i,start in enumerate(data['starts']):
            direct=estimate(data['logp'][start-64:start+1],windows=c.windows)
            for key in ('H','a','sigma','mu'):
                self.assertAlmostEqual(actual[key][i],getattr(direct,key),places=12)
        changed={k:np.copy(v) for k,v in data.items()}
        first=int(data['starts'][0])
        changed['logp'][first+1:]+=100
        again=EstimatorCache(changed,config,(.06,.98)).get(c)
        for key in actual:
            self.assertEqual(actual[key][0],again[key][0])

    def test_fixed_H_driver_and_exogenous_profiles(self):
        config=TuningConfig(H_windows=(64,),drift_windows=(8,),sigma_windows=(16,))
        world=dict(id='joint_H0.7',group='changing',H=.7,mu=None,profile='joint')
        data=synthetic_market(config,'coarse',world,1,30)
        n=len(data['logp'])-1
        latent=simulate_log_prices(0,1,.7,n,config.D,seed=simulation_seed(config,'coarse',world,1))[0]
        np.testing.assert_allclose(np.diff(data['logp']),data['true_a']/config.D+data['true_sigma']*np.diff(latent))
        self.assertEqual(float(data['H']),.7)
        for start in data['starts']:
            self.assertEqual(len(np.unique(data['true_a'][start:start+config.D])),1)
            self.assertEqual(len(np.unique(data['true_sigma'][start:start+config.D])),1)
        seeds={simulation_seed(config,s,world,i) for s in ('coarse','validation','test') for i in range(4)}
        self.assertEqual(len(seeds),12)

    def test_grid_completion_and_edge_rejection(self):
        grid=complete_boundary_grid(full_grid())
        self.assertAlmostEqual(grid.mu2.iloc[1],.25)
        self.assertTrue(grid.mu2_imputed.iloc[1])
        b=boundary_object(grid,.2)
        self.assertEqual(b.diagnostics().imputed.tolist(),[1,1])
        broken=full_grid(); broken.loc[0,'mu2_fit']=np.nan
        with self.assertRaisesRegex(ValueError,'endpoint'):
            complete_boundary_grid(broken)

    def test_equal_group_weights_and_replicate_uncertainty(self):
        rows=[]
        metrics=['model','buy_hold','regimes_only','sign_drift','true_parameters','H_bias','a_bias',
                 'sigma_bias','H_mse','a_mse','sigma_mse','cash_share','policy_share','drawdown',
                 'model_second_moment','buy_hold_second_moment']
        for rep in range(2):
            for group,count,value in [('constant',3,1.),('changing',1,3.)]:
                for k in range(count):
                    rows.append(dict(replicate=rep,group=group,scenario=f'{group}{k}',
                                     **{m:value for m in metrics}))
        result=score_rows(rows)
        self.assertEqual(result['score'],2.)
        self.assertEqual(result['replicate_se'],0.)
        self.assertEqual(result['replicates'],2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/'bank'
        cfg=Config(n_fine=4,n_exercise=2,M_train=32,M_test=32,K=1,
                   H_grid=(.3,.5,.7),mu_grid=(-1.,0.,1.))
        run_experiment(cfg,self.run,batch_size=16,M_validation=32)
        self.config=TuningConfig(H_windows=(32,48),drift_windows=(4,8),sigma_windows=(8,),
                                 methods=('rs',),alphas=(.1,.2),true_H=(.3,.5,.7),
                                 stationary_mu=(-.5,.5),profiles=('drift12',),D=2,
                                 coarse_replicates=2,validation_replicates=2,test_replicates=2,
                                 coarse_months=4,validation_months=4,test_months=4,top_k=2)

    def run_tuning(self,output,**kwargs):
        with patch('boundary_analysis.preparation.load_experiment_statistics',return_value=({},ideal_nodes())):
            return run_synthetic_tuning(self.run,output,self.config,**kwargs)

    def test_batch_decisions_match_scalar_and_reuse_alpha_cache(self):
        bank=PolicyBank(self.run)
        data=synthetic_market(self.config,'coarse',scenarios(self.config)[0],0,4)
        est=dict(H=np.full(4,.3),sigma=np.full(4,.05),a=np.zeros(4),mu=np.zeros(4))
        boundary=boundary_object(complete_boundary_grid(full_grid()),.2)
        cache=DecisionCache(data,bank,2)
        actual=cache.rewards(est,boundary)
        from estimation.types import ParamEstimate
        expected=[]
        for start in data['starts']:
            e=ParamEstimate(0,.05,.3,0,.3,0,.01,32,2,'rs')
            d=decide_block(data['logp'][start:start+3],e,bank,boundary)
            expected.append(d.tau_day)
        np.testing.assert_array_equal(actual['tau'],expected)
        with patch('experiments.policy.FixedPolicy.decide',side_effect=AssertionError('repeated policy evaluation')):
            cache.rewards(est,boundary)

    def test_interruption_resume_freeze_and_locked_real_application(self):
        interrupted=self.root/'resume'
        def stop(e):
            if e['done']==1:raise RuntimeError('interrupt')
        with self.assertRaises(RuntimeError):self.run_tuning(interrupted,progress=stop)
        self.assertFalse((interrupted/'frozen.json').exists())
        result=self.run_tuning(interrupted)
        self.assertEqual(len(result['validation']),self.config.top_k)
        self.assertEqual(len(result['test']),1)
        full=self.run_tuning(self.root/'full')
        for stage in ('coarse','validation','test'):
            pd.testing.assert_frame_equal(result[stage],full[stage])
        frozen_before=(interrupted/'frozen.json').read_bytes()
        with patch('tuning.runner.synthetic_market',side_effect=AssertionError('simulation')),\
             patch('tuning.cache.EstimatorCache.get',side_effect=AssertionError('estimation')):
            self.run_tuning(interrupted)
        self.assertEqual((interrupted/'frozen.json').read_bytes(),frozen_before)
        selection=load_frozen_selection(interrupted)
        self.assertEqual(selection.candidate.key,result['validation'].candidate.iloc[0])
        self.assertIn(selection.candidate.key,result['test'].candidate.tolist())
        bank=PolicyBank(self.run)
        path=simulate_log_prices(.01,.05,.5,80,D=2,seed=71)[0]
        dates=pd.date_range('2000-01-01',periods=len(path))
        deployed=selection.backtest(path,dates,bank=bank)
        np.testing.assert_array_equal(deployed.window_H,selection.windows.H)
        with self.assertRaisesRegex(ValueError,'override'):
            selection.backtest(path,dates,bank=bank,window=2)
        with self.assertRaisesRegex(ValueError,'another policy bank'):
            selection.backtest(path,dates,bank=SimpleNamespace(identity='other',D=2))
        with patch('tuning.frozen.dependency_versions',return_value={'modified':'version'}):
            with self.assertRaisesRegex(ValueError,'environment changed'):
                load_frozen_selection(interrupted)
        policy=next((self.run/'policies').glob('*.json'))
        original=policy.read_text()
        policy.write_text(original+'\n')
        with self.assertRaisesRegex(ValueError,'policy bank'):
            load_frozen_selection(interrupted)
        policy.write_text(original)
        p=interrupted/'frozen_boundaries.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            load_frozen_selection(interrupted)

    def test_changed_config_rejected(self):
        output=self.root/'versioned';self.run_tuning(output)
        with self.assertRaisesRegex(ValueError,'TUNING_ID'):
            run_synthetic_tuning(self.run,output,replace(self.config,seed=4))

    def test_test_must_finish_before_market_use(self):
        output=self.root/'unfinished_test'
        def stop(e):
            if e['stage']=='test':raise RuntimeError('interrupt audit')
        with self.assertRaises(RuntimeError):self.run_tuning(output,progress=stop)
        self.assertTrue((output/'frozen.json').exists())
        with self.assertRaisesRegex(FileNotFoundError,'notebook 06'):
            load_frozen_selection(output)
        self.run_tuning(output)
        load_frozen_selection(output)


if __name__=='__main__':unittest.main()
