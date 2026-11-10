"""Counterfactual actions reuse the original estimates, policies and chronology."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market.action_audit import evaluate_actions, summarize_actions, run_action_audit
from visualization.actions import plot_action_audit
from config import Config
from infrastructure.identity import node_key
from boundary_analysis.curves import Boundaries
from tuning.config import Candidate


class SavedRule:
    def __init__(self,day):self.day=day;self.stopped_at_zero=day==0
    def decide(self,x,cfg):return np.full(len(x),self.day-1,dtype=int)


class Bank:
    identity='fixture'
    D=2
    cfg=Config(n_fine=2,n_exercise=2,K=1)
    def __init__(self,root,day):self.run_dir=Path(root);self.rule=SavedRule(day)
    def nearest_node(self,h,mu):return (.3,float(mu))
    def nearest(self,h,mu):return (*self.nearest_node(h,mu),self.rule)


def fixture(root,day=1):
    bank=Bank(root,day)
    candidate=Candidate(32,2,2,'rs',.1)
    boundaries=Boundaries.from_frame(pd.DataFrame(dict(H=[.3,.7],mu1=[-.5,-.5],mu2=[.5,.5])),.05)
    selection=SimpleNamespace(candidate=candidate,identity=bank.identity,D=2,boundaries=boundaries)
    increments=np.array([.03,-.01,-.02,.04,.01,.02,-.04,.01,.05,-.02,-.01,.04])
    logp=np.r_[0.,np.cumsum(increments)]
    dates=pd.date_range('2000-01-01',periods=len(logp))
    starts=np.arange(0,12,2);ends=starts+2
    regions=np.array(['cash','policy','hold']*2)
    mu=np.tile([-1.,0.,1.],2)
    tau=np.where(regions=='cash',0,np.where(regions=='hold',2,day))
    result=pd.DataFrame(dict(decision_idx=starts,end_idx=ends,decision_date=dates[starts],end_date=dates[ends],
                             H=.3,mu=mu,sigma=.05,regime=regions,node_H=.3,node_mu=mu,
                             tau_day=tau,r_model=logp[starts+tau]-logp[starts],
                             r_buy_hold=logp[ends]-logp[starts],
                             window_H=32,window_a=2,window_sigma=2,method='rs',boundary_alpha=.1))
    directory=Path(root)/'policies';directory.mkdir(parents=True,exist_ok=True)
    for m in (-1.,0.,1.):(directory/f'{node_key(.3,m)}.json').write_text('{}')
    return logp,dates,result,bank,selection


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)

    def test_fixed_hold_and_cash_policies(self):
        for day in (0,2):
            logp,dates,result,bank,selection=fixture(self.tmp.name,day)
            with patch('estimation.pipeline.estimate',side_effect=AssertionError('estimation')),\
                 patch('experiments.training.train_policy',side_effect=AssertionError('training')):
                rows=evaluate_actions(logp,dates,result,bank=bank,selection=selection)
            np.testing.assert_array_equal(rows.r_cash,0)
            np.testing.assert_allclose(rows.r_policy,rows.r_hold if day==2 else 0,atol=0)
            np.testing.assert_array_equal(rows.tau_policy_day,day)
            self.assertTrue(rows.policy_stopped_at_zero.eq(day==0).all())

    def test_known_early_exit_and_decomposition(self):
        logp,dates,result,bank,selection=fixture(self.tmp.name)
        rows=evaluate_actions(logp,dates,result,bank=bank,selection=selection)
        np.testing.assert_allclose(rows.r_policy,np.diff(logp)[::2])
        np.testing.assert_allclose(rows.loc[rows.regime.eq('policy'),'r_policy'],
                                   rows.loc[rows.regime.eq('policy'),'r_model'])
        before=result.copy()
        # Force one resample with no cash: its conditional mean must be skipped.
        idx=np.tile(np.array([1,2,4,5,1,2]),(100,1))
        with patch('market.action_audit.stationary_bootstrap_indices',return_value=idx) as bootstrap:
            summary,comparisons=summarize_actions(rows,n_boot=100)
        self.assertEqual(bootstrap.call_args.args[:2],(len(rows),100))
        self.assertAlmostEqual(summary.model_minus_hold_contribution.sum(),(rows.r_model-rows.r_hold).mean())
        cash=comparisons[comparisons.regime.eq('cash')]
        self.assertTrue(cash.boot_low.isna().all())
        self.assertTrue(cash.status.eq('insufficient_data').all())
        self.assertEqual(cash.valid_bootstraps.sum(),0)
        pd.testing.assert_frame_equal(before,result)
        with self.assertRaisesRegex(ValueError,'chronology'):
            summarize_actions(rows[rows.regime.eq('policy')],n_boot=100)

    def test_pair_bootstrap_matches_joint_calendar_resampling(self):
        logp,dates,result,bank,selection=fixture(self.tmp.name)
        rows=evaluate_actions(logp,dates,result,bank=bank,selection=selection)
        idx=np.tile(np.arange(6),(100,1))
        idx[::2]=np.array([0,1,1,2,3,5])
        with patch('market.action_audit.stationary_bootstrap_indices',return_value=idx):
            _,comparison=summarize_actions(rows,n_boot=100)
        mask=rows.regime.to_numpy()[idx]=='policy'
        delta=(rows.r_policy-rows.r_hold).to_numpy()[idx]
        means=np.where(mask,delta,0).sum(axis=1)/mask.sum(axis=1)
        row=comparison[comparison.regime.eq('policy') & comparison.a.eq('policy') & comparison.b.eq('hold')].iloc[0]
        self.assertAlmostEqual(row.boot_low,np.quantile(means,.025))
        self.assertAlmostEqual(row.boot_high,np.quantile(means,.975))

    def test_mismatched_sources_are_rejected(self):
        logp,dates,result,bank,selection=fixture(self.tmp.name)
        bad=result.copy();bad.loc[0,'end_date']=dates[3]
        with self.assertRaisesRegex(ValueError,'dates'):
            evaluate_actions(logp,dates,bad,bank=bank,selection=selection)
        bad=result.copy();bad['r_buy_hold']-=.001
        with self.assertRaisesRegex(ValueError,'rewards'):
            evaluate_actions(logp,dates,bad,bank=bank,selection=selection)
        with self.assertRaisesRegex(ValueError,'consecutive'):
            evaluate_actions(logp,dates,result.iloc[[0,2,3]],bank=bank,selection=selection)
        bad=result.copy();bad.loc[0,'window_a']=8
        with self.assertRaisesRegex(ValueError,'configuration'):
            evaluate_actions(logp,dates,bad,bank=bank,selection=selection)

    def test_cache_never_replays_policy_or_downloads_prices(self):
        root=Path(self.tmp.name)
        logp,dates,result,bank,selection=fixture(root)
        result.to_csv(root/'blocks.csv',index=False)
        prices=root/'prices.csv'
        pd.Series(np.exp(logp)*100,index=dates,name='close').to_csv(prices,index_label='date',float_format='%.6f')
        with self.assertRaisesRegex(FileNotFoundError,'RECOMPUTE_ACTION_AUDIT'):
            run_action_audit(root,selection,bank=bank,prices_path=prices,n_boot=100)
        first=run_action_audit(root,selection,bank=bank,prices_path=prices,n_boot=100,recompute=True)
        with patch.object(bank,'nearest',side_effect=AssertionError('policy replay')),\
             patch('market.data.download_prices',side_effect=AssertionError('network')):
            cached=run_action_audit(root,selection,bank=bank,prices_path=prices,n_boot=100)
        pd.testing.assert_frame_equal(first['summary'],cached['summary'])
        self.assertTrue((root/'action_audit/metadata.json').exists())
        with self.assertRaisesRegex(ValueError,'settings changed'):
            run_action_audit(root,selection,bank=bank,prices_path=prices,n_boot=200)
        fig=plot_action_audit(cached['summary'],cached['comparisons'])
        self.assertEqual(len(fig.axes),2)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__=='__main__':unittest.main()
