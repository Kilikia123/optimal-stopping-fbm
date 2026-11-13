"""Visual diagnostics preserve counts and never rerun the model."""
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
import numpy as np
import pandas as pd

from visualization.boundaries import plot_final_boundaries
from visualization.market import plot_parameters, plot_boundary_diagnostics, plot_saved_market_results, plot_regime_shares
from boundary_analysis.curves import Boundaries
from tuning.config import Candidate, TuningConfig
from tuning.scenarios import scenarios
from tuning.progress import TextTuningProgress, audit_tuning_run


def fixture():
    c=Candidate(1008,1008,1008,'dfa',.1)
    b=Boundaries.from_frame(pd.DataFrame(dict(H=[.3,.4,.5,.6,.7],
                           mu1=[-.5,-.25,0,-.25,-.5],mu2=[.5,.25,0,.25,.5])),.05)
    dates=pd.date_range('2000-01-01',periods=6,freq='30D')
    r=pd.DataFrame(dict(decision_date=dates,end_date=dates+pd.Timedelta(days=29),
                       H=[.3,.4,.5,.6,.7,.6],mu=[-1.,0.,.3,-.1,1.,-1.],sigma=.05,
                       regime=['cash','policy','hold','policy','hold','cash'],
                       r_model=[0,.01,.02,-.01,.03,0],r_buy_hold=[-.02,.015,.02,-.03,.03,.01],
                       r_regimes_only=[0,.015,.02,-.03,.03,0],r_sign_drift=[0,0,.02,0,.03,0],
                       window_H=1008,window_a=1008,window_sigma=1008,method='dfa',boundary_alpha=.1))
    r['a']=r.mu*r.sigma
    return r,SimpleNamespace(candidate=c,windows=c.windows,boundaries=b)


class PlotTests(unittest.TestCase):
    def tearDown(self):plt.close('all')

    def test_all_parameters_and_regime_counts_are_visible(self):
        r,s=fixture()
        intervals=r[['decision_date','H','mu','a','sigma']].copy()
        for name in ('H','mu','a','sigma'):
            intervals[f'{name}_low']=intervals[name]+.01
            intervals[f'{name}_high']=intervals[name]+.02
        f=plot_parameters(r,windows=s.windows,intervals=intervals)
        self.assertEqual([a.get_ylabel() for a in f.axes],['H','a','sigma','mu'])
        f=plot_boundary_diagnostics(r,s.boundaries)
        np.testing.assert_allclose([p.get_width() for p in f.axes[1].patches],[1/3]*3)
        offsets=[x.get_offsets() for x in f.axes[0].collections if len(x.get_offsets())==2]
        self.assertEqual(sum(len(x) for x in offsets),6)
        plot_regime_shares(r)

    def test_csv_graph_rebuild_does_not_simulate(self):
        r,s=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            r.to_csv(Path(tmp)/'blocks.csv',index=False)
            with patch('adapter.simulate_paths',side_effect=AssertionError('simulation')),\
                 patch('market.backtest.run_backtest',side_effect=AssertionError('backtest')):
                figures=plot_saved_market_results(tmp,s)
            for name in ('cumulative','parameter_history','boundary_diagnostics','regime_shares','final_boundaries'):
                self.assertIn(name,figures)
                self.assertTrue((Path(tmp)/f'{name}.png').exists())
            r.loc[0,'regime']='hold'
            with self.assertRaisesRegex(ValueError,'regimes'):
                plot_boundary_diagnostics(r,s.boundaries)

    def test_final_curves_use_locked_values_and_distinct_markers(self):
        _,s=fixture()
        s.boundaries=replace(s.boundaries,statuses={
            1:np.array(['fitted','fitted','analytic','fitted','fitted']),
            2:np.array(['fitted','interpolated','analytic','fitted_unconfirmed','fitted'])})
        before=s.boundaries.frame().copy()
        with tempfile.TemporaryDirectory() as tmp,\
             patch('adapter.simulate_paths',side_effect=AssertionError('simulation')),\
             patch('boundary_analysis.preparation.prepare_boundary_grids',side_effect=AssertionError('refit')):
            fig=plot_final_boundaries(s,output_dir=tmp)
            self.assertTrue((Path(tmp)/'final_boundaries.png').exists())
        self.assertEqual(len(fig.axes),2)
        self.assertEqual(fig.axes[1].get_xlim(),(.35,.65))
        self.assertIn('α=0.1',fig._suptitle.get_text())
        self.assertIn('DFA',fig._suptitle.get_text())
        self.assertTrue(any('1008' in t.get_text() for t in fig.texts))
        for ax in fig.axes:
            self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections if isinstance(c,PathCollection)),2*len(s.boundaries.H))
            np.testing.assert_allclose(ax.collections[5].get_offsets(),[[.4,.25]])
            np.testing.assert_allclose(ax.collections[4].get_offsets(),[[.6,.25]])
        pd.testing.assert_frame_equal(before,s.boundaries.frame())

    def test_final_boundary_intervals_preserve_saved_endpoints_and_fit(self):
        _,s=fixture()
        s.boundaries=replace(s.boundaries,statuses={
            1:np.array(['fitted','fitted','analytic','fitted','fitted']),
            2:np.array(['fitted','interpolated','analytic','fitted_unconfirmed','fitted'])})
        frame=s.boundaries.frame()
        frame['confidence']=.95
        for side in (1,2):
            frame[f'mu{side}_low']=frame[f'mu{side}']-.15
            frame[f'mu{side}_high']=frame[f'mu{side}']+.20
            frame.loc[frame.H.eq(.5),[f'mu{side}_low',f'mu{side}_high']]=0.
        before=frame.copy()
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'frozen_boundaries.csv'
            frame.to_csv(source,index=False)
            s.boundaries=replace(s.boundaries,source=str(source))
            with patch('boundary_analysis.preparation.prepare_boundary_grids',side_effect=AssertionError('refit')):
                fig=plot_final_boundaries(s)
            for ax in fig.axes:
                bars={c.get_label():c for c in ax.collections if isinstance(c,LineCollection)}
                self.assertEqual(len(bars['mu1_intervals'].get_segments()),5)
                self.assertEqual(len(bars['mu2_intervals'].get_segments()),3)
                np.testing.assert_allclose(bars['mu1_intervals'].get_segments()[1],[[.4,-.4],[.4,-.05]])
                # Zoom includes interval endpoints, not just the fitted curve.
                if ax is fig.axes[1]:
                    self.assertLessEqual(ax.get_ylim()[0],-.4)
                for bar in bars.values():
                    for segment in bar.get_segments():
                        if ax.get_xlim()[0]<=segment[0,0]<=ax.get_xlim()[1]:
                            self.assertLessEqual(ax.get_ylim()[0],segment[:,1].min())
                            self.assertGreaterEqual(ax.get_ylim()[1],segment[:,1].max())
            pd.testing.assert_frame_equal(pd.read_csv(source),before)
            self.assertTrue(any('95%' in text.get_text() for text in fig.legends[0].texts))
            frame.loc[0,'mu1']=-.9
            frame.to_csv(source,index=False)
            with self.assertRaisesRegex(ValueError,'differs'):
                plot_final_boundaries(s)

    def test_final_graph_rejects_incomplete_grid(self):
        _,s=fixture()
        s.boundaries=replace(s.boundaries,mu2=np.array([.5,np.nan,0,.25,.5]))
        with self.assertRaisesRegex(ValueError,'complete'):
            plot_final_boundaries(s)


class ProgressTests(unittest.TestCase):
    def test_text_output_and_cache_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scores/coarse/c';p.mkdir(parents=True)
            (p/'row.json').write_text('{}')
            with patch('sys.stdout',new_callable=io.StringIO) as stream:
                with TextTuningProgress(tmp) as reporter:
                    reporter(dict(stage='coarse',done=1,total=2))
                    reporter(dict(stage='coarse',done=2,total=2))
            self.assertIn('сохранено ранее 1',stream.getvalue())
            report=json.loads((Path(tmp)/'last_execution.json').read_text())
            self.assertEqual(report['stages']['coarse']['cached_at_start'],1)
            self.assertTrue(report['completed'])

    def test_audit_detects_missing_score(self):
        cfg=TuningConfig(H_windows=(32,),drift_windows=(4,),sigma_windows=(8,),
                         methods=('rs',),alphas=(.1,),true_H=(.5,),stationary_mu=(0.,),profiles=('drift12',),
                         coarse_replicates=2,validation_replicates=2,test_replicates=2,
                         coarse_months=2,validation_months=2,test_months=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'experiment.json').write_text(json.dumps(dict(config=asdict(cfg))))
            for stage in ('coarse','validation','test'):
                pd.DataFrame(dict(candidate=['c'])).to_csv(root/f'{stage}_ranking.csv',index=False)
                rows=[]
                for rep in range(2):
                    for world in scenarios(cfg):
                        rows.append(dict(candidate='c',scenario=world['id'],replicate=rep))
                        path=root/'scores'/stage/'c'/f"{world['id']}_{rep}.json"
                        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
                        m=root/'markets'/stage/f"{world['id']}_{rep}.npz"
                        m.parent.mkdir(parents=True,exist_ok=True)
                        np.savez(m,starts=np.arange(2),logp=np.zeros(cfg.warmup+2*cfg.D+1))
                pd.DataFrame(rows).to_csv(root/f'{stage}_realizations.csv',index=False)
            self.assertTrue(audit_tuning_run(root).complete.all())
            p=root/'test_realizations.csv';r=pd.read_csv(p);r.iloc[:-1].to_csv(p,index=False)
            self.assertFalse(audit_tuning_run(root).set_index('stage').loc['test','complete'])


if __name__=='__main__':unittest.main()
