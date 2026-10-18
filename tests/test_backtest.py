"""Backtest mechanics on a tiny trained bank: no look-ahead, accounting, regimes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from osfbm import adapter
from osfbm.backtest import (by_period, calibration, decide_block, hac_se, max_drawdown,
                            paired_difference, run_backtest, summarize, window_search)
from osfbm.config import Config
from osfbm.confidence import FixedPolicy
from osfbm.data import block_frame, month_blocks
from osfbm.estimation import estimate, simulate_log_prices
from osfbm.experiment import run_experiment
from osfbm.policy_bank import CASH, HOLD, POLICY, Boundaries, PolicyBank, ValueTable, point_boundaries

D = 4  # tiny month for speed; the estimation window is still long


class PolicyDecideTests(unittest.TestCase):
    def test_decide_matches_rewards(self):
        cfg = Config(n_fine=8, n_exercise=4, M_train=64, K=2)
        B, _ = adapter.simulate_paths(0.7, cfg, 3)
        rng = np.random.default_rng(0)
        policy = FixedPolicy(rng.normal(size=(3, cfg.sig_dim)), rng.normal(size=3), False)
        X = adapter.drifted(B, 0.4, cfg)
        idx = policy.decide(X, cfg)
        np.testing.assert_allclose(policy.rewards(B, 0.4, cfg),
                                   X[np.arange(len(B)), cfg.exercise_index[idx]])
        self.assertTrue(np.all((idx >= 0) & (idx < cfg.n_exercise)))
        with self.assertRaises(ValueError):
            policy.decide(X[:, :-1], cfg)


class BlockTests(unittest.TestCase):
    def test_blocks_are_consecutive_and_respect_warmup(self):
        blocks = month_blocks(100, D=21, warmup=30)
        self.assertEqual(blocks[0, 0], 30)
        np.testing.assert_array_equal(blocks[1:, 0], blocks[:-1, 1])
        np.testing.assert_array_equal(blocks[:, 1] - blocks[:, 0], 21)
        self.assertLess(blocks[-1, 1], 100)
        frame = block_frame(pd.date_range("2000-01-01", periods=100, freq="B"), blocks)
        self.assertEqual(list(frame.columns), ["decision_date", "end_date", "decision_idx", "end_idx"])


class BoundaryTests(unittest.TestCase):
    def test_interpolation_missing_mu2_and_regimes(self):
        frame = pd.DataFrame({"H": [0.2, 0.4, 0.6], "mu1": [-1.0, -0.5, 0.0],
                              "mu2": [np.nan, 1.0, 2.0]})
        b = Boundaries.from_frame(frame, 0.04)
        self.assertAlmostEqual(float(b.mu1_at(0.3)), -0.75)
        self.assertTrue(np.isinf(b.mu2_at(0.2)))
        self.assertAlmostEqual(float(b.mu2_at(0.5)), 1.5)
        self.assertEqual(b.regime(0.3, -0.9), CASH)
        self.assertEqual(b.regime(0.5, 1.6), HOLD)
        self.assertEqual(b.regime(0.5, 0.2), POLICY)
        self.assertEqual(b.regime(0.2, 100.0), POLICY)  # mu2 missing: never "hold"
        self.assertEqual(b.regime(0.1, 100.0), POLICY)  # below the table: clamped to H=0.2
        self.assertAlmostEqual(float(b.mu1_at(0.9)), 0.0)  # above the table: clamped to H=0.6
        brownian = Boundaries.brownian()
        self.assertEqual(brownian.regime(0.5, 1e-9), HOLD)
        self.assertEqual(brownian.regime(0.3, -1e-9), CASH)
        self.assertEqual(brownian.regime(0.3, 0.0), CASH)

    def test_point_boundaries_on_brownian_values(self):
        mu = np.arange(-1.0, 1.01, 0.25)
        nodes = pd.DataFrame({"H": 0.5, "mu": mu, "V": np.maximum(0, mu), "complete": True})
        nodes["excess"] = nodes["V"] - nodes["mu"]
        table = point_boundaries(nodes, mu, epsilon=0.3)
        self.assertEqual((table.loc[0, "mu1"], table.loc[0, "mu2"]), (0.25, -0.25))


class BacktestWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name) / "bank"
        cls.cfg = Config(n_fine=D, n_exercise=D, M_train=300, M_test=300, K=2, seed=77,
                         H_grid=(0.5, 0.7), mu_grid=tuple(np.arange(-2.0, 2.01, 0.5)))
        run_experiment(cls.cfg, cls.run_dir, batch_size=150, M_validation=120)
        cls.bank = PolicyBank(cls.run_dir)
        cls.values = ValueTable.from_run(cls.run_dir)
        cls.boundaries = Boundaries.from_frame(
            pd.DataFrame({"H": [0.5, 0.7], "mu1": [-0.5, -1.0], "mu2": [0.5, 1.5]}), 0.1)
        cls.logp = simulate_log_prices(0.3, 1.0, 0.6, 20 * D + 300, D, seed=12)[0]
        cls.dates = pd.date_range("1990-01-01", periods=len(cls.logp), freq="B")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bank_lookup_and_values(self):
        self.assertEqual(self.bank.D, D)
        self.assertEqual(self.bank.nearest_node(0.63, 0.74), (0.7, 0.5))
        h, m, policy = self.bank.nearest(0.52, -0.1)
        self.assertEqual((h, m), (0.5, 0.0))
        self.assertIsInstance(policy, FixedPolicy)
        self.assertEqual(self.bank.coverage(), 1.0)
        self.assertTrue(np.isfinite(self.values.value(0.5, 0.0)))
        self.assertEqual(self.values.nearest_node(0.99, 9.0), (0.7, 2.0))

    def test_decide_block_actions(self):
        block = self.logp[300: 300 + D + 1]
        est = estimate(self.logp[:301], D)
        cash = decide_block(block, est.__class__(**{**est.as_row(), "mu": -5.0}),
                            self.bank, self.boundaries, self.values)
        hold = decide_block(block, est.__class__(**{**est.as_row(), "mu": 5.0}),
                            self.bank, self.boundaries, self.values)
        middle = decide_block(block, est.__class__(**{**est.as_row(), "mu": 0.0, "H": 0.7}),
                              self.bank, self.boundaries, self.values)
        self.assertEqual((cash.action, cash.tau_day), (CASH, 0))
        self.assertEqual((hold.action, hold.tau_day), (HOLD, D))
        self.assertEqual(middle.regime, POLICY)
        self.assertTrue(0 <= middle.tau_day <= D)
        self.assertEqual((middle.node_H, middle.node_mu), (0.7, 0.0))

    def test_no_look_ahead_and_accounting(self):
        window = 300
        res = run_backtest(self.logp, self.dates, window=window, D=D, bank=self.bank,
                           boundaries=self.boundaries, values=self.values)
        self.assertEqual(len(res), len(month_blocks(len(self.logp), D, window)))
        # Perturb everything after the end of block 5: earlier rows must not change.
        cut = int(res.loc[5, "end_idx"])
        other = self.logp.copy()
        other[cut + 1:] += np.linspace(0, 3, len(other) - cut - 1)
        res2 = run_backtest(other, self.dates, window=window, D=D, bank=self.bank,
                            boundaries=self.boundaries, values=self.values)
        pd.testing.assert_frame_equal(res.iloc[:6], res2.iloc[:6])
        # Rewards are the realised log returns of the chosen day.
        for _, row in res.iterrows():
            s, e, tau = int(row.decision_idx), int(row.end_idx), int(row.tau_day)
            self.assertAlmostEqual(row.r_buy_hold, self.logp[e] - self.logp[s])
            expected = 0.0 if tau == 0 else self.logp[s + tau] - self.logp[s]
            self.assertAlmostEqual(row.r_model, expected)
            self.assertGreaterEqual(row.r_oracle + 1e-12, max(row.r_model, row.r_buy_hold, 0.0))
            self.assertAlmostEqual(row.r_sign_drift, row.r_buy_hold if row.a > 0 else 0.0)
            self.assertTrue(row.regime in (CASH, POLICY, HOLD))
        # Costs reduce every invested reward by a round trip.
        costly = run_backtest(self.logp, self.dates, window=window, D=D, bank=self.bank,
                              boundaries=self.boundaries, values=self.values, cost=0.001)
        np.testing.assert_allclose(costly.r_buy_hold, res.r_buy_hold - 0.002)
        invested = res.tau_day > 0
        np.testing.assert_allclose(costly.r_model[invested], res.r_model[invested] - 0.002)
        np.testing.assert_allclose(costly.r_model[~invested], 0.0)

    def test_brownian_boundaries_reduce_to_sign_of_drift(self):
        res = run_backtest(self.logp, self.dates, window=300, D=D, bank=None,
                           boundaries=Boundaries.brownian(), values=None)
        np.testing.assert_allclose(res.r_regimes_only, res.r_sign_drift)
        self.assertTrue(set(res.regime) <= {CASH, HOLD})

    def test_statistics(self):
        res = run_backtest(self.logp, self.dates, window=300, D=D, bank=self.bank,
                           boundaries=self.boundaries, values=self.values)
        table = summarize(res)
        self.assertEqual(list(table.index), ["model", "regimes_only", "sign_drift", "buy_hold", "oracle"])
        self.assertTrue(np.isfinite(table["mean"]).all())
        self.assertAlmostEqual(table.loc["buy_hold", "frac_invested"], 1.0)
        diff = paired_difference(res, "model", "buy_hold", n_boot=200)
        self.assertLessEqual(diff["boot_low"], diff["boot_high"])
        self.assertAlmostEqual(diff["mean_diff"], (res.r_model - res.r_buy_hold).mean())
        self.assertAlmostEqual(hac_se(np.ones(50)), 0.0)
        self.assertAlmostEqual(max_drawdown([1, -2, 1]), 2.0)
        periods = by_period(res, freq="YS")
        self.assertEqual(periods["months"].sum(), len(res))
        cal = calibration(res, bins=3)
        self.assertEqual(cal["n"].sum(), res[["pred_model", "r_model"]].dropna().shape[0])
        table, results = window_search(self.logp, self.dates, windows=(200, 300),
                                       split_date=self.dates[300 + 10 * D],
                                       D=D, bank=self.bank, boundaries=self.boundaries,
                                       values=self.values)
        self.assertEqual(set(results), {200, 300})
        self.assertIn(table.attrs["chosen_window"], (200, 300))


if __name__ == "__main__":
    unittest.main()
