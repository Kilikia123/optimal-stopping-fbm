"""Estimators recover the parameters of simulated fBm price paths."""
import unittest

import numpy as np

from osfbm.estimation import (bootstrap_estimates, detrend, dfa_hurst, estimate,
                              log_returns, percentile_ci, rs_hurst, simulate_log_prices)

D = 21


class SimulationTests(unittest.TestCase):
    def test_shape_time_scale_and_drift(self):
        paths = simulate_log_prices(a=1.2, sigma=0.5, H=0.6, n_days=D * 40, D=D,
                                    seed=5, n_paths=400, log_s0=3.0)
        self.assertEqual(paths.shape, (400, D * 40 + 1))
        np.testing.assert_allclose(paths[:, 0], 3.0)
        # After 40 months the mean log price is log_s0 + 40 a; the fBm part has
        # standard deviation sigma * 40**H across paths.
        final = paths[:, -1] - 3.0
        self.assertLess(abs(final.mean() - 48.0), 4 * 0.5 * 40 ** 0.6 / np.sqrt(400))
        self.assertLess(abs(final.std(ddof=1) / (0.5 * 40 ** 0.6) - 1), 0.15)


class EstimatorTests(unittest.TestCase):
    def test_detrend_removes_the_linear_trend(self):
        logp = np.cumsum(np.r_[0.0, 0.01 + 0.02 * np.random.default_rng(1).normal(size=99)])
        y = detrend(logp)
        self.assertAlmostEqual(y[0], 0.0)
        self.assertAlmostEqual(y[-1], 0.0)
        np.testing.assert_allclose(np.diff(y), log_returns(logp) - log_returns(logp).mean())

    def test_drift_does_not_change_H_or_sigma(self):
        base = simulate_log_prices(0.0, 1.0, 0.6, 600, D, seed=2)[0]
        k = np.arange(len(base))
        shifted = base + 5.0 / D * k  # add a drift of 5 per month
        e0, e1 = estimate(base, D), estimate(shifted, D)
        self.assertAlmostEqual(e1.a - e0.a, 5.0, places=9)
        self.assertAlmostEqual(e1.H, e0.H, places=12)
        self.assertAlmostEqual(e1.sigma, e0.sigma, places=12)

    def test_sigma_formula_and_hurst_recovery(self):
        for H, sigma in ((0.3, 2.0), (0.5, 1.0), (0.7, 0.5)):
            paths = simulate_log_prices(0.4, sigma, H, 1000, D, seed=7, n_paths=60)
            ests = [estimate(p, D) for p in paths]
            H_mean = np.mean([e.H_raw for e in ests])
            s_mean = np.mean([e.sigma for e in ests])
            self.assertLess(abs(H_mean - H), 0.08, msg=f"H={H}: {H_mean}")
            # The competing formula std * D**(-H) would be off by D**(2H) >= 6.
            self.assertLess(abs(s_mean / sigma - 1), 0.25, msg=f"H={H}: {s_mean}")
            wrong = np.mean([e.std_daily * D ** (-e.H) for e in ests])
            self.assertLess(wrong / sigma, 0.2)

    def test_iid_series_has_H_one_half_under_both_methods(self):
        x = np.random.default_rng(3).normal(size=(40, 1000))
        rs = np.mean([rs_hurst(row) for row in x])
        dfa = np.mean([dfa_hurst(row) for row in x])
        self.assertLess(abs(rs - 0.5), 0.05)
        self.assertLess(abs(dfa - 0.5), 0.05)

    def test_estimate_fields_and_clipping(self):
        logp = simulate_log_prices(0.0, 1.0, 0.5, 300, D, seed=9)[0]
        e = estimate(logp, D, method="dfa", clip=(0.4, 0.45))
        self.assertEqual((e.n_returns, e.D, e.method), (300, D, "dfa"))
        self.assertTrue(0.4 <= e.H <= 0.45)
        self.assertAlmostEqual(e.mu, e.a / e.sigma)
        self.assertAlmostEqual(e.sigma, e.std_daily * D ** e.H)
        with self.assertRaises(ValueError):
            estimate(logp, D, method="wavelet")
        with self.assertRaises(ValueError):
            estimate(logp[:10], D)

    def test_bootstrap_rows_and_interval(self):
        logp = simulate_log_prices(0.3, 1.0, 0.55, 400, D, seed=4)[0]
        e = estimate(logp, D)
        boot = bootstrap_estimates(e, 400, n_boot=30, seed=1)
        self.assertEqual(len(boot), 30)
        self.assertTrue(np.isfinite(boot[["a", "sigma", "H", "mu"]]).all().all())
        low, high = percentile_ci(boot["H"], 0.9)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
