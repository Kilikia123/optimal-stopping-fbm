"""Analytical boundary checks and a small, resumable notebook workflow."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import adapter
import experiments.legacy_grid as results
from boundary_analysis.fixed import boundaries_from_scan
from config import Config
from core import NodeResult
from experiments.statistics import load_experiment
from experiments.training import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def brownian_nodes(cfg):
    return [NodeResult(
        H=0.5, mu=mu, V=max(0.0, mu * cfg.T), V_raw=mu * cfg.T,
        SE=0.1, p0=float(mu <= 0), p1=float(mu > 0),
        E_tau=cfg.T if mu > 0 else 0.0, stopped_at_zero=mu <= 0,
        config_hash=cfg.hash,
    ) for mu in cfg.mu_grid]


class EpsilonBoundariesTest(unittest.TestCase):
    def test_brownian_overlap_and_se_independence(self):
        cfg = Config(mu_grid=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0))
        b = boundaries_from_scan(brownian_nodes(cfg), cfg, epsilon=0.25)
        self.assertEqual((b.mu1_epsilon, b.mu2_epsilon, b.boundary_difference), (0.25, -0.25, -0.5))
        self.assertEqual(b.mu1_bracket, (0.25, 0.5))
        self.assertEqual(b.mu2_bracket, (-0.5, -0.25))
        self.assertIn("перекрытие", b.note)
        other = cfg.replace(tol_se=1000)
        bb = boundaries_from_scan(brownian_nodes(other), other, epsilon=0.25)
        self.assertEqual(b.boundary_difference, bb.boundary_difference)
        from dataclasses import replace
        noisy = [replace(r, SE=999.0) for r in brownian_nodes(cfg)]
        self.assertEqual(boundaries_from_scan(noisy, cfg, epsilon=0.25).boundary_difference, -0.5)

    def test_horizon_and_epsilon_scaling(self):
        # At T=4, epsilon=1 corresponds to T=1, epsilon=1/2 for H=1/2.
        unit = Config(mu_grid=(-1.0, -0.5, 0.0, 0.5, 1.0))
        long = unit.replace(T=4.0, mu_grid=tuple(mu / 2 for mu in unit.mu_grid))
        a = boundaries_from_scan(brownian_nodes(unit), unit, epsilon=0.5)
        b = boundaries_from_scan(brownian_nodes(long), long, epsilon=1.0)
        self.assertEqual(b.mu1_epsilon, a.mu1_epsilon / 2)
        self.assertEqual(b.mu2_epsilon, a.mu2_epsilon / 2)
        self.assertEqual(b.boundary_difference, -0.5)

    def test_missing_transitions_do_not_become_edge_estimates(self):
        for grid in [(-2.0, -1.0), (1.0, 2.0), (0.0,)]:
            cfg = Config(mu_grid=grid)
            b = boundaries_from_scan(brownian_nodes(cfg), cfg, epsilon=0.25)
            self.assertIsNone(b.mu1_epsilon)
            self.assertIsNone(b.mu2_epsilon)
            self.assertIsNone(b.boundary_difference)

    def test_nonuniform_grid_brackets(self):
        cfg = Config(mu_grid=(-2.0, -0.3, 0.1, 0.9, 2.0))
        b = boundaries_from_scan(brownian_nodes(cfg), cfg, epsilon=0.25)
        self.assertEqual(b.mu1_bracket, (0.1, 0.9))
        self.assertEqual(b.mu2_bracket, (-0.3, 0.1))

    def test_invalid_and_repeated_transitions(self):
        from dataclasses import replace
        cfg = Config(mu_grid=(-2.0, -1.0, 0.0, 1.0))
        nodes = brownian_nodes(cfg)
        for epsilon in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                boundaries_from_scan(nodes, cfg, epsilon=epsilon)
        with self.assertRaises(ValueError):
            boundaries_from_scan([], cfg, epsilon=0.1)
        with self.assertRaises(ValueError):
            boundaries_from_scan(nodes + nodes[:1], cfg, epsilon=0.1)
        with self.assertRaises(ValueError):
            boundaries_from_scan([replace(nodes[0], H=0.3), *nodes[1:]], cfg, epsilon=0.1)
        nodes[1] = replace(nodes[1], V=0.5)
        b = boundaries_from_scan(nodes, cfg, epsilon=0.1)
        self.assertIsNone(b.mu1_epsilon)
        self.assertIn("повторный вход", b.note)


class SavedRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "result_optimal_stopping" / "runs" / "smoke"
        self.cfg = Config(n_exercise=2, n_fine=4, M_train=64, M_test=80, K=2,
                          H_grid=(0.3, 0.5), mu_grid=(-1.0, 0.0, 1.0))

    def test_checkpoint_resume_and_config_identity(self):
        count = 0
        def stop_after_two(node):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            results.run_grid(self.cfg, self.run, progress=stop_after_two)
        cfg, saved = results.load_run(self.run)
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(len(saved), 2)
        with patch.object(results, "value_at", wraps=results.value_at) as price:
            full = results.run_grid(self.cfg, self.run)
        self.assertEqual(price.call_count, 4)
        self.assertEqual(len(full), 6)
        with patch.object(adapter, "simulate_pair", side_effect=AssertionError("resimulation")):
            pd.testing.assert_frame_equal(full, results.run_grid(self.cfg, self.run))
        # The old node hash excludes grids; run identity must still reject a changed grid.
        with self.assertRaises(ValueError):
            results.prepare_run(self.run, self.cfg.replace(mu_grid=(-1.0, 0.0)))
        with self.assertRaises(ValueError):
            results.prepare_run(self.run, self.cfg.replace(H_grid=(0.3,)))

    def execute_notebook(self, name, overrides=None):
        notebook = json.loads((ROOT / "notebooks" / name).read_text())
        namespace = {"__name__": "__main__"}
        with patch.object(Path, "cwd", return_value=self.root),\
             patch.object(plt, "show"), contextlib.redirect_stdout(io.StringIO()),\
             contextlib.redirect_stderr(io.StringIO()):
            for index, c in enumerate(notebook["cells"]):
                if c["cell_type"] != "code":
                    continue
                source = "".join(c["source"])
                if overrides and ("cfg = Config(" in source or source.startswith("RUN_ID =")):
                    namespace.update(overrides)
                    continue
                if overrides and source.startswith("H =") and "H" in overrides:
                    namespace["H"] = overrides["H"]
                    continue
                exec(compile(source, f"{name}:cell{index}", "exec"), namespace)
        plt.close("all")
        return namespace

    def test_failed_checkpoint_preserves_previous_nodes(self):
        results.run_grid(self.cfg, self.run)
        path = self.run / "nodes.csv"
        original = path.read_bytes()
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                results.append(path, [], key=["config_hash", "H", "mu"])
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.run.glob("*.tmp")))

    def test_notebook_calculates_then_library_analyzes_without_training(self):
        self.execute_notebook("03-grid.ipynb", {
            "MODE": "fresh", "cfg": self.cfg, "RUN_ID": "smoke", "RUN_DIR": self.run,
            "BATCH_SIZE": 40, "M_VALIDATION": 40, "COLLECT_PAIRED": True,
        })
        with patch.object(adapter, "simulate_paths", side_effect=AssertionError("analysis simulates")),\
             patch("experiments.training.train_policy", side_effect=AssertionError("analysis trains")):
            _, nodes, boundaries = load_experiment(self.run, epsilon=.01)
            _, other_nodes, other_boundaries = load_experiment(self.run, epsilon=.2)
        self.assertTrue(nodes.complete.all())
        self.assertEqual(len(nodes), len(self.cfg.H_grid) * len(self.cfg.mu_grid))
        self.assertEqual(len(boundaries), 2 * len(self.cfg.H_grid))
        pd.testing.assert_frame_equal(nodes, other_nodes)
        self.assertTrue(boundaries.epsilon.eq(.01).all())
        self.assertTrue(other_boundaries.epsilon.eq(.2).all())

    def test_analysis_marks_partial_run(self):
        def interrupt(event):
            if event['stage'] == 'test':
                raise RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            run_experiment(self.cfg, self.run, batch_size=40, M_validation=40, progress=interrupt)
        with patch.object(adapter, "simulate_paths", side_effect=AssertionError("analysis simulates")):
            _, nodes, boundaries = load_experiment(self.run, epsilon=.01)
        self.assertLess(len(nodes[nodes.complete]), len(self.cfg.H_grid) * len(self.cfg.mu_grid))
        self.assertTrue(boundaries.status.eq("incomplete_test").all())


if __name__ == "__main__":
    unittest.main()
