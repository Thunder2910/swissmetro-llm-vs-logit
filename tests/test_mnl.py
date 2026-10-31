"""Checks for the logit likelihood and the Swissmetro benchmark."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BIOGEME_B01A, DATA_PATH
from src.data_utils import availability_matrix, ensure_data, estimation_sample, load_raw
from src.mnl import (
    SPEC_BASE,
    design_matrix,
    fit_mnl,
    loglik_and_derivatives,
    null_loglik,
)


class TestDerivatives(unittest.TestCase):
    def test_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        n, k = 40, 4
        x = rng.normal(size=(n, 3, k))
        # Make it look like ASCs plus two attributes so columns vary.
        avail = np.ones((n, 3), dtype=bool)
        avail[:5, 2] = False
        theta = rng.normal(scale=0.3, size=k)
        v = np.einsum("njk,k->nj", x, theta)
        v = np.where(avail, v, -np.inf)
        m = np.max(np.where(np.isfinite(v), v, -1e100), axis=1, keepdims=True)
        e = np.exp(np.where(avail, v - m, -1e300))
        p = e / e.sum(axis=1, keepdims=True)
        draws = rng.random(n)
        cdf = np.cumsum(p, axis=1)
        y = (draws[:, None] > cdf).sum(axis=1).astype(int)

        theta0 = np.zeros(k)
        _, grad, hess, _ = loglik_and_derivatives(theta0, x, avail, y, True)
        eps = 1e-6
        num = np.zeros(k)
        for j in range(k):
            step = np.zeros(k)
            step[j] = eps
            lp, _, _, _ = loglik_and_derivatives(theta0 + step, x, avail, y, False)
            lm, _, _, _ = loglik_and_derivatives(theta0 - step, x, avail, y, False)
            num[j] = (lp - lm) / (2 * eps)
        self.assertTrue(np.allclose(grad, num, atol=1e-5))

        # One column of the Hessian versus finite differences of the gradient.
        num_h = np.zeros(k)
        for j in range(k):
            step = np.zeros(k)
            step[j] = eps
            _, gp, _, _ = loglik_and_derivatives(theta0 + step, x, avail, y, False)
            _, gm, _, _ = loglik_and_derivatives(theta0 - step, x, avail, y, False)
            num_h[j] = (gp[0] - gm[0]) / (2 * eps)
        self.assertTrue(np.allclose(hess[0], num_h, atol=1e-4))


class TestSwissmetroBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = load_raw(ensure_data(DATA_PATH))
        cls.df = estimation_sample(raw)

    def test_sample_counts_match_biogeme(self) -> None:
        df = self.df
        self.assertEqual(len(df), BIOGEME_B01A["n_obs"])
        self.assertEqual(int((df["CHOICE"] == 1).sum()), 908)
        self.assertEqual(int((df["CHOICE"] == 2).sum()), 4090)
        self.assertEqual(int((df["CHOICE"] == 3).sum()), 1770)
        avail = availability_matrix(df)
        self.assertEqual(int(avail[:, 0].sum()), 6768)
        self.assertEqual(int(avail[:, 1].sum()), 6768)
        self.assertEqual(int(avail[:, 2].sum()), 5607)
        self.assertAlmostEqual(null_loglik(avail), BIOGEME_B01A["null_loglik"], places=2)
        # GA holders do not pay a trip fare in the estimation variables.
        ga = df["GA"].to_numpy() == 1
        self.assertTrue(np.all(df.loc[ga, "train_cost"].to_numpy() == 0))
        self.assertTrue(np.all(df.loc[ga, "sm_cost"].to_numpy() == 0))
        self.assertGreater(float(df.loc[ga, "TRAIN_CO"].median()), 100)

    def test_base_model_matches_published_estimates(self) -> None:
        result = fit_mnl(self.df, SPEC_BASE)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.loglik, BIOGEME_B01A["loglik"], places=2)
        self.assertAlmostEqual(result.null_loglik, BIOGEME_B01A["null_loglik"], places=2)
        by_name = dict(zip(result.names, result.coef))
        for name, published in BIOGEME_B01A["coef"].items():
            self.assertAlmostEqual(by_name[name], published, places=3)
        # Zero parameter vector is the equal-shares null.
        x, _ = design_matrix(self.df, SPEC_BASE)
        avail = availability_matrix(self.df)
        ll0, _, _, _ = loglik_and_derivatives(
            np.zeros(len(result.names)), x, avail, self.df["y"].to_numpy(), False
        )
        self.assertAlmostEqual(ll0, result.null_loglik, places=6)


if __name__ == "__main__":
    unittest.main()
