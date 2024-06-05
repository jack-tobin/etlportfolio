import unittest

import numpy as np
from etlportfolio.portfolio import (
    CVaRPortfolio,
    ETLPortfolio,
    LPMPortfolio,
    MVOPortfolio,
    Portfolio,
)


# Dummy subclass to test abstract class
class DummyPortfolio(Portfolio):
    def risk_name(self):
        return "Dummy Risk"

    def risk_criterion(self, weights: np.ndarray) -> float:
        return np.sum(weights**2)

    def objective(self, weights: np.ndarray) -> float:
        return -np.sum(weights * self.forecast_returns)


class TestPortfolioOptimization(unittest.TestCase):
    def setUp(self):
        # Initialize common variables
        self.assets = ["Asset1", "Asset2", "Asset3"]
        self.forecast_returns = np.array([0.1, 0.2, 0.15])
        self.forecast_variance = np.array(
            [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]],
        )

    def test_portfolio_load_forecasts(self):
        portfolio = DummyPortfolio()
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )

        self.assertEqual(portfolio.assets, self.assets)
        self.assertEqual(portfolio.n_assets, len(self.assets))
        np.testing.assert_array_equal(portfolio.forecast_returns, self.forecast_returns)
        np.testing.assert_array_equal(
            portfolio.forecast_variance, self.forecast_variance,
        )

    def test_mv_portfolio_risk_criterion(self):
        portfolio = MVOPortfolio(lambda_=0.5)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        weights = np.array([0.4, 0.3, 0.3])
        risk = portfolio.risk_criterion(weights)
        expected_risk = np.sqrt(
            np.dot(weights.T, np.dot(self.forecast_variance, weights)),
        )
        self.assertAlmostEqual(risk, expected_risk)

    def test_mv_portfolio_objective(self):
        portfolio = MVOPortfolio(lambda_=0.5)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        weights = np.array([0.4, 0.3, 0.3])
        objective = portfolio.objective(weights)
        port_ret = np.dot(weights.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.forecast_variance, weights)))
        expected_objective = port_ret - (0.5 * port_vol**2 * 0.5)
        self.assertAlmostEqual(
            objective, -expected_objective,
        )  # Note: the objective function is negated

    def test_lpm_portfolio_risk_criterion(self):
        portfolio = LPMPortfolio(delta=0.5)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        weights = np.array([0.4, 0.3, 0.3])
        risk = portfolio.risk_criterion(weights)
        self.assertIsInstance(risk, float)

    def test_cvar_portfolio_risk_criterion(self):
        portfolio = CVaRPortfolio(zeta=0.5, theta=0.05)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        weights = np.array([0.4, 0.3, 0.3])
        risk = portfolio.risk_criterion(weights)
        self.assertIsInstance(risk, float)

    def test_etl_portfolio_risk_criterion(self):
        portfolio = ETLPortfolio(k=0.1)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        weights = np.array([0.4, 0.3, 0.3])
        risk = portfolio.risk_criterion(weights)
        self.assertIsInstance(risk, float)

    def test_optimize(self):
        portfolio = MVOPortfolio(lambda_=0.5)
        portfolio.load_forecasts(
            self.assets, self.forecast_returns, self.forecast_variance,
        )
        portfolio.optimise()
        self.assertIsNotNone(portfolio.weights)
        self.assertAlmostEqual(np.sum(portfolio.weights), 1, places=6)
        self.assertTrue(all(round(w, 4) >= 0 for w in portfolio.weights))
