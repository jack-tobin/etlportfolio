"""Portfolio optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
from etlportfolio.optimized.risk_criteria import compute_lpm_integrand, compute_cvar_integrand, compute_portfolio_moments


@dataclass
class Portfolio(ABC):
    """Portfolio.

    Base class for non-traditional objective function portfolio optimisation.
    All child classes must specify a risk_criterion method that performs the
    computation of the risk criterion, and an objective method that serves
    as the objective function in the portfolio optimisation objective.

    """

    assets: list[str] = field(init=False)
    n_assets: int = field(init=False)
    weights: np.ndarray = field(init=False)
    forecast_returns: np.ndarray = field(init=False)
    forecast_variance: np.ndarray = field(init=False)

    def load_forecasts(
        self,
        assets: list,
        forecast_returns: np.ndarray,
        forecast_variance: np.ndarray,
    ) -> None:
        """Load in forecast returns and variances.

        Forecast returns and variances  will be used in the formulation of the
        objective function.

        Parameters
        ----------
        assets : list[str]
            List of asset names
        forecast_returns : np.ndarray
            Array of asset forecast returns
        forecast_variance : np.ndarray
            Array of asset covariance matrix

        """
        self.assets = assets
        self.n_assets = len(self.assets)
        self.forecast_returns = forecast_returns
        self.forecast_variance = forecast_variance

    @property
    @abstractmethod
    def risk_name(self) -> str:
        """Return name of risk criterion."""

    @abstractmethod
    def risk_criterion(self, weights: np.ndarray) -> float:
        """Compute a generic risk criterion given a set of weights.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights

        """

    @abstractmethod
    def objective(self, weights: np.ndarray) -> float:
        """Compute the portfolio objective function given a set of weights.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """

    def optimise(self) -> None:
        """Optimize portfolio objective function.

        This is done by changing weights vector. Constraints are that weights
        must be non-negative and sum to 100%.

        Raises
        ------
        RuntimeError
            If the objective result is not a success, the function will return
            a warning to notify the user of such result.

        """
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "ineq", "fun": lambda x: x},
        ]

        optimized = minimize(
            fun=self.objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            options={"disp": False},
        )

        self.weights = optimized.x

        if not optimized.success:
            raise RuntimeError(f"Error in optimization: {optimized.message}")

    def frontier(self) -> plt.Axes:
        """Generate a Monte Carlo efficient frontier of returns against risk.

        Uses random returns from a Dirichlet distribution.

        """
        n_simulations = 10000
        generator = np.random.Generator(np.random.PCG64())
        weights = generator.dirichlet(np.tile(1, self.n_assets), n_simulations)

        portfolio_returns = np.dot(weights, self.forecast_returns)
        portfolio_risk = np.apply_along_axis(self.risk_criterion, arr=weights, axis=1)
        return_over_risk = portfolio_returns / portfolio_risk

        cmap = sns.color_palette("viridis", as_cmap=True)
        fig, ax = plt.subplots(1)
        ax.scatter(x=portfolio_risk, y=portfolio_returns, c=return_over_risk, cmap=cmap)
        ax.set_ylabel("Portfolio return")
        ax.set_xlabel("Portfolio " + self.risk_name)
        fig.suptitle(f"Efficient frontier of {n_simulations} simulated portfolios")

        return ax


@dataclass
class MVOPortfolio(Portfolio):
    """Mean-Variance Optimisation class.

    Computes a mean-variance optimised portfolio given a value of lambda, the
    risk aversion coefficient, and a set of forecast returns and covariances.

    Variance is defined as:

        Var(R) = w' Sigma w

    where w is a vector of portfolio weights and Sigma is a positive definite
    covariance matrix of asset returns.

    Mean-Variance objective is a quadratic utility maximisation function
    of the form:

        max_w U(w) = E(R) - 0.5 * lambda * Var(R)

    Class inherits from class Portfolio; required methods are risk_criterion
    which takes arguments: self, weights; and objective, which takes arguments:
    self, weights.

    """

    lambda_: float = field(default=None)

    @property
    def risk_name(self) -> str:
        return "Volatility"

    def risk_criterion(self, weights: np.ndarray) -> float:
        """Compute portfolio variance.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights

        """
        _, port_vol = compute_portfolio_moments(weights, self.forecast_returns, self.forecast_variance)
        return port_vol

    def objective(self, weights: np.ndarray) -> float:
        """Compute portfolio objective function.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights

        """
        port_ret = np.dot(weights.T, self.forecast_returns)
        port_vol = self.risk_criterion(weights)
        port_util = port_ret - (0.5 * port_vol**2 * self.lambda_)

        return port_util * -1


@dataclass
class LPMPortfolio(Portfolio):
    """Mean-Lower Partial Moment optimisation class.

    Performs a portfolio optimisation of mean returns against a lower partial
    moment risk criterion.

    LPM is defined as:

        LPM_n(tau) = int_{-inf}^tau (x - tau)^n f(x) dx

    where f(x) is the PDF of returns.

    Mean-LPM objective function is:

        max_w U(w) = E(R) - delta * LPM

    Class inherits from class Portfolio; required methods are risk_criterion
    which takes arguments: self, weights; and objective, which takes arguments:
    self, weights.

    """

    delta: float
    power: int = field(default=2)
    tau: float = field(default=0)

    @property
    def risk_name(self) -> str:
        return "Lower Partial Moment"

    def risk_criterion(self, weights: np.ndarray) -> float:
        """Compute the lower partial moment risk criterion.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret, port_vol = compute_portfolio_moments(weights, self.forecast_returns, self.forecast_variance)
        lpm_result = quad(
            lambda x: compute_lpm_integrand(x, self.tau, self.power, port_ret, port_vol),
            -np.inf,
            self.tau,
        )
        return lpm_result[0]

    def objective(self, weights: np.ndarray) -> float:
        """Compute LPM portfolio objective function.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret = np.dot(weights.T, self.forecast_returns)
        lpm = self.risk_criterion(weights)

        if self.power % 2 == 0:
            port_util = port_ret - self.delta * lpm
        else:
            port_util = port_ret + self.delta * lpm

        return port_util * -1


@dataclass
class CVaRPortfolio(Portfolio):
    """Mean-Conditional Value-at-Risk portfolio optimisation class.

    Performs a portfolio optimisation of mean returns over a conditional VaR
    risk criterion. CVaR is defined as:

        VaR = F^{-1}(theta)
        CVaR = int_{-inf}^VaR

    where F is the CDF of returns.

    Mean-CVaR objective is:

        max_w U(w) = E(R) - zeta * CVaR**power

    Note that the 'theta' attribute is the significance level, not the
    confidence level. That is, it is equivalent to "1 - alpha" where alpha
    is confidence. Say you want a 95% CVaR, in this case theta should be set
    to 1 - 0.95 = 0.05.

    Class inherits from class Portfolio; required methods are risk_criterion
    which takes arguments: self, weights; and objective, which takes arguments:
    self, weights.

    """

    zeta: float
    theta: float = field(default=0.05)
    power: int = field(default=1)

    @property
    def risk_name(self) -> str:
        return "Conditional Value-at-Risk"

    def risk_criterion(self, weights: np.ndarray) -> float:
        """Compute conditional value at risk criterion.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret, port_vol = compute_portfolio_moments(weights, self.forecast_returns, self.forecast_variance)
        var = norm.ppf(q=self.theta, loc=port_ret, scale=port_vol)
        cvar_result = quad(
            lambda x: compute_cvar_integrand(x, port_ret, port_vol),
            -np.inf,
            var,
        )
        return cvar_result[0] / self.theta

    def objective(self, weights: np.ndarray) -> float:
        """Compute objective function for mean-cvar optimisation.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret = np.dot(weights.T, self.forecast_returns)
        cvar = self.risk_criterion(weights)

        if self.power % 2 == 0:
            port_util = port_ret - self.zeta * cvar**self.power
        else:
            port_util = port_ret + self.zeta * cvar**self.power

        return port_util * -1


@dataclass
class ETLPortfolio(Portfolio):
    """Mean-expected tail loss portfolio optimisation class.

    Performs a portfolio optimisation of portfolio returns over an expected tail
    loss risk constraint. The expected tail loss constraint was proposed by
    Geman et. al. (2015).

    The ETL is defined as:

        ETL = Pr(X < K) * E(X | X < K)
            = int_{-inf}^k x f(x) dx

    where K is a user-defined risk aversion threshold of returns. It functions
    as a maximum acceptable loss level.

    The Mean_ETL objective is:

        max_w U(w) = E(R) - ETL

    Class inherits from class Portfolio; required methods are risk_criterion
    which takes arguments: self, weights; and objective, which takes arguments:
    self, weights.

    """

    k: float

    @property
    def risk_name(self) -> str:
        return "Expected Tail Loss"

    def risk_criterion(self, weights: np.ndarray) -> float:
        """Compute  expected tail loss risk constraint.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret, port_vol = compute_portfolio_moments(weights, self.forecast_returns, self.forecast_variance)
        etl_result = quad(
            lambda x: compute_cvar_integrand(x, port_ret, port_vol),
            -np.inf,
            self.k,
        )
        return etl_result[0]

    def objective(self, weights: np.ndarray) -> float:
        """Compute objective function for Mean-ETL optimisation.

        Parameters
        ----------
        weights : np.ndarray
            Vector of portfolio weights.

        """
        port_ret = np.dot(weights.T, self.forecast_returns)
        etl = self.risk_criterion(weights)
        port_util = port_ret + etl  # etl always negative

        return port_util * -1
