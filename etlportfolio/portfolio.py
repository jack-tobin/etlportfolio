#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   portfolio.py
@Time    :   2022/08/09 15:23:26
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""


from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
from seaborn import color_palette


class Portfolio(ABC):
    """Portfolio base class.

    Base class for non-traditional objective function portfolio optimisation.
    All child classes must specify a risk_criterion method that performs the
    computation of the risk criterion, and an objective method that serves
    as the objective function in the portfolio optimisation objective.

    """

    __risk_name__: str

    def __init__(self) -> None:
        """Construct instance of class Portfolio."""
        super().__init__()

        # empty attributes
        self._assets = None
        self.n_assets = None
        self.weights = None
        self._forecast_returns = None
        self._forecast_variance = None

    @property
    def assets(self) -> int:
        """Assets in portfolio."""
        return self._assets

    @assets.setter
    def assets(self, new_assets: np.ndarray) -> None:
        self._assets = new_assets
        self.n_assets = len(self._assets)

    @property
    def forecast_returns(self) -> np.ndarray:
        """Forecast returns."""
        return self._forecast_returns

    @forecast_returns.setter
    def forecast_returns(self, new_forecast_returns: np.ndarray) -> None:
        if self.assets is None:
            raise ValueError('Attempting to set forecast_returns without having set assets.')
        self._forecast_returns = new_forecast_returns

    @property
    def forecast_variance(self) -> np.ndarray:
        """Forecast variances."""
        return self._forecast_variance

    @forecast_variance.setter
    def forecast_variance(self, new_forecast_variance: np.ndarray) -> None:
        if self.assets is None:
            raise ValueError('Attempting to set _forecast_variance without having set assets.')
        self._forecast_variance = new_forecast_variance

    @abstractmethod
    def risk_criterion(self, wts: np.ndarray) -> float:
        """Compute generic risk criterion given a set of weights.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights

        Returns
        -------
        float
            The value of the risk criterion.

        """

    @abstractmethod
    def objective(self, wts: np.ndarray) -> float:
        """Compute the portfolio objective function value given weights.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portoflio weights.

        Returns
        -------
        float
            The value of the portfolio objective function.

        """

    def optimise(self):
        """Perform constrained optimisation of portfolio objective function.
        
        This is done by changing weights vector. Constraints are that
        weights must be non-negative and sum to 100%.

        Warnings
        --------
        If the objective result is not a success, the function will return
        a warning to notify the user of such result.

        Returns
        -------
        Portfolio
            Instance of class Portfolio.

        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x}]

        min_obj = minimize(fun=self.objective,
                           x0=np.tile(1 / self.n_assets, self.n_assets),
                           constraints=constraints,
                           options={'disp': False})

        self.weights = min_obj.x

        if not min_obj.success:
            warnings.warn(min_obj.message)

        return self

    def frontier(self):
        """Generate a Monte Carlo efficient frontier of portfolios.
        
        Plots returns against the portfolio risk criterion. Uses random
        returns from a Dirichlet distribution.

        Returns
        -------
        matplotlib.pyplot.axis
            Plot figure axis.

        """
        n_sim = 10000

        # Simulate n_sim portfolios.
        weights = np.random.dirichlet(np.tile(1, self.n_assets), n_sim)
        port_rets = np.dot(weights, self.forecast_returns)
        port_risks = np.apply_along_axis(
            self.risk_criterion, arr=weights, axis=1)
        port_ret_risk = port_rets / port_risks

        # plot on scatterplot
        cmap = color_palette('viridis', as_cmap=True)
        fig, ax = plt.subplots(1)
        ax.scatter(x=port_risks, y=port_rets, c=port_ret_risk, cmap=cmap)
        ax.set_ylabel('Portfolio return')
        ax.set_xlabel('Portfolio ' + self.__risk_name__)
        fig.suptitle('Efficient frontier of {:,.0f} simulated portfolios'.format(n_sim))

        return ax


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
    which takes arguments: self, wts; and objective, which takes arguments:
    self, wts.
    """

    ____risk_name____ = 'Volatility'

    def __init__(self, lambda_: float = None) -> None:
        """Create instance of class MVOPortfolio.

        Parameters
        ----------
        lambda_ : float
            Lambda risk aversion coefficient, by default None.

        """
        super().__init__()
        self._lambda_ = lambda_

    @property
    def lambda_(self) -> float:
        """Lambda risk aversion coefficient."""
        return self._lambda_

    @lambda_.setter
    def lambda_(self, new_lambda_: float) -> None:
        self._lambda_ = new_lambda_

    def risk_criterion(self, wts: np.ndarray) -> float:
        """Compute portfolio variance.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights

        Returns
        -------
        float
            Portfolio variance given weights.

        """
        return np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

    def objective(self, wts: np.ndarray) -> float:
        """
        Portfolio objective function.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights

        Returns
        -------
        float
            Portfolio utility value.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = self.risk_criterion(wts)
        port_util = port_ret - (0.5 * port_vol**2 * self.lamb)

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util


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
    which takes arguments: self, wts; and objective, which takes arguments:
    self, wts.
    """

    __risk_name__ = 'Lower Partial Moment'

    def __init__(self, delta: float = None, power: int = 2, tau: float = 0) -> None:
        """Create instance of class LPMPortfolio.

        Parameters
        ----------
        delta : float
            Delta risk aversion coefficient. Defaults to None.
        power : int
            Power to raise LPM to. Defaults to 2.
        tau : float
            Distribution partition threshold. Defaults to 0.

        """
        super().__init__()

        self._delta = None
        self._tau = None
        self._power = None

        self.delta = delta
        self.tau = tau
        self.power = power

    @property
    def delta(self) -> float:
        """Delta risk aversion coefficient."""
        return self._delta

    @delta.setter
    def delta(self, new_delta: float) -> None:
        self._delta = new_delta

    @property
    def tau(self) -> float:
        """Tau distribution partition parameter."""
        return self._tau

    @tau.setter
    def tau(self, new_tau: float) -> None:
        self._tau = new_tau

    @property
    def power(self) -> int:
        """Power parameter."""
        return self._power

    @power.setter
    def power(self, new_power: int) -> None:
        self._power = new_power

    def risk_criterion(self, wts: np.ndarray) -> float:
        """Compute the lower partial moment risk criterion.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            Value of lower partial moment.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        lpm = quad(lambda x: (self.tau - x)**self.power * norm.pdf(x, loc=port_ret, scale=port_vol),
                   -np.inf, self.tau)[0]

        return lpm

    def objective(self, wts: np.ndarray) -> float:
        """Compute LPM portfolio objective function.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            Value of portfolio utility

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        lpm = self.risk_criterion(wts)

        # portfolio objective to maximise
        # if power is odd, then lpm is negative
        factor = -1 if self.power % 2 == 0 else 1
        port_util = port_ret + (factor * self.delta * lpm)

        # set to negative as scipy only minimises
        port_util *= -1

        return port_util


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
    which takes arguments: self, wts; and objective, which takes arguments:
    self, wts.
    """

    __risk_name__ = 'Conditional Value-at-Risk'

    def __init__(self, zeta: float = None, theta: float = 0.05,
                 power: int = 1) -> None:
        """Create instance of class CVaRPortfolio

        Parameters
        ----------
        zeta : float
            Risk aversion coefficient. Defaults to None.
        theta : float
            Significance level i.e. 1 - alpha. Defaults to 0.05.
        power : int
            Power to raise CVaR to in objective function. Defaults to 1.

        """
        super().__init__()

        self._zeta = None
        self._theta = None
        self._power = None
        
        self.zeta = zeta
        self.theta = theta
        self.power = power

    @property
    def zeta(self) -> float:
        """Zeta risk aversion coefficient."""
        return self._zeta

    @zeta.setter
    def zeta(self, new_zeta: float) -> None:
        self._zeta = new_zeta

    @property
    def theta(self) -> float:
        """Theta significance level."""
        return self._theta

    @theta.setter
    def theta(self, new_theta: float) -> None:
        self._theta = new_theta

    @property
    def power(self) -> int:
        """Power parameter."""
        return self._power

    @power.setter
    def power(self, new_power: int) -> None:
        self._power = new_power

    def risk_criterion(self, wts: np.ndarray) -> float:
        """Compute conditional value at risk criterion.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            Value of conditional value at risk.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        var = norm.ppf(q=self.theta, loc=port_ret, scale=port_vol)
        cvar = quad(lambda x: x * norm.pdf(x, loc=port_ret, scale=port_vol),
                    -np.inf, var)[0] / self.theta

        return cvar

    def objective(self, wts: np.ndarray) -> float:
        """Compute portfolio objective function for mean-cvar optimisation.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            portfolio utility value.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        cvar = self.risk_criterion(wts)

        # portfolio objective to maximise
        # if n is even, then cvar is always positive. Otherwise it's negative
        factor = -1 if self.power % 2 == 0 else 1
        port_util = port_ret + (factor * self.zeta * cvar**self.power)

        # set to negative as scipy only minimises
        port_util *= -1

        return port_util


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
    which takes arguments: self, wts; and objective, which takes arguments:
    self, wts.
    """

    __risk_name__ = 'Expected Tail Loss'

    def __init__(self, kappa: float = None) -> None:
        """Create instance of class ETLPortfolio.

        Parameters
        ----------
        kappa : float
            Maximum acceptable loss threshold. Defaults to None.
        """

        # instantiate parent class
        super().__init__()

        # assign attributes
        self._kappa = kappa

    @property
    def kappa(self) -> float:
        """Kappa risk aversion coefficient."""
        return self._kappa

    @kappa.setter
    def kappa(self, new_kappa: float) -> None:
        self._kappa = new_kappa

    def risk_criterion(self, wts: np.ndarray) -> float:
        """Compute expected tail loss risk constraint.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            Value of ETL.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        # compute ETL
        etl = quad(lambda x: x * norm.pdf(x, loc=port_ret, scale=port_vol),
                   -np.inf, self.k)[0]

        return etl

    def objective(self, wts: np.ndarray) -> float:
        """Compute portfolio objective function for Mean-ETL optimisation.

        Parameters
        ----------
        wts : np.ndarray
            Vector of portfolio weights.

        Returns
        -------
        float
            Portfolio utility value.

        """
        port_ret = np.dot(wts.T, self.forecast_returns)
        etl = self.risk_criterion(wts)

        # portfolio objective to maximise. ETL always negative
        port_util = port_ret + etl

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util
