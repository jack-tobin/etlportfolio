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
from typing_extensions import Self
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


class Portfolio(ABC):
    """
    Base class for non-traditional objective function portfolio optimisation.
    All child classes must specify a risk_criterion method that performs the
    computation of the risk criterion, and an objective method that serves
    as the objective function in the portfolio optimisation objective.
    """

    # str name of risk criterion.
    RISK_NAME = str

    def __init__(self) -> None:
        """
        Creates an instance of class Portfolio
        """

        # initiate ABC class
        super().__init__()

        # empty attributes
        self.assets = None
        self.n_assets = None
        self.weights = None
        self.forecast_returns = None
        self.forecast_variance = None

    def load_forecasts(self, assets: list, forecast_returns: np.ndarray,
                       forecast_variance: np.ndarray) -> Self:
        """
        Loads in forecast returns and variances, which will be used in the
        formulation of the objective function.

        Args:
            assets (list): List of asset names
            forecast_returns (np.ndarray): Array of asset forecast returns
            forecast_variance (np.ndarray): Array of asset covariance matrix

        Returns:
            Self: Instance of class Portfolio.
        """

        # extract aggregated simple return moments
        self.assets = assets
        self.n_assets = len(self.assets)
        self.forecast_returns = forecast_returns
        self.forecast_variance = forecast_variance

        return self

    @abstractmethod
    def risk_criterion(self, wts: np.ndarray) -> float:
        """
        Abstract method to compute a generic risk criterion given a set of
        weights wts.

        Args:
            wts (np.ndarray): Vector of portfolio weights

        Returns:
            float: The value of the risk criterion.
        """

    @abstractmethod
    def objective(self, wts: np.ndarray) -> float:
        """
        Abstract method to compute the portfolio objective function given a
        set of weights wts.

        Args:
            wts (np.ndarray): Vector of portoflio weights.

        Returns:
            float: The value of the portfolio objective function.
        """

    def optimise(self):
        """
        Performs constrained optimisation of portfolio objective function by
        changing weights vector. Constraints are that weights must be
        non-negative and sum to 100%.

        Warnings:
            If the objective result is not a success, the function will return
            a warning to notify the user of such result.

        Returns:
            Self: Instance of class Portfolio.
        """

        # constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x}]

        # minimize
        min_obj = minimize(fun=self.objective,
                           x0=np.tile(1 / self.n_assets, self.n_assets),
                           constraints=constraints,
                           options={'disp': False})

        # get solution
        self.weights = min_obj.x

        # raise warning if objective did not successfully finish
        if not min_obj.success:
            warnings.warn(min_obj.message)

        return self

    def frontier(self):
        """
        Generates a Monte Carlo efficient frontier of portfolio returns against
        the portfolio risk criterion. Uses random returns from a Dirichlet
        distribution.

        Returns:
            matplotlib.pyplot.axis: Plot figure axis.
        """

        # number of simulations
        n_sim = 10000

        # random weights
        weights = np.random.dirichlet(np.tile(1, self.n_assets), n_sim)

        # compute returns based on current mean_rets
        port_rets = np.dot(weights, self.forecast_returns)

        # compute risk based on objectives
        port_risks = np.apply_along_axis(
            self.risk_criterion, arr=weights, axis=1)

        # return / risk
        port_ret_risk = port_rets / port_risks

        # plot on scatterplot
        cmap = sns.color_palette('viridis', as_cmap=True)

        fig, ax = plt.subplots(1)
        ax.scatter(x=port_risks, y=port_rets, c=port_ret_risk, cmap=cmap)
        ax.set_ylabel('Portfolio return')
        ax.set_xlabel('Portfolio ' + self.RISK_NAME)
        fig.suptitle('Efficient frontier of {:,.0f} simulated portfolios'.format(n_sim))

        return ax


class MVOPortfolio(Portfolio):
    """
    Mean-Variance Optimisation class.

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

    # name of objective function
    RISK_NAME = 'Volatility'

    def __init__(self, lamb: float = None) -> None:
        """
        Create instance of class MVOPortfolio.

        Args:
            lamb (float, optional): Lambda risk aversion coefficient. Defaults to None.
        """

        # instantiate parent class
        super().__init__()

        # empty attributes
        self._lamb = lamb

    @property
    def lamb(self) -> float:
        """
        lamb property.

        Returns:
            float: Value of lambda risk aversion coefficient.
        """

        return self._lamb

    @lamb.setter
    def lamb(self, value: float):
        """
        Setter for lamb property.

        Args:
            value (float): Value to assign to lamb risk aversion coefficient.
        """

        self._lamb = value

    def risk_criterion(self, wts: np.ndarray) -> float:
        """
        Computes portfolio variance.

        Args:
            wts (np.ndarray): Vector of portfolio weights

        Returns:
            float: Portfolio variance given weights.
        """

        # compute variance
        port_var = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        return port_var

    def objective(self, wts: np.ndarray) -> float:
        """
        Portfolio objective function.

        Args:
            wts (np.ndarray): Vector of portfolio weights

        Returns:
            float: Portfolio utility value.
        """

        # compute portfolio return and variance
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = self.risk_criterion(wts)

        # portfolio objective to maximise
        port_util = port_ret - (0.5 * port_vol**2 * self.lamb)

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util


class LPMPortfolio(Portfolio):
    """
    Mean-Lower Partial Moment optimisation class.

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

    # name of risk criterion.
    RISK_NAME = 'Lower Partial Moment'

    def __init__(self, delta: float = None, power: int = 2, tau: float = 0) -> None:
        """
        Creates instance of class LPMPortfolio.

        Args:
            delta (float, optional): Delta risk aversion coefficient. Defaults to None.
            power (int, optional): Power to raise LPM to. Defaults to 2.
            tau (float, optional): Distribution partition threshold. Defaults to 0.
        """

        # instantiate parent class
        super().__init__()

        # empty attributes
        self._delta = delta
        self._tau = tau
        self._power = power

    @property
    def delta(self) -> float:
        """
        delta risk aversion property.

        Returns:
            float: value of delta risk aversion coefficient.
        """

        return self._delta

    @delta.setter
    def delta(self, value: float):
        """
        delta risk aversion property setter.

        Args:
            value (float): value to assign to delta risk aversion coefficient.
        """

        self._delta = value

    @property
    def tau(self) -> float:
        """
        tau distribution partition property.

        Returns:
            float: value of tau distribution partition value.
        """

        return self._tau

    @tau.setter
    def tau(self, value: float):
        """
        Setter for tau portfolio distribution partition property.

        Args:
            value (float): value to assign to tau distribution partition.
        """

        self._tau = value

    @property
    def power(self) -> int:
        """
        power property.

        Returns:
            int: value of power property.
        """

        return self._power

    @power.setter
    def power(self, value: int):
        """
        Setter for power property.

        Args:
            value (int): value to assign to power property.
        """

        self._power = value

    def risk_criterion(self, wts: np.ndarray) -> float:
        """
        Computes the lower partial moment risk criterion.

        Args:
            wts (np.ndarray): Vector of portfolio weights.

        Returns:
            float: value of lower partial moment.
        """

        # distribution moments
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        # LPM
        lpm = quad(lambda x: (self.tau - x)**self.power * norm.pdf(x, loc=port_ret, scale=port_vol),
                   -np.inf, self.tau)[0]

        return lpm

    def objective(self, wts: np.ndarray) -> float:
        """
        LPM portfolio objective function.

        Args:
            wts (np.ndarray): Vector of portfolio weights.

        Returns:
            float: value of portfolio utility
        """

        # compute portfolio return and variance
        port_ret = np.dot(wts.T, self.forecast_returns)

        # compute LPM
        lpm = self.risk_criterion(wts)

        # portfolio objective to maximise
        # if power is odd, then lpm is negative
        if self.power % 2 == 0:
            port_util = port_ret - self.delta * lpm
        else:
            port_util = port_ret + self.delta * lpm

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util


class CVaRPortfolio(Portfolio):
    """
    Mean-Conditional Value-at-Risk portfolio optimisation class.

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

    # name of objective function
    RISK_NAME = 'Conditional Value-at-Risk'

    def __init__(self, zeta: float = None, theta: float = 0.05,
                 power: int = 1) -> None:
        """
        Creates instance of class CVaRPortfolio

        Args:
            zeta (float, optional): Risk aversion coefficient. Defaults to None.
            theta (float, optional): Significance level i.e. 1 - alpha. Defaults to 0.05.
            power (int, optional): Power to raise CVaR to in objective function. Defaults to 1.
        """

        # instantiate parent class.
        super().__init__()

        # assign attributes
        self._zeta = zeta
        self._theta = theta
        self._power = power

    @property
    def zeta(self) -> float:
        """
        Zeta risk aversion coefficient property.

        Returns:
            float: value of zeta risk aversion coefficient.
        """

        return self._zeta

    @zeta.setter
    def zeta(self, value: float):
        """
        Setter for zeta

        Args:
            value (float): value to assign to zeta
        """

        self._zeta = value

    @property
    def theta(self) -> float:
        """
        Theta significance property.

        Returns:
            float: value of theta significance level
        """

        return self._theta

    @theta.setter
    def theta(self, value: float):
        """
        Setter for theta significance level

        Args:
            value (float): value to assign to theta.
        """

        self._theta = value

    @property
    def power(self) -> int:
        """
        Power property.

        Returns:
            int: value of power.
        """

        return self._power

    @power.setter
    def power(self, value: int):
        """
        Setter for power property.

        Args:
            value (int): value to assign to power.
        """

        self._power = value

    def risk_criterion(self, wts: np.ndarray) -> float:
        """
        Computation of conditional value at risk criterion.

        Args:
            wts (np.ndarray): Vector of portfolio weights.

        Returns:
            float: value of conditional value at risk.
        """

        # dist moments
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        # compute CVaR
        var = norm.ppf(q=self.theta, loc=port_ret, scale=port_vol)
        cvar = quad(lambda x: x * norm.pdf(x, loc=port_ret, scale=port_vol),
                    -np.inf, var)[0] / self.theta

        return cvar

    def objective(self, wts: np.ndarray) -> float:
        """
        Portfolio objective function for mean-cvar optimisation

        Args:
            wts (np.ndarray): vector of portfolio weights.

        Returns:
            float: portfolio utility value
        """

        # compute portfolio return and variance
        port_ret = np.dot(wts.T, self.forecast_returns)

        # compute cvar
        cvar = self.risk_criterion(wts)

        # portfolio objective to maximise
        # if n is even, then cvar is always positive. Otherwise it's negative
        if self.power % 2 == 0:
            port_util = port_ret - self.zeta * cvar**self.power
        else:
            port_util = port_ret + self.zeta * cvar**self.power

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util


class ETLPortfolio(Portfolio):
    """
    Mean-expected tail loss portfolio optimisation class.

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

    # name of objective function
    RISK_NAME = 'Expected Tail Loss'

    def __init__(self, k: float = None) -> None:
        """
        Creates instance of class ETLPortfolio.

        Args:
            k (float, optional): Maximum acceptable loss threshold. Defaults to None.
        """

        # instantiate parent class
        super().__init__()

        # assign attributes
        self._k = k

    @property
    def k(self) -> float:
        """
        K property.

        Returns:
            float: value of K
        """

        return self._k

    @k.setter
    def k(self, value: float):
        """
        Setter of K

        Args:
            value (float): value to set to K.
        """

        self._k = value

    def risk_criterion(self, wts: np.ndarray) -> float:
        """
        Computation of expected tail loss risk constraint.

        Args:
            wts (np.ndarray): Vector of portfolio weights.

        Returns:
            float: value of ETL
        """

        # dist moments
        port_ret = np.dot(wts.T, self.forecast_returns)
        port_vol = np.sqrt(np.dot(wts.T, np.dot(self.forecast_variance, wts)))

        # compute ETL
        etl = quad(lambda x: x * norm.pdf(x, loc=port_ret, scale=port_vol),
                   -np.inf, self.k)[0]

        return etl

    def objective(self, wts: np.ndarray) -> float:
        """
        Portfolio objective function for Mean-ETL optimisation

        Args:
            wts (np.ndarray): Vector of portfolio weights.

        Returns:
            float: Portfolio utility value.
        """

        # compute portfolio return and variance
        port_ret = np.dot(wts.T, self.forecast_returns)

        # compute etl
        etl = self.risk_criterion(wts)

        # portfolio objective to maximise
        port_util = port_ret + etl  # etl always negative

        # set to negative as goal is to minimise this objective function
        port_util *= -1

        return port_util
