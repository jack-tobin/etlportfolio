#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test.py
@Time    :   2022/08/12 10:11:35
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""


import sys
import os
import numpy as np

# update path so etlportfolio is findable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

# testing import of etlportfolio object
from etlportfolio import data, portfolio

# gather pricing data from Yahoo! Finance.

# ETF assets
tickers = ['URTH', 'EEM', 'AGG', 'REET', 'DBC']

# download data
prices = data.get_prices(tickers, '1W')

# convert to returns
returns = prices.pct_change().fillna(0)

# make simple estimates of returns and covariance matrices
forecast_rets = np.mean(returns)
forecast_covs = np.cov(returns, rowvar=False)

# make an instance of class ETLPortfolio
etl = portfolio.ETLPortfolio()

# set risk aversion loss threshold
etl.k = -0.02

# feed in forecast data
etl.load_forecasts(tickers, forecast_rets, forecast_covs)

# get optimal weights
etl.optimise()
opt_weights = etl.weights
