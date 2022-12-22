#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   data.py
@Time    :   2022/06/29 22:49:17
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""


import yfinance as yf
import pandas as pd


def get_prices(tickers_list: list, freq: str) -> pd.DataFrame:
    """Download price data for list of tickers at a given frequency.
    
    Returns a pd.DataFrame of prices for each asset.

    Parameters
    ----------
    tickers_list : list
        List of assets to grab data for
    freq : str
        frequency desired i.e. '1D', '1W', '1M' etc.

    Returns
    -------
    pd.DataFrame
        DataFrame of pricing data at requested frequency.

    """
    price_data = yf.download(tickers_list, auto_adjust=True)['Close']
    price_data.dropna(inplace=True)
    price_data = price_data[tickers_list]
    price_data = price_data.groupby(pd.Grouper(freq=freq)).last()
    return price_data
