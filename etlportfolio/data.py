import pandas as pd
import yfinance as yf


def get_prices(tickers_list: list, freq: str) -> pd.DataFrame:
    """Download price data for list of tickers at a given frequency.

    Returns a pd.DataFrame of prices for each asset.

    Parameters
    ----------
    tickers_list : list[str]
        list of assets to grab data for
    freq : str
        frequency desired i.e. '1D', '1W', '1M' etc.

    Returns
    -------
    pd.DataFrame
        DataFrame of pricing data at requested frequency.

    """
    price_data = yf.download(tickers_list, auto_adjust=True)["Close"]

    # convert to freq data
    price_data.dropna(inplace=True)
    price_data = price_data[tickers_list]  # reorder columns
    price_data = price_data.groupby(pd.Grouper(freq=freq)).last()

    return price_data
