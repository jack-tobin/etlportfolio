import unittest
from unittest.mock import patch

import pandas as pd
from etlportfolio.data import get_prices
from pandas.testing import assert_frame_equal


class TestGetPrices(unittest.TestCase):
    @patch("yfinance.download")
    def test_get_prices_daily(self, mock_download):
        # Mocking yfinance response
        tickers_list = ["AAPL", "MSFT"]
        freq = "1D"
        dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            data={
                ("Close", "AAPL"): [150, 152, 153, 155, 157],
                ("Close", "MSFT"): [210, 212, 215, 217, 219],
            },
            index=dates,
        )
        mock_download.return_value = data

        expected_output = data.xs("Close", axis=1).groupby(pd.Grouper(freq=freq)).last()

        result = get_prices(tickers_list, freq)
        assert_frame_equal(result, expected_output)

    @patch("yfinance.download")
    def test_get_prices_weekly(self, mock_download):
        # Mocking yfinance response
        tickers_list = ["AAPL", "MSFT"]
        freq = "1W"
        dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            data={
                ("Close", "AAPL"): [150, 152, 153, 155, 157],
                ("Close", "MSFT"): [210, 212, 215, 217, 219],
            },
            index=dates,
        )
        mock_download.return_value = data

        expected_output = data.xs("Close", axis=1).groupby(pd.Grouper(freq=freq)).last()

        result = get_prices(tickers_list, freq)
        assert_frame_equal(result, expected_output)
