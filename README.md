# etlportfolio
Expected tail loss portfolio optimisation in Python

## Description
This package contains portfolio optimisation class structures to perform alternative portfolio optimisation exercises. Included in the package are a mean-variance portfolio object, two common downside risk-oriented portfolio objects (CVaR and the lower partial moment), as well as an 'expected tail loss' (ETL) portfolio object based on the model of Geman et. al. (2015). Through these objects a user can implement downside risk optimised portfolios in a tractable fashion.

## Installation


## Usage

```python

# import modules
from etlportfolio import data, portfolio

# historical asset return data
tickers = [...]
returns = data.get_prices(tickers, '1W')

# simple historical estimates of returns and VCV, though this
# can be generated from any forecasting process.
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
etl.weights
```

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)

