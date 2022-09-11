import pandas as pd
import numpy as np
import yfinance as yf

from metrics import _profit_and_loss, _cumulative_returns, _mean_returns, _std_returns, \
    _skewness, _kurtosis, _max_drawdown, _sharpe_ratio, _CVaR
from tickers_inventory import _tickers

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


def _yahoo_import(symbol: dict):
    """
    Downloads selected tickers from Yahoo.

    Parameters
    ----------
    symbol: dict
    Input as dictionary values

    Returns
    -------
    Daily Prices as Adjusted Close (includes dividends)
    """
    print("\nFetching prices ... ")
    data = yf.download(symbol, interval="1d", period="max")
    daily_prices = data['Adj Close']

    return daily_prices


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean `pandas.DataFrame` from
    **missing entries, np. inf, Nan**.

    Parameters
    ----------
    df: pandas.DataFrame
        Table to be cleaned.

    Returns
    -------
    df: pandas.DataFrame
        Cleaned df without infinities or Nan.
    """
    # remove infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop NaN values
    return df.dropna()


def _normalize(df):
    """
    Min-max prices normalizer.

    Parameters
    ----------
    df: pandas.Dataframe
        Prices frame to be normalized.

    Returns
    -------
    df: pandas.Dataframe
        Min-max normalized prices.
    """

    _min = df.min()
    _max = df.max()

    # time series normalization part
    y = (df - _min) / (_max - _min)

    return y


def log_normalize(df):
    """
    Log prices normalizer.

    Parameters
    ----------
    df: pandas.Dataframe
        Prices frame to be normalized.

    Returns
    -------
    df: pandas.Dataframe
        Log normalized prices.
    """

    # time series normalization part
    y = np.log(df) - np.log(df.shift(1))

    return y


# Import daily prices, max period
dailyPrices = _yahoo_import(_tickers).dropna()

""" Test and train sets """

# Prices
train_Prices = _normalize(dailyPrices.loc['2015-01-01':'2020-12-30']).dropna()
test_Prices = _normalize(dailyPrices.loc['2022-01-03':'2022-06-28']).dropna()

unnorm_train_Prices = dailyPrices.loc['2015-01-01':'2020-12-30']
unnorm_test_Prices = dailyPrices.loc['2022-01-01':'2022-07-25']

# Returns
train_Returns = _clean(dailyPrices['2015-01-01':'2020-12-31'].pct_change())
test_Returns = _clean(dailyPrices['2022-01-01':'2022-07-26'].pct_change())

# Log returns
daily_LogReturns = log_normalize(train_Prices).dropna()

# Evaluation
eval_prices = _normalize(dailyPrices.loc['2007-01-01':'2011-12-29']).dropna()
unnorm_eval_prices = dailyPrices.loc['2007-01-01':'2011-12-29']
eval_Returns = _clean(dailyPrices['2007-01-01':'2011-12-30'].pct_change())

# Benchmarks
SP500_prices = _yahoo_import('^GSPC')
SP500_returns_tr = _clean(SP500_prices.pct_change())[train_Prices.index]
SP500_returns_te = _clean(SP500_prices.pct_change())[test_Prices.index]

DJIA_prices = _yahoo_import('^DJI')
DJIA_returns_tr = _clean(DJIA_prices.pct_change())[train_Prices.index]
DJIA_returns_te = _clean(DJIA_prices.pct_change())[test_Prices.index]

best_stock_tr = _profit_and_loss(test_Returns).tail(1).idxmax(axis=1)
best_stock_tr = _profit_and_loss(test_Returns[best_stock_tr.values])

best_stock_te = _profit_and_loss(test_Returns).tail(1).idxmax(axis=1)
best_stock_te = _profit_and_loss(test_Returns[best_stock_te.values])

# Prints
print("CUM", _cumulative_returns(SP500_returns_te))
print("PNL", _profit_and_loss(SP500_returns_te))
print("MEAN", _mean_returns(SP500_returns_te))
print("STD", _std_returns(SP500_returns_te))
print("SKEW", _skewness(SP500_returns_te))
print("\n KURT", _kurtosis(SP500_returns_te))
print("SHARPE", _sharpe_ratio(SP500_returns_te))
print("MDD", _max_drawdown(SP500_returns_te))
print("CVAR", _CVaR(SP500_returns_te))
