import numpy as np
import pandas as pd

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Machine limits for floating point types.
# The difference between 1.0 and the next smallest representable
# float larger than 1.0 (approximately 2.22e-16).
epsilon = np.finfo(float).eps


def _sharpe_ratio(returns: [pd.DataFrame]) -> pd.DataFrame:

    """
    :parameter returns: pandas.Dataframe
    :return: Sharpe ratio: pandas.Dataframe
    """
    sr = (np.sqrt(len(returns)) * np.mean(returns)) / (np.std(returns) + epsilon)

    return sr


def _differential_sharpe_ratio(net_worth):

    """
    :param net_worth:
    :return: Differential Sharpe Ratio
    """

    nw_change = np.nan_to_num(np.diff(net_worth) / net_worth[:-1], posinf=0, neginf=0)

    A = np.mean(nw_change)
    B = np.mean(nw_change ** 2)

    delta_A = nw_change[-1] - A
    delta_B = nw_change[-1] ** 2 - B

    dsr = (B * delta_A - 0.5 * A * delta_B) / (B - A ** 2) ** (3 / 2)

    return np.nan_to_num(dsr)


def _max_drawdown(returns):

    """
    Maximum Drawdown (MDD).
    See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

    Note: Intended for single-column frames

    :param returns: pandas.Dataframe
    :return: Maximum Drawdown: pandas.Dataframe
    """

    s = (returns + 1).cumprod()
    mdd = np.ptp(s) / s.max()

    return mdd


def _cumulative_returns(returns):

    cum_returns = returns.copy()
    cum_returns = np.add(cum_returns, 1)
    cum_returns = cum_returns.cumprod(axis=0)
    cum_returns = np.subtract(cum_returns, 1)

    return cum_returns


def _profit_and_loss(returns):

    pnl = returns.copy()
    pnl = np.add(pnl, 1)
    pnl = pnl.cumprod(axis=0)

    return pnl


def _mean_returns(returns):

    mean = returns.mean(axis=0)

    return mean


def _std_returns(returns):

    std = returns.std(axis=0)

    return std


def _skewness(returns):

    skew = returns.skew(axis=0)

    return skew


def _kurtosis(returns):

    kurt = returns.kurt(axis=0)

    return kurt


def _win_to_loss(returns):

    average_win = returns[returns > 0].mean(axis=0)
    average_loss = returns[returns < 0].mean(axis=0)
    wtl = np.abs((average_win + epsilon) / (average_loss + epsilon))

    return wtl


def _profitability_per_trade(returns):

    sum_win = np.sum(returns > 0) / len(returns)
    sum_loss = np.sum(returns < 0) / len(returns)
    average_win = returns[returns > 0].mean(axis=0)
    average_loss = returns[returns < 0].mean(axis=0)

    ppt = (sum_win * average_win) - (sum_loss * average_loss)

    return ppt


def _CVaR(returns):

    # 99% Normal CVaR
    var = np.percentile(returns, 100-99)
    cvar = np.mean(returns[returns <= var])

    return cvar


def _combined_metrics(returns):

    frame = {
        "Cumulative returns": _cumulative_returns(returns),
        "Profit and Loss": _profit_and_loss(returns),
        "Average return": _mean_returns(returns),
        "Standard deviation of returns": _std_returns(returns),
        "Skewness of returns": _skewness(returns),
        "Kurtosis of returns": _kurtosis(returns),
        "Sharpe ratio": _sharpe_ratio(returns),
        "Max Drawdown": _max_drawdown(returns),
        "Average returns to average losses ratio": _win_to_loss(returns),
        "Profitability per trade": _profitability_per_trade(returns),
        "Conditional Value at Risk -- 99%": _CVaR(returns)
    }

    stats = pd.DataFrame.from_dict(frame)
    stats.to_excel(excel_writer="statistics/stats.xlsx")

    return stats
