import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import seaborn as sns

from config import td3_config
from metrics import _profit_and_loss, _cumulative_returns
from tickers_inventory import _ticker_to_name
from fetch_data import SP500_returns_tr, SP500_returns_te, DJIA_returns_tr, DJIA_returns_te, \
    unnorm_train_Prices, unnorm_test_Prices, best_stock_tr, best_stock_te

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 10})


def _pnl_plot(returns, prices, path="statistics figures/"):
    _pnl_ = _profit_and_loss(returns)
    _pnl_[prices.index[0]] = 1.0
    _pnl_ = _pnl_.sort_index()
    _pnl_ = _pnl_.astype("float64")

    if td3_config['max_timesteps'] > 0:
        SP500_returns = SP500_returns_tr  # todo change
        DJIA_returns = DJIA_returns_tr
        best_stock = best_stock_tr
    elif td3_config['max_timesteps'] == 0:
        SP500_returns = SP500_returns_te  # todo change
        DJIA_returns = DJIA_returns_te
        best_stock = best_stock_te

    spy_pnl = _profit_and_loss(SP500_returns)
    spy_pnl[returns.index[0]] = 1
    spy_pnl = spy_pnl.sort_index()
    spy_pnl = spy_pnl.reindex(_pnl_.index)

    dia_pnl = _profit_and_loss(DJIA_returns)
    dia_pnl[returns.index[0]] = 1
    dia_pnl = dia_pnl.sort_index()
    dia_pnl = dia_pnl.reindex(_pnl_.index)

    PnL_best_stock = best_stock

    PnL_best_stock = pd.concat([PnL_best_stock, pd.DataFrame(1,
                                index=[prices.index[0]], columns=[best_stock.columns[0]])], axis=0)
    PnL_best_stock = PnL_best_stock.sort_index()
    PnL_best_stock = PnL_best_stock.squeeze('columns')

    sns.set_style("whitegrid", {'ax.grid': True, 'ax.edgecolor': 'black',
                                'grid.linestyle': ':', 'grid.linewidth': '2'})

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(_pnl_, label=f'{returns.name}', color='navy', linewidth=1.5)
    plt.plot(spy_pnl, label="Standard & Poor's 500", color='darkorange', linewidth=1.5)
    plt.plot(dia_pnl, label='Dow Jones Industrial Average', color='darkred', linewidth=1.5)

    ax.set(title=f'{returns.name or "Strategy"}: 'f'Profit & Loss',
           ylabel='Wealth Level')

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    plt.legend(frameon=True, fancybox=True, loc='best').set_draggable(True)

    plt.tight_layout()
    fig.savefig(path + "Profit & Loss" + ".png", format='png', dpi=200)
    plt.close()


def cumulative_returns_plot(returns, prices, path="statistics figures/"):
    cr = _cumulative_returns(returns)
    cr[prices.index[0]] = 0.0
    cr = cr.sort_index()
    cr = cr.astype("float64")

    if td3_config['max_timesteps'] > 0:
        SP500_returns = SP500_returns_tr
        DJIA_returns = DJIA_returns_tr
    elif td3_config['max_timesteps'] == 0:
        SP500_returns = SP500_returns_te
        DJIA_returns = DJIA_returns_te

    spy_cr = _cumulative_returns(SP500_returns)
    spy_cr[returns.index[0]] = 0.0
    spy_cr = spy_cr.sort_index()
    spy_cr = spy_cr.reindex(cr.index)

    dia_cr = _cumulative_returns(DJIA_returns)
    dia_cr[returns.index[0]] = 0.0
    dia_cr = dia_cr.sort_index()
    dia_cr = dia_cr.reindex(cr.index)

    fig, ax = plt.subplots(figsize=(12.4, 7))
    cr.index = pd.to_datetime(cr.index)
    plt.plot(cr.index, cr, label='TD3 Profit & Loss', color='navy', linewidth=1.5)
    plt.plot(cr.index, spy_cr, label="Standard & Poor's 500", color='darkorange', linewidth=1.5)
    plt.plot(cr.index, dia_cr, label='Dow Jones Industrial Average', color='darkred', linewidth=1.5)

    ax.set(title=f'{returns.name or "Strategy"}: '
                 f'Cumulative Returns',
           ylabel='Cumulative Returns')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    plt.legend()
    plt.style.use('seaborn-paper')
    plt.tight_layout()
    fig.savefig(path + "Cumulative Returns" + ".png", format='png', dpi=150)
    plt.close()


def _trade_plot(prices, weights, path="statistics figures/"):
    if td3_config['max_timesteps'] > 0:
        prices = unnorm_train_Prices

    elif td3_config['max_timesteps'] == 0:
        prices = unnorm_test_Prices

    for ticker in prices:
        if ticker != "CASH":
            _name = _ticker_to_name(ticker)
        if ticker == "CASH":
            _name = "Cash"

        fig, axes = plt.subplots(figsize=(6.8, 4), nrows=2, sharex='all', gridspec_kw={
            'height_ratios': [3.8, 1], 'wspace': 0.01})

        prices[ticker].index = pd.to_datetime(prices[ticker].index)
        weights[ticker].index = pd.to_datetime(weights[ticker].index)

        axes[0].plot(prices[ticker].index, prices[ticker].values, color='navy', linewidth=1.5)
        axes[1].bar(weights[ticker].index, weights[ticker].values, color='forestgreen', alpha=.6)
        axes[0].set(title='%s: Prices ~ Portfolio Weights'
                          % _name, ylabel='Price, $p_{t}$')
        axes[1].set(ylabel='Weight, $w_{t}$', ylim=[0, 1])
        fig.subplots_adjust(hspace=0.05)

        axes[1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[0].get_xaxis().set_visible(True)
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

        plt.style.use('seaborn-paper')
        plt.tight_layout()

        fig.savefig(path + str(ticker) + ".png", format='png', dpi=150)

        plt.close()


def _rewards_plotter(episodes, rewards, path="statistics figures/"):
    # Evolution of rewards only for the training process
    if td3_config['max_timesteps'] > 0:
        episodes = np.arange(1, episodes + 1)

        fig, ax = plt.subplots()
        plt.plot(episodes, rewards)

        ax.set(title="Rewards over time", xlabel="No. of episodes", ylabel="Rewards")
        plt.style.use('seaborn-paper')
        plt.tight_layout()

        fig.savefig(path + "Rewards ~ Episodes" + ".png", format='png')

        plt.close()


def _simulation_plot(df):

    fig, ax = plt.subplots(figsize=(9, 6))

    bins = np.linspace(-.1, .1, 100)
    for ticker in df.columns:
        name = _ticker_to_name(ticker)

        (mu_ticker, sigma_ticker) = norm.fit(df[ticker])
        x_ticker = norm.pdf(bins, mu_ticker, sigma_ticker)
        sigma_ticker = round(sigma_ticker, 2)
        plt.plot(bins, x_ticker, alpha=1, linewidth=1.4, label=f'{name} - {ticker}, ({sigma_ticker})')
        plt.style.use('seaborn-paper')
        plt.tight_layout()

    ax.set(ylabel='Probability density', xlabel='Daily returns (%)')

    plt.style.use('seaborn-paper')
    plt.legend()
    plt.tight_layout()

    plt.savefig('portfolio returns dist 2' + ".png", bbox_inches='tight', dpi=350, format='png')
    plt.show()


def _simple_plot(df: pd.DataFrame):
    """
    Plot and dist plot for prices or returns.

    Parameters
    ----------
    df:
        Prices or Returns.

    Returns
    -------
    plt, sns
        plots
    """
    # series should be dataframe e.g. daily_returns['AVAX-USD']

    fig, ax = plt.subplots()

    plt.plot(df, color='darkblue', alpha=0.4)
    fig.autofmt_xdate()
    plt.show()

    fig.savefig("Returns.png", format='png')

    # Plot returns distribution

    fig2, ax2 = plt.subplots()

    (mu, sigma) = norm.fit(df)
    skewness, kurt = skew(df), kurtosis(df)

    bins = np.linspace(-.04, .04, 100)
    plt.hist(df, bins=bins, density=True, facecolor='darkblue', alpha=0.4)

    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, color='red', linewidth=1.5)

    # plot
    plt.ylabel('Probability Density')
    plt.title(r'$\mathrm{Histogram\ of\ Returns:}\ \mu=%.3f,\ \sigma=%.3f ,\ skew=%.3f ,\ kurt=%.3f}$' % (
        mu, sigma, skewness, kurt), fontsize=10)
    plt.grid(True)

    fig2.savefig("Dist.png", format='png')
    plt.show()
