# dictionary used to fetch data.
# tickers compatible with Yahoo Finance

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")

cryptocurrencies = {

    'Bitcoin USD': 'BTC-USD',
    'Ethereum USD': 'ETH-USD',
    'Binance Coin': 'BNB-USD',
    'FTX Token USD': 'FTT-USD',
    'Cardano USD': 'ADA-USD',
    'Solana USD': 'SOL-USD',
    'Polkadot USD': 'DOT-USD',
    'Avalanche USD': 'AVAX-USD',
    'Kusama USD': 'KSM-USD',
    'Cosmos USD': 'ATOM-USD',
    'Litecoin USD': 'LTC-USD',
    'Ripple USD': 'XRP-USD',
    'Chainlink USD': 'LINK-USD',
    'Decentraland USD': 'MANA-USD',
    'The Sandbox USD': 'SAND-USD',
    'Monero USD': 'XMR-USD',
    'Aave USD': 'AAVE-USD',
    'Terra USD': 'LUNA1-USD',
    'Uniswap USD': 'UNI1-USD',
    'Polygon USD': 'MATIC-USD',
    'TRON USD': 'TRX-USD',
    'VeChain USD': 'VET-USD',
    'Fantom USD': 'FTM-USD',
    'Kava USD': 'KAVA-USD'

}

universe = {

    # Cryptocurrencies ##

    'Polkadot': 'DOT-USD',
    'Ethereum': 'ETH-USD',
    'Avalanche': 'AVAX-USD',
    'Solana': 'SOL-USD',
    'Cosmos': 'ATOM-USD',
    'Bitcoin': 'BTC-USD',
    'Cardano': 'ADA-USD',
    'Polygon': 'MATIC-USD',

    # Stocks ##

    'Microsoft Corporation': 'MSFT',
    'Exxon Mobil Corporation': 'XOM',
    'NVIDIA Corporation': 'NVDA',
    'Apple Inc.': 'AAPL',
    'Tesla, Inc.': 'TSLA',
    'Alphabet Inc.': 'GOOG',
    'Berkshire Hathaway Inc Class B': 'BRK-B',
    'Meta Platforms, Inc.': 'META',
    'JPMorgan Chase & Co.': 'JPM',
    'Johnson & Johnson': 'JNJ',
    'Amazon.com Inc.': 'AMZN',
    'Goldman Sachs Group Inc.': 'GS',
    'Boeing Co.': 'BA',
    'UnitedHealth Group Inc.': 'UNH',
    'Lockheed Martin Corporation ': 'LMT',
    'Moderna, Inc. ': 'MRNA',
    'The Coca-Cola Company': 'KO',
    'The Procter & Gamble Company': 'PG',
    'Advanced Micro Devices, Inc.': 'AMD',
    'Pfizer Inc.': 'PFE',

    # ETFs ##

    'Invesco QQQ Trust': 'QQQ',
    'Utilities Select Sector SPDR Fund': 'XLU',
    'iShares 0-5 Year TIPS Bond ETF': 'STIP',
    'iShares 20+ Year Treasury Bond ETF': 'TLT',
    'SPDR Portfolio S&P 500 High Dividend ETF': 'SPYD',
    'Vanguard Developed Markets Index Fund': 'VEA',
    'SPDR S&P 500 ETF Trust': 'SPY',
    'SPDR Dow Jones Industrial Average ETF Trust': 'DIA',

    # Commodities ##

    'United States Oil Fund, LP': 'USO',
    'SPDR Gold Shares': 'GLD',
}

portfolio = {
    'Apple Inc.': 'AAPL',
    'Boeing Co.': 'BA',
    'Lockheed Martin Corporation ': 'LMT',
    'Exxon Mobil Corporation': 'XOM',
    'UnitedHealth Group Inc.': 'UNH',
    'The Coca-Cola Company': 'KO'
}

_tickers = list(portfolio.values())
uni_tickers = list(universe.values())
cryptocurrencies_tickers = list(cryptocurrencies.values())


def _ticker_to_name(ticker) -> str:
    """
    Searches the portfolio dictionaries and
    returns name of asset (dict.keys()) as a string when provided a ticker (dict.values()).
    _name and ticker must be of the form dict.keys() and dict.values() respectively.

    :param ticker: dict.values()
    :return: _name: dict.keys() as a string
    """
    if ticker in universe.values():
        _portfolio = universe
    elif ticker in cryptocurrencies.values():
        _portfolio = cryptocurrencies
    elif ticker in portfolio.values():
        _portfolio = portfolio

    else:
        raise ValueError(f" '{ticker}' is not included in any of the portfolios.")

    _name = list(_portfolio.keys()) \
        [list(_portfolio.values()).index(ticker)]

    return _name
