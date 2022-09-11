import typing

import gym
import numpy as np
import pandas as pd

from metrics import _combined_metrics, \
    _differential_sharpe_ratio, _profit_and_loss
from _plotter import _pnl_plot, cumulative_returns_plot, \
    _trade_plot, _rewards_plotter

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class TradeEnv(gym.Env):
    """
    **Class** to implement a basis-serving
    trading environment based on OpenAI Gym.
    """

    class Store:
        """
        **Class** to store rewards and actions performed by the agent.

        actions: pandas.DataFrame
        Table of actions performed by agent

        rewards: pandas.DataFrame
        Table of rewards received by agent
        """

        def __init__(self, index, columns):

            # actions
            self.actions = pd.DataFrame(
                columns=columns, index=index, dtype=np.float64)
            self.actions.iloc[0] = np.zeros(len(columns))

            # rewards
            self.rewards = pd.DataFrame(
                columns=columns, index=index, dtype=np.float64)
            self.rewards.iloc[0] = np.zeros(len(columns))

    def __init__(self,
                 prices: typing.Optional[pd.DataFrame],
                 returns: typing.Optional[pd.DataFrame],
                 unnorm_prices: typing.Optional[pd.DataFrame],
                 cash: bool, trading_cost: float):

        # trading cost
        self.trading_cost = trading_cost

        # prices
        self._prices = prices
        self._unnorm_prices = unnorm_prices
        self._returns = returns

        if cash:  # include cash or not
            self._prices["CASH"] = 0.0
            self._unnorm_prices["CASH"] = 1.0
            self._returns["CASH"] = 0.0

        # in case of 2 consecutive 0. prices, returns result in infinity
        # prices are normalized, so 0. signifies the lowest historical price
        self._returns = self._returns.replace([np.inf, -np.inf], 0)

        # reward lists initializers
        self.net_worth = [0]
        self.reward_list = [0]

        # Action and observation spaces
        tickers: int = len(self._prices.columns)

        self.action_space = gym.spaces.Box(low=0., high=1.,
                                           shape=(tickers,),
                                           dtype=np.float64)

        self.observation_space = gym.spaces.Box(-np.inf,
                                                np.inf,
                                                (tickers,),
                                                dtype=np.float64)

        # Initiate a counter to keep track of the index
        self._counter = 0

        # Agent's Profit and Loss
        self.agents = {}
        self._pnl = pd.DataFrame(index=self.dates, columns=[
            agent.name for agent in self.agents])

        self._fig, self._axes = None, None

    @property
    def dates(self):
        return self._prices.index

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.dates[self._counter]

    @property
    def _max_episode_steps(self):
        return len(self.dates)

    def _get_observation(self) -> object:

        ob = self._prices.loc[self.index, :]

        return ob

    def _get_reward(self, action):

        self._counter += 1

        self.net_worth.append(_profit_and_loss((self._unnorm_prices.pct_change().loc[self.index]
                                                * action * (1 - self.trading_cost)).sum())[0])
        net_worth_np = np.asarray(self.net_worth.copy())

        """DSR reward"""
        # https://github.com/AchillesJJ/DSR

        _reward = _differential_sharpe_ratio(net_worth=net_worth_np)

        """Simple returns reward"""
        # _reward = (self._unnorm_prices.pct_change().loc[self.index]
        #            * np.round(action, decimals=5) * (1 - self.trading_cost)).sum()

        """Log returns reward"""
        # https://quant.stackexchange.com/questions/60023/calculating-portfolio-log-returns

        # reward_t = np.log(np.sum(self._unnorm_prices.loc[self.index] * action * (1 - self.trading_cost)))
        # self.reward_list.append(reward_t)
        # _reward = reward_t - self.reward_list[-2]

        # algorithm simple returns for plotting
        alg_return = (self._unnorm_prices.pct_change().loc[self.index] * action).sum()

        return _reward, alg_return

    def _get_done(self) -> bool:

        return self.index == self.dates[-1]

    def _get_info(self) -> dict:
        return {}

    def _available_agents(self):
        """
        Check agents' availability.
        """

        if len(self.agents) == 0:
            raise RuntimeError('There is no agent registered in the environment')

    def register(self, agent):
        """
        Register an agent to the environment.
        """

        # verify interface
        if not hasattr(agent, 'name'):
            raise ValueError('Agent must have a name attribute.')

        # verify uniqueness
        if agent.name not in self.agents:
            self.agents[agent.name] = self.Store(
                columns=self._prices.columns.to_list(), index=self.dates)

    def step(self, action: typing.Union[object, typing.Dict[str, object]]):
        """
        The agent takes a step and observes the environment.
        """

        self._available_agents()

        observation = self._get_observation()
        info = self._get_info()

        # iterate over actions
        for A in action:
            # action validity check
            if action not in self.action_space:
                raise ValueError(
                    'The following invalid action was attempted: %s' % A
                )

        reward, alg_return = self._get_reward(action)

        done = self._get_done()

        return observation, reward, done, info, alg_return

    def reset(self, **kwargs) -> object:

        self._available_agents()

        # set time to zero
        self._counter = 0

        # get initial observation
        ob = self._get_observation()

        return ob

    def summary(self, prices, actions, rewards, episodes, ep_rewards):
        """
        Generate statistics summary and figures.
        Returns
        -------
        table: pd.DataFrame
            Strategy report.
        """
        summary = {}

        for agent in self.agents:

            weights = actions
            weights.name = agent

            rewards = rewards.squeeze().rename(agent)

            # Generate and save metrics in Excel file
            summary[agent] = _combined_metrics(rewards)

            # Figures ##

            # PnL
            _pnl_plot(rewards, prices)
            # Cumulative returns
            cumulative_returns_plot(rewards, prices)
            # Prices ~ Portfolio weights
            _trade_plot(prices, weights)
            # Rewards over episodes
            _rewards_plotter(episodes, ep_rewards)

        return summary
