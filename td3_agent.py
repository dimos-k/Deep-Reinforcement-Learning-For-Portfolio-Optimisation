import os

import pandas as pd
from gym import register
import numpy as np
import numpy.random
import torch
from datetime import datetime

from td3_replay_buffer import ReplayBuffer
from gym_env import TradeEnv
from td3 import TD3
from config import td3_config, train_universe

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")

startTime = datetime.now()

file_name = f"td3_TradeEnv_seed{td3_config['seed']}"
if not os.path.exists("./td3_results"):
    os.makedirs("./td3_results")
if td3_config['save_model'] and not os.path.exists("./td3_models"):
    os.makedirs("./td3_models")

# Construct TradeEnv environment and register it in Open AI gym
env = TradeEnv(prices=train_universe["prices"],
               returns=train_universe["returns"],
               unnorm_prices=train_universe["un-normalized prices"],
               cash=True, trading_cost=.001)

register(id='TradeEnv', entry_point='gym_env:TradeEnv')

# Extract state and actions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Set seeds
env.seed(td3_config['seed'])
env.action_space.seed(td3_config['seed'])  # action space needs seed setting \
# when sampling, for reproducibility
torch.manual_seed(td3_config['seed'])
np.random.seed(td3_config['seed'])

# Initialize policy
# Target policy smoothing is scaled wrt the action scale
kwargs = {"state_dim": state_dim, "action_dim": action_dim,
          "max_action": max_action,
          "discount": td3_config['discount'],
          "tau": td3_config['tau'],
          "policy_noise": td3_config['policy_noise'] * max_action,
          "noise_clip": td3_config['noise_clip'] * max_action,
          "policy_freq": td3_config['policy_freq']}

# Define policy
policy = TD3(**kwargs)

# Register td3Agent in TradeEnv
env.register(agent=policy)

# Initialize replay buffer
replay_buffer = ReplayBuffer(state_dim, action_dim)

if td3_config['load_model'] != "":
    policy_file = file_name \
        if td3_config['load_model'] == "trained" \
        else td3_config['load_model']
    policy.load(f"./td3_models/{policy_file}")


def _evaluate(seed, eval_episodes):
    """
    **Evaluates** policy over given episodes.

    :param seed: int
    :param eval_episodes: int, number of episodes
    :return: Average reward per episode.
    """

    eval_env = TradeEnv(prices=train_universe["evaluation prices"],
                        returns=train_universe["evaluation returns"],
                        unnorm_prices=train_universe["evaluation un-normalized prices"],
                        cash=True, trading_cost=.001)

    eval_env.register(agent=policy)

    _tmp_actions = pd.DataFrame()
    _tmp_rewards = pd.DataFrame(columns=["Sum of rewards"])
    total_reward = 0

    if td3_config['max_timesteps'] == 0:
        eval_episodes = 1
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _, alg_return = eval_env.step(action)

            if td3_config['max_timesteps'] == 0:
                _tmp_actions = pd.concat([_tmp_actions, pd.DataFrame(action.reshape(1, -1),
                                                                     index=[state.name], columns=state.index)], axis=0)

                _tmp_rewards = pd.concat([_tmp_rewards, pd.DataFrame(alg_return,
                                                                     index=[state.name], columns=["Sum of rewards"])],
                                         axis=0)

            total_reward += reward

    _tmp_rewards = _tmp_rewards.reindex(train_universe["evaluation prices"].index).shift(1)
    _tmp_rewards.iloc[0] = 0.0

    avg_reward = total_reward / eval_episodes

    print("\nValidation on different prices ...")
    print("-----------------------------------")
    print(f"~~ Avg. reward over {eval_episodes} episode(s): {avg_reward:.3f}, seed = {seed}. ~~")
    print("--------------------------------------------------------\n")

    # Export a strategy statistics summary with metrics and plots for the trained model
    if td3_config['max_timesteps'] == 0:
        eval_env.summary(prices=train_universe["evaluation prices"],
                         actions=_tmp_actions,
                         rewards=_tmp_rewards,
                         episodes=0,
                         ep_rewards=0)

    return avg_reward


# Evaluate untrained policy
evaluations = [_evaluate(seed=td3_config['seed'],
                         eval_episodes=10)]

# Initiating policy
state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

# To store rewards and actions after every timestep
temp_rewards = pd.DataFrame(columns=["Sum of rewards"])
temp_actions = pd.DataFrame()
ep_rewards = []

for t in range(int(td3_config['max_timesteps'])):

    episode_timesteps += 1

    # Select action randomly when below start time steps
    if t < td3_config['start_timesteps']:

        action = np.random.dirichlet(np.ones(action_dim))

    # Select action from policy when above start time steps
    else:

        action = (policy.select_action(np.array(state))).clip(0., max_action)

    # Perform action with step from TradeEnv
    next_state, reward, done, _, alg_return = env.step(action)

    done_bool = float(done) if episode_timesteps < td3_config['max_timesteps'] else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    # Train agent when surpassing start time steps
    if t >= td3_config['start_timesteps']:
        policy.train(replay_buffer=replay_buffer, batch_size=td3_config['batch_size'])

    # Finishing timestep
    state = next_state
    episode_reward += reward

    # Episode finishes upon reaching the last day of prices
    if done:
        # +1 to account for 0 indexing. +0 on episode time steps since it will increment +1 even if done=True
        print("\nEpisode statistics ...")
        print("-------------------------")
        print(f"Total Time steps: {t + 1}")
        print(f"Episode Num: {episode_num + 1}")
        print(f"Episode Time steps: {episode_timesteps}")
        print(f"Episode Reward: {episode_reward:.3f}")
        print("-------------------------\n")

        # Reset environment.
        # Episode finishes upon reaching last day of prices
        # A new episode starts from the 1st day again
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Evaluate trained policy for a fixed no. of episodes
    if (t + 1) % td3_config['eval_freq'] == 0:
        evaluations.append(_evaluate(seed=td3_config['seed'], eval_episodes=3))
        np.save(f"./td3_results/{file_name}", evaluations)
        if td3_config['save_model']:
            policy.save(f"./td3_models/{file_name}")

print("\nTime taken for completion:",
      datetime.now() - startTime)
