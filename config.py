# Policy Name: Twin Delayed Deep Deterministic Policy Gradients (TD3)
# This dictionary contains all args necessary to initiate TD3 and td3_agent

from fetch_data import train_Prices, train_Returns, test_Prices, test_Returns, \
    unnorm_train_Prices, unnorm_test_Prices, \
    eval_prices, eval_Returns, unnorm_eval_prices

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Twin Delayed Deep Deterministic Policy Gradients (TD3) configuration dictionary

# The model is tested as it trains.
# When loading a saved model, the model is evaluated before training.
# To test a trained model, set "load_model" to "trained" and max_timesteps to 0.

td3_config = {

    "seed": 12,  # Sets Gym, PyTorch and Numpy seeds
    "start_timesteps": 0 * (len(train_Prices) - 1),  # random policy time steps
    "eval_freq": 0 * (len(train_Prices) - 1),  # How often we evaluate
    "max_timesteps": 0 * (len(train_Prices) - 1),  # Max steps to run environment
    "expl_noise": 0,  # Std of Gaussian exploration noise added to action
    "buffer_size": int(5e4),  # experience replay buffer size
    "batch_size": 32,  # Batch size for replay buffer
    "discount": .999,  # Discount factor
    "actor_learning_rate": 1e-4,   # -Adam- optimizer learning rate for Actor
    "critic_learning_rate": 1e-4,  # -Adam- optimizer learning rate for Critic
    "tau": .001,  # Target network update rate
    "policy_noise": .1,  # Noise added to target policy during critic update
    "noise_clip": .2,  # Range to clip target policy noise
    "policy_freq": 2,  # Frequency of delayed policy updates
    "save_model": True,  # Option to save trained model and optimizer parameters
    "load_model": "trained"  # Load model,
    # "" doesn't load, "trained" uses saved file_name from the td3_models folder

}

# Training vs. testing configuration dictionaries

train_universe = {
    "prices": train_Prices,
    "un-normalized prices": unnorm_train_Prices,
    "returns": train_Returns,

    # Validation
    "evaluation prices": eval_prices,
    "evaluation un-normalized prices": unnorm_eval_prices,
    "evaluation returns": eval_Returns
}

test_universe = {
    "prices": test_Prices,
    "un-normalized prices": unnorm_test_Prices,
    "returns": test_Returns
}

if td3_config['max_timesteps'] == 0:

    train_universe["evaluation prices"] = test_universe["prices"]
    train_universe["evaluation un-normalized prices"] = test_universe["un-normalized prices"]
    train_universe["evaluation returns"] = test_universe["returns"]
