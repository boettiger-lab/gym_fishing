import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import gym_fishing
import numpy as np

from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import SAC
from tuning_utils_sb3 import *

def create_env(n_envs=0, eval_env=None):
    env = gym.make("fishing-v1")
    return env

def create_model(*_args, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return SAC(env=create_env(), policy=MlpPolicy, verbose=1, **kwargs)

hyperparam_optimization("sac", create_model, create_env, n_trials=20,
                                             n_timesteps=int(1e6),
                                             n_jobs=1, seed=0,
                                             sampler_method='tpe', pruner_method='median',
                                             verbose=1)
