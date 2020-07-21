import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
from gym import spaces
import gym_fishing
import numpy as np


import torch as th

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
output_string = ""
for i in range(1):
    env = gym.make('fishing-v1')
    model = SAC("MlpPolicy", env, verbose=2)
    model.learn(total_timesteps=int(1e5), log_interval=int(1e3))
    model.save(f"sb3_sac")
