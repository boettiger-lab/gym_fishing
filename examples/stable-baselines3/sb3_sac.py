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

env = gym.make('fishing-v1')
policy_kwargs = dict(activation_fn=th.nn.ReLU, 
                     net_arch=[100, 100, 100, 100, 100], 
                     log_std_init=.22)
model = SAC("MlpPolicy", 
            env, 
            verbose=2, 
            learning_rate=0.001,
            batch_size=16, 
            buffer_size=1000, 
            train_freq=8, 
            tau=0.01, 
            ent_coef=0.004, 
            gamma=0.995, 
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=int(1e6), log_interval=int(1e3))
model.save("sb3_sac")


