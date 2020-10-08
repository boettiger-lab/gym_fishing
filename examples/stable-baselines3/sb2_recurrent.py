import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import gym_fishing
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=100, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[100, 'lstm', dict(vf=[200, 200, 200, 200], pi=[200, 200, 200, 200])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

env = make_vec_env('fishing-v1', n_envs=4)
model = PPO2(CustomLSTMPolicy, env, verbose=2, learning_rate=0.001, ent_coef=0.05, gamma=0.995)
model.learn(total_timesteps=int(2e6), log_interval=int(1e3))
model.save("sb2_ppo2_recurrent")
