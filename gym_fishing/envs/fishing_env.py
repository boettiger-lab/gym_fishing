import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import BaseFishingEnv


class FishingEnv(BaseFishingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 r=0.3,
                 K=1.0,
                 price=1.0,
                 sigma=0.01,
                 init_state=0.75,
                 init_harvest=0.0125,
                 Tmax=100,
                 n_actions=3,
                 file_=None
                 ):
        super().__init__(r, K, price, sigma, init_state, init_harvest, Tmax, file_)
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)

