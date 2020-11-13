import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import BaseFishingEnv


class FishingModelError(BaseFishingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K_mean=1.0,
                 r_mean=0.3,
                 price=1.0,
                 sigma=0.02,
                 sigma_p=0.1,
                 init_state=0.75,
                 init_harvest=0.0125,
                 Tmax=100,
                 file_=None):
        super().__init__(price=price, 
                         sigma=sigma, 
                         init_state=init_state, 
                         init_harvest=init_harvest, 
                         Tmax=Tmax, 
                         file_=file_)
        self.K_mean = K_mean
        self.r_mean = r_mean
        self.sigma_p = sigma_p
        self.K = np.clip(np.random.normal(self.K_mean, self.sigma_p), 0, 1e6)
        self.r = np.clip(np.random.normal(self.r_mean, self.sigma_p), 0, 1e6)

    def reset(self):
        self.K = np.clip(np.random.normal(self.K_mean, self.sigma_p), 0, 1e6)
        self.r = np.clip(np.random.normal(self.r_mean, self.sigma_p), 0, 1e6)
        self.fish_population = np.array([self.init_state])
        self.harvest = self.init_harvest
        self.action = 0
        self.years_passed = 0
        return self.fish_population
