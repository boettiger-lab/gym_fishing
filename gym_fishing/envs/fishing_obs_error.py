
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import BaseFishingEnv


class FishingObsError(BaseFishingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K=1.0,
                 r=0.3,
                 price=1.0,
                 sigma=0.02,
                 sigma_m=0.02,
                 init_state=0.75,
                 init_harvest=0.0125,
                 Tmax=100,
                 file_=None):

        super().__init__(r, K, price, sigma, init_state, init_harvest, Tmax, file_)
        self.sigma_m = sigma_m
        self.observed = observation_noise(self.fish_population[0],
                                          self.sigma_m,
                                          self.observation_space)

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.harvest = (action + 1) * self.K

        self.harvest_draw(self.harvest)
        self.population_draw()
        self.observed = observation_noise(self.fish_population[0],
                                          self.sigma_m,
                                          self.observation_space)

        # should be the instanteous reward, not discounted
        reward = max(self.price * self.harvest, 0.0)
        self.reward = reward
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.fish_population <= 0.0:
            done = True

        return self.observed, reward, done, {}

    def reset(self):
        self.fish_population = np.array([self.init_state])
        self.observed = observation_noise(self.fish_population[0],
                                          self.sigma_m,
                                          self.observation_space)
        self.harvest = self.init_harvest
        self.action = 0
        self.years_passed = 0
        return self.fish_population


def observation_noise(mu, sigma, observation_space):
    # x = np.random.uniform(mu - sigma, mu + sigma)
    x = np.random.normal(mu, sigma)
    return np.clip(x, observation_space.low, observation_space.high)
