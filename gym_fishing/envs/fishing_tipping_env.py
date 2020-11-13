
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import BaseFishingEnv 


class FishingTippingEnv(BaseFishingEnv, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K = 1.0,
                 r = 0.3,
                 C = 0.5,
                 price = 1.0,
                 sigma = 0.0,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 Tmax = 100,
                 file_ = None):
                   
        super().__init__(r, K, price, sigma, init_state, init_harvest, Tmax, file_)
        self.C = C
        
    def harvest_draw(self, quota):
        ## index (fish.population[0]) to avoid promoting float to array
        self.harvest = min(self.state[0], quota)
        self.state = max(self.state - self.harvest, 0.0)
        return self.harvest
    
    def population_draw(self):
        self.state = max(
          self.state * np.exp(self.r *
                             (1 -self.state / self.K) *
                             (self.state - self.C) +
                             self.state * self.sigma * np.random.normal(0,1)
                             ), 
          0)
        return self.state

    
    def step(self, action):
      
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.harvest = action
        
        self.harvest_draw(self.harvest)
        self.population_draw()
        
        ## should be the instanteous reward, not discounted
        reward = max(self.price * self.harvest, 0.0)
        self.reward = reward
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.state <= 0.0:
            done = True

        return self.state, reward, done, {}
        

