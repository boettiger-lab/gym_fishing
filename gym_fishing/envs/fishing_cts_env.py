

import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np

class FishingCtsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.fish_population = np.array([1.0])
        self.K = 1.0
        self.r = 0.1
        self.price = 1.0
        self.sigma = 0.0
        
        self.years_passed = 0

        self.harvest = (self.r * self.K / 4.0) / 2.0
        
        self.action_space = spaces.Box(np.array([0]), np.array([2 * self.K]), dtype = np.float)
        self.observation_space = spaces.Box(np.array([0]), np.array([2 * self.K]), dtype = np.float)
        
    def harvest_draw(self, quota):
        """
        Select a value to harvest at each time step.
        """
        
        ## index (fish.population[0]) to avoid promoting float to array
        self.harvest = min(self.fish_population[0], quota)
        self.fish_population = max(self.fish_population - self.harvest, 0.0)
        return self.harvest
    
    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.
        """
        self.fish_population = max(
                                self.fish_population + self.r * self.fish_population \
                                * (1.0 - self.fish_population / self.K) \
                                + self.fish_population * self.sigma * np.random.normal(0,1),
                                0.0)
        return self.fish_population

    
    def step(self, action):
      
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        action = np.clip(action, 0, 2 * self.K)[0]
        
        
        harvest = action
        self.harvest = harvest
        
        self.harvest_draw(harvest)
        self.population_draw()
        
        #self.reward += self.price * self.harvest
        reward = max(self.price * self.harvest, 0.0)
        
        self.years_passed += 1
        done = bool(self.years_passed > 1000)

        if self.fish_population <= 0.0:
            done = True
            return self.fish_population, reward, done, {}
        
        
        return self.fish_population, reward, done, {}
        
    
    def reset(self, 
              init_state = 0.75,
              r = 0.1,
              K = 1.0,
              price = 1.0,
              sigma = 0.0
              ):
        self.fish_population = np.array([init_state])
        self.r = r
        self.K = K
        self.price = price
        self.sigma = sigma
        
        
        self.harvest = self.r * self.K / 4.0 / 2.0
        
        self.years_passed = 0
        return self.fish_population
  
  
    def render(self, mode='human'):
        pass
  
  
    def close(self):
        pass
