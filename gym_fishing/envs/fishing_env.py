

import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from csv import writer


class FishingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 r = 0.1,
                 K = 1.0,
                 price = 1.0,
                 sigma = 0.05,
                 gamma = 0.99,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 n_actions = 3
                 ):
        ## Action and state           
        self.fish_population = np.array([init_state])
        self.harvest = init_harvest
        
        ## parameters
        self.K = K
        self.r = r
        self.price = price
        self.sigma = sigma
        self.gamma = gamma
        
        # for reporting purposes only
        self.reward = 0.0
        self.action = 0
        self.years_passed = 0
        
        
        ## for reset
        self.init_state = init_state
        self.init_harvest = init_harvest
        
        ## Set the action space
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(np.array([0]), 
                                            np.array([2 * self.K]), 
                                            dtype = np.float)
        
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
      
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        if self.n_actions > 3:
          self.harvest = ( action / self.n_actions ) * self.K
        
        ## Discrete actions: increase, decrease, stay the same
        else:
          if action == 0:
            self.harvest = self.harvest
          elif action == 1:
            self.harvest = 1.2 * self.harvest
          else:
            self.harvest = 0.8 * self.harvest
            

        self.harvest_draw(self.harvest)
        self.population_draw()
        
        reward = max(self.price * self.harvest, 0.0)
        
        
        self.reward += reward * self.gamma ** self.years_passed
        
        self.years_passed += 1
        done = bool(self.years_passed >= 1000)

        if self.fish_population <= 0.0:
            done = True
            return self.fish_population, reward, done, {}
        
        
        return self.fish_population, reward, done, {}
        
    
    def reset(self):
        self.fish_population = np.array([self.init_state])
        
        self.harvest = self.r * self.K / 4.0 / 2.0
        self.reward = 0.0
        self.years_passed = 0
        return self.fish_population
  
  
    def render(self, mode='human'):
      row_contents = [self.years_passed, 
                      self.fish_population[0], 
                      self.harvest, 
                      self.action, 
                      self.reward]
      with open("fishing.csv", 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(row_contents)
  

  
    def close(self):
        pass
