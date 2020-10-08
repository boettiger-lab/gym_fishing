
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from csv import writer
from pandas import read_csv
import matplotlib.pyplot as plt


class FishingCtsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K = 1.0,
                 r = 0.1,
                 price = 1.0,
                 sigma = 0.0,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 Tmax = 100,
                 file = "fishing.csv"):
                   
                   
        ## parameters
        self.K = K
        self.r = r
        self.price = price
        self.sigma = sigma
        ## for reset
        self.init_state = init_state
        self.init_harvest = init_harvest
        self.Tmax = Tmax
        # for reporting purposes only
        self.file = file
        self.action = 0
        self.years_passed = 0
        
        self.fish_population = np.array([1.0])
        self.harvest = (self.r * self.K / 4.0) / 2.0
        
        self.action_space = spaces.Box(np.array([0]), np.array([self.K]), dtype = np.float)
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
      
        action = np.clip(action, 0, 2 * self.K)[0]
        self.harvest = action
        
        self.harvest_draw(self.harvest)
        self.population_draw()
        
        #self.reward += self.price * self.harvest
        reward = max(self.price * self.harvest, 0.0)
        
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.fish_population <= 0.0:
            done = True

        return self.fish_population, reward, done, {}
        
    
    def reset(self):
        self.fish_population = np.array([self.init_state])
        self.write_obj = open(self.file, 'w+')
        self.harvest = 0.1 * 1 / 4.0 / 2.0
        self.years_passed = 0
        return self.fish_population
  
  
    def render(self, mode='human'):
      row_contents = [self.years_passed, 
                      self.fish_population[0],
                      self.action,
                      self.harvest]
      csv_writer = writer(self.write_obj)
      csv_writer.writerow(row_contents)
      return row_contents
  
    def close(self):
      close(self.file)

    
    def plot(self, output = "fishing.png"):
      results = read_csv(self.file,
                          names=['time','state','action','reward'])
      fig, axs = plt.subplots(3,1)
      axs[0].plot(results.state)
      axs[0].set_ylabel('state')
      axs[1].plot(results.action)
      axs[1].set_ylabel('action')
      axs[2].plot(results.reward)
      axs[2].set_ylabel('reward')
      fig.tight_layout()
      plt.savefig(output)
      plt.close("all")
