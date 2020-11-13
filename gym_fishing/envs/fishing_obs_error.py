
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import harvest_draw, population_draw, csv_entry, simulate_mdp, plot_mdp


class FishingObsError(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K = 1.0,
                 r = 0.3,
                 price = 1.0,
                 sigma = 0.02,
                 sigma_m = 0.02,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 Tmax = 100,
                 file = None):
                   
                   
        ## parameters
        self.K = K
        self.r = r
        self.price = price
        self.sigma = sigma
        self.sigma_m = sigma_m
        ## for reset
        self.init_state = init_state
        self.init_harvest = init_harvest
        self.Tmax = Tmax
        # for reporting purposes only
        if(file != None):
          self.write_obj = open(file, 'w+')
          
        self.action = 0
        self.years_passed = 0
        self.reward = 0
        self.fish_population = np.array([init_state])

        self.harvest = (self.r * self.K / 4.0) / 2.0
        
        self.action_space = spaces.Box(np.array([0]), np.array([self.K]), dtype = np.float32)
        self.observation_space = spaces.Box(np.array([0]), np.array([2 * self.K]), dtype = np.float32)

        self.observed = observation_noise(self.fish_population[0], 
                                  self.sigma_m, 
                                  self.observation_space)
   

    def step(self, action):
      
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        self.harvest = action
        
        harvest_draw(self, self.harvest)
        population_draw(self)
        self.observed = observation_noise(self.fish_population[0], 
                                          self.sigma_m, 
                                          self.observation_space)

        ## should be the instanteous reward, not discounted
        reward = max(self.price * self.harvest, 0.0)
        self.reward = reward
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.fish_population <= 0.0:
            done = True

        return self.observed, self.reward, done, {}
        
    
    def reset(self):
        self.fish_population = np.array([self.init_state])
        self.observed = observation_noise(self.fish_population[0], 
                                          self.sigma_m, 
                                          self.observation_space)
        self.harvest = self.init_harvest
        self.action = 0
        self.years_passed = 0
        return self.fish_population
  
    def render(self, mode='human'):
      return csv_entry(self)
  
    def close(self):
      if(self.write_obj != None):
        self.write_obj.close()

    def simulate(env, model, reps = 1):
      return simulate_mdp(env, model, reps)
      
    def plot(self, df, output = "results.png"):
      return plot_mdp(self, df, output)
      




      

def observation_noise(mu, sigma, observation_space):
  # x = np.random.uniform(mu - sigma, mu + sigma)
  x = np.random.normal(mu, sigma)
  return np.clip(x, observation_space.low, observation_space.high)
    

