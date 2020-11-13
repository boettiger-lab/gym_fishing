
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import harvest_draw, population_draw, \
  csv_entry, simulate_mdp, plot_mdp, estimate_policyfn

class AbstractFishingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 r = 0.3,
                 K = 1.0,
                 price = 1.0,
                 sigma = 0.01,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 Tmax = 100,
                 n_actions = 3,
                 file = None
                 ):
        ## Action and state           
        self.fish_population = np.array([init_state])
        self.harvest = init_harvest
        self.reward = 0
        ## parameters
        self.K = K
        self.r = r
        self.price = price
        self.sigma = sigma
        ## for reset
        self.init_state = init_state
        self.init_harvest = init_harvest
        
        # for reporting purposes only
        self.action = 0
        self.years_passed = 0
        self.Tmax = Tmax
        if(file != None):
          self.write_obj = open(file, 'w+')


        ## Set the action space
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(np.array([0]), 
                                            np.array([2 * self.K]), 
                                            dtype = np.float32)
        

    def step(self, action):
      
        action = np.clip(action, int(0), int(self.n_actions))
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
      
        harvest_draw(self, self.harvest)
        population_draw(self)
        reward = max(self.price * self.harvest, 0.0)
        
        ## recording purposes only
        self.action = int(action)
        self.reward = reward
        self.years_passed += 1
        done = bool(self.years_passed >= self.Tmax)

        if self.fish_population <= 0.0:
            done = True
            return self.fish_population, self.reward, done, {}

        return self.fish_population, self.reward, done, {}
        
    def reset(self):
        self.fish_population = np.array([self.init_state])
        self.harvest = self.init_harvest
        self.action = 0
        self.reward = 0
        self.years_passed = 0
        return self.fish_population


    def render(self, mode='human'):
      return csv_entry(self)
  
    def close(self):
      if(self.write_obj != None):
        self.write_obj.close()

    def simulate(env, model, reps = 1):
      return simulate_mdp(env, model, reps)

    def policyfn(env, model, reps = 1):
      return estimate_policyfn(env, model, reps)
      
    def plot(self, df, output = "results.png"):
      return plot_mdp(self, df, output)






class FishingEnv(AbstractFishingEnv):
    def __init__(self, **kargs):
        super(FishingEnv, self).__init__(**kargs)


class FishingEnv100(AbstractFishingEnv):
    def __init__(self, **kargs):
        super(FishingEnv100, self).__init__(n_actions = 100, **kargs)
  
