import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from gym_fishing.envs.shared_env import BaseFishingEnv

class FishingEnv(BaseFishingEnv, gym.Env):
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
                 file_ = None
                 ):
        super().__init__(r, K, price, sigma, init_state, init_harvest, Tmax, file_)
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        

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
      
        self.harvest_draw(self.harvest)
        self.population_draw()
        reward = max(self.price * self.harvest, 0.0)
        
        ## recording purposes only
        self.action = int(action)
        self.reward = np.array([reward])
        self.years_passed += 1
        done = bool(self.years_passed >= self.Tmax)

        if self.fish_population <= 0.0:
            done = True
            return self.fish_population, self.reward, done, {}

        return self.fish_population, self.reward, done, {}
        
