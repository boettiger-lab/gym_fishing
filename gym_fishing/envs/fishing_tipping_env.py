
import math
from math import floor
import gym
from gym import spaces, logger, error, utils
from gym.utils import seeding
import numpy as np
from csv import writer
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt


class FishingTippingEnv(gym.Env):
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
                 file = None):
                   
                   
        ## parameters
        self.K = K
        self.r = r
        self.C = C
        self.price = price
        self.sigma = sigma
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
        
        self.state = np.array([1.0])
        self.harvest = (self.r * self.K / 4.0) / 2.0
        
        self.action_space = spaces.Box(np.array([0]), np.array([self.K]), dtype = np.float32)
        self.observation_space = spaces.Box(np.array([0]), np.array([2 * self.K]), dtype = np.float32)
        
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
      
#        action = np.clip(action, 0, 2 * self.K)[0]
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
        
    
    def reset(self):
        self.state = np.array([self.init_state])
        self.harvest = self.init_harvest
        self.action = 0
        self.years_passed = 0
        return self.state
  
    def render(self, mode='human'):
      row_contents = [self.years_passed, 
                      self.state[0],
                      self.action,
                      self.reward]
      csv_writer = writer(self.write_obj)
      csv_writer.writerow(row_contents)
      return row_contents
  
    def close(self):
      if(self.write_obj != None):
        self.write_obj.close()


    def simulate(env, model, reps = 1):
      row = []
      for rep in range(reps):
        obs = env.reset()
        reward = 0
        for t in range(env.Tmax-1):
          action, _state = model.predict(obs)
          row.append([t, obs, action, reward, rep])
          obs, reward, done, info = env.step(action)
          if done:
            break
        row.append([t+1, obs, None, reward, rep])
      df = DataFrame(row, columns=['time', 'state', 'action', 'reward', "rep"])
      return df
    
    def plot(self, df, output = "fishing.png"):
      fig, axs = plt.subplots(3,1)
      for i in range(np.max(df.rep)):
        results = df[df.rep == i]
        episode_reward = np.cumsum(results.reward)                    
        axs[0].plot(results.time, results.state, color="blue", alpha=0.3)
        import pdb; pdb.set_trace()
        axs[1].plot(results.time.iloc[:-1], results.action.iloc[:-1], color="blue", alpha=0.3)
        axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)
      
      axs[0].set_ylabel('state')
      axs[1].set_ylabel('action')
      axs[2].set_ylabel('reward')
      fig.tight_layout()
      plt.savefig(output)
      plt.close("all")

      
 