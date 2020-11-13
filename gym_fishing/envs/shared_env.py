import numpy as np
from csv import writer
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from gym import spaces

# This is the parent fishing class that holds a lot of common methods shared
# across the gym_fishing environments.


class BaseFishingEnv():

    def __init__(self, 
                 r = 0.3,
                 K = 1.0,
                 price = 1.0,
                 sigma = 0.01,
                 init_state = 0.75,
                 init_harvest = 0.0125,
                 Tmax = 100,
                 file_ = None
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
        if(file_ != None):
            self.write_obj = open(file_, 'w+')


        ## Set the action space
        self.action_space = spaces.Box(np.array([0]), np.array([self.K]), dtype = np.float32)
        self.observation_space = spaces.Box(np.array([0]), np.array([2 * self.K]), 
                                             dtype = np.float32)
    def harvest_draw(self, quota):
        """
        Select a value to harvest at each time step.
        """

        # index (fish.population[0]) to avoid promoting float to array
        self.harvest = min(self.fish_population[0], quota)
        self.fish_population = max(self.fish_population - self.harvest, 0.0)
        return self.harvest

    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.
        """
        self.fish_population = max(
            self.fish_population + self.r * self.fish_population
            * (1.0 - self.fish_population / self.K)
            + self.fish_population * self.sigma * np.random.normal(0, 1),
            0.0)
        return self.fish_population

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

        if self.fish_population <= 0.0:
            done = True

        return self.fish_population, reward, done, {}
    
    def reset(self):
        self.fish_population = np.array([self.init_state])
        self.harvest = self.init_harvest
        self.action = 0
        self.years_passed = 0
        return self.fish_population

    def simulate(self, model, reps=1):
        row = []
        for rep in range(reps):
            obs = self.reset()
            for t in range(env.Tmax):
                action, _state = model.predict(obs)
                obs, reward, done, info = self.step(action)
                row.append([t, obs[0], action, reward, int(rep)])
                if done:
                    break
        df = DataFrame(
            row, columns=['time', 'state', 'action', 'reward', "rep"])
        return df

    def policyfn(self, model, reps=1, n=50):
        row = []
        state_range = np.linspace(self.observation_space.low,
                                  self.observation_space.high,
                                  num=n,
                                  dtype=self.observation_space.dtype)
        for rep in range(reps):
            for obs in state_range:
                action, _state = model.predict(obs)
                row.append([obs[0], action, rep])

        df = DataFrame(row, columns=['state', 'action', 'rep'])
        return df

    def plot(self, df, output="results.png"):
        fig, axs = plt.subplots(3, 1)
        for i in range(np.max(df.rep)):
            results = df[df.rep == i]
            episode_reward = np.cumsum(results.reward)
            axs[0].plot(results.time, results.state, color="blue", alpha=0.3)
            axs[1].plot(results.time, results.action, color="blue", alpha=0.3)
            axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)

        axs[0].set_ylabel('state')
        axs[1].set_ylabel('action')
        axs[2].set_ylabel('reward')
        fig.tight_layout()
        plt.savefig(output)
        plt.close("all")
    
    def csv_entry(self):
        row_contents = [self.years_passed,
                        self.state[0],
                        self.action,
                        self.reward]
        csv_writer = writer(self.write_obj)
        csv_writer.writerow(row_contents)
        return row_contents
    
    def render(self, mode='human'):
        return self.csv_entry()
    
    def close(self):
      if(self.write_obj != None):
           self.write_obj.close()
