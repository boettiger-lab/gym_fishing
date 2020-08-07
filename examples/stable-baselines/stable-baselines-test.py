import gym
import gym_fishing
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams

env = gym.make('fishing-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

def simulate(trained_model, total_timesteps):
   obs = env.reset()
   y = []
   total_rewards = []
   all_rewards = []
   for i in range(total_timesteps):
     action, _states = trained_model.predict(obs)
     obs, rewards, dones, info = env.step(action)
     y.append(action)
     total_rewards.append(rewards)
     all_rewards.append(sum(total_rewards))
   def get_action():
       return y
   def get_reward():
       return all_rewards
   get_action()
   get_reward()
   env.render()
   env.close()

   #Change size of plot
   rcParams['figure.figsize'] = 30, 8
   x = np.linspace(0, total_timesteps, total_timesteps)
   # Create a figure and an axes.
   fig, ax = plt.subplots()
   ax.scatter(x, get_action())
   ax.set_xlabel('Timesteps')
   ax.set_ylabel('Harvest')


    
   rcParams['figure.figsize'] = 8, 5
   x = np.linspace(0, total_timesteps, total_timesteps)
   # Create a figure and an axes
   fig, bx = plt.subplots()  
   bx.plot(x, get_reward(), label='linear')
   bx.set_xlabel('Timesteps')
   bx.set_ylabel('Cumulative Reward')
   
   
simulate(trained_model, 1000)
