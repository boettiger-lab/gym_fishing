import numpy as np
import gym
import gym_fishing
from stable_baselines3 import A2C

env = gym.make('fishing-v0', 
               file = "results/a2c.csv", 
               fig = "results/a2c.png",
               n_actions = 100)
               
model = A2C('MlpPolicy', env, verbose=2)
model.learn(total_timesteps=200000)

model.save("results/a2c")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

## Simulation for visualization purposes
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  
#  if done:
#    obs = env.reset()

env.plot()

