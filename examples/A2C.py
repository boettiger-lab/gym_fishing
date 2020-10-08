import numpy as np
import gym
import gym_fishing
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('fishing-v0', 
               file = "results/a2c.csv", 
               n_actions = 3)
               
model = A2C('MlpPolicy', env, verbose=2)
model.learn(total_timesteps=20000)

model.save("results/a2c")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean reward:", mean_reward, "std:", std_reward)

## Simulation for visualization purposes
obs = env.reset()
for i in range(1000):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  
  if done:
    env.plot()
    obs = env.reset()
