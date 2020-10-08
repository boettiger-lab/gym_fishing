import numpy as np
import gym
import gym_fishing
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('fishing-v1',  file = "results/td3.csv")
model = TD3('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200000)

model.save("results/td3")

model = TD3.load("results/td3")
env = gym.make('fishing-v1',  file = "results/td3.csv")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print("mean reward:", mean_reward, "std:", std_reward)

## Simulation for visualization purposes
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()

env.plot(output = "results/td3.png")
   # obs = env.reset()
