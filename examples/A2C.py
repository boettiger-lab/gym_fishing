import numpy as np
import gym
import gym_fishing
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard

env = gym.make('fishing-v0', 
               file = "results/a2c.csv", 
               n_actions = 3)
               
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500000)

## Simulation for visualization purposes
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
env.plot("results/a2c.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("A2C", mean_reward, std_reward)

model.save("results/a2c")
