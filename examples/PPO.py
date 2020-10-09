import numpy as np
import gym
import gym_fishing
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard

env = gym.make('fishing-v0', 
               file = "results/ppo.csv", 
               n_actions = 3)
               
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=300000)


## Simulation for visualization purposes
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
env.plot("results/ppo.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
## Primative leaderboard
leaderboard("PPO", mean_reward, std_reward)


model.save("results/ppo")
