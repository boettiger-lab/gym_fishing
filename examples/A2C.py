import numpy as np
import gym
import gym_fishing
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard

env = gym.make('fishing-v0')
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=500000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("A2C", mean_reward, std_reward)

model.save("results/a2c")
