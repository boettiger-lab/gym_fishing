import numpy as np
import gym
import gym_fishing
from stable_baselines3 import DQN

env = gym.make('fishing-v0')
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200)

## Simulate a run with the trained model, visualize result
df = env.simulate(model)
env.plot(df, "dqn.png")

## Evaluate model
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print("mean reward:", mean_reward, "std:", std_reward)


## Save and reload the model
model.save("dqn")
model = DQN.load("dqn")
