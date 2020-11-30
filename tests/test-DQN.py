import numpy as np
import gym
import gym_fishing
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def test_dqn():
  env = gym.make('fishing-v0')
  check_env(env)
  
  model = DQN('MlpPolicy', env, verbose=1)
  model.learn(total_timesteps=200)
  
  ## Simulate a run with the trained model, visualize result
  df = env.simulate(model)
  env.plot(df, "dqn.png")
  
  ## Evaluate model
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

  ## Save and reload the model
  model.save("dqn")
  model = DQN.load("dqn")


