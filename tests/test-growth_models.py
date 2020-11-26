import gym
import gym_fishing
import numpy as np
from gym_fishing.models.policies import msy, escapement, user_action
from stable_baselines3.common.env_checker import check_env

def test_env():
  env = gym.make('fishing-v6', sigma=0) 
  check_env(env)
  #model = user_action(env)
  model = msy(env)
  df = env.simulate(model)
  env.plot(df, "msy.png")

  model = escapement(env)
  df = env.simulate(model)
  env.plot(df, "escapement.png")

  
test_env()
