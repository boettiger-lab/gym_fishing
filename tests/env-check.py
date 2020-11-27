import gym
import gym_fishing
from stable_baselines3.common.env_checker import check_env

def test_discrete:
  env0 = gym.make('fishing-v0')
  check_env(env0)
  env0.close()

def test_cts:
  env1 = gym.make('fishing-v1')
  check_env(env1)

def test_tipping:  
  env2 = gym.make('fishing-v2')
  check_env(env2)

def test_uncertainty:
  env4 = gym.make('fishing-v4')
  check_env(env4)


