
#Remove Warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


#Imports
import gym_fishing
import pandas as pd
import gym
import numpy as np
from collections import defaultdict
import itertools

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import matplotlib.pyplot as plt
from pylab import rcParams


#Environment
env = gym.make('fishing-v0')
model = DQN(MlpPolicy, env , verbose=2)
trained_model = model.learn(total_timesteps=10000)


#Monte Carlo Estimation
#Gives slightly better rewards and gives more 'under the hood' details


def mc_control_epsilon_greedy(env, num_episodes, discount=.99):

  """
  Monte Carlo Prediction - Estimates value function through sampling
  """

  #set variables as dictionaries with type float
  
  return_sum = defaultdict(float)
  #return_occurence = defaultdict(float)

  Value = defaultdict(float)

  for i in range(1, num_episodes + 1):
    if i % 1 ==0:
      print("\rEpisode {}/{}".format(i, num_episodes), end="") #carriage return
            

#Generate episode with (state, action, reward) tuple

    episode = []
    obs = env.reset()
    for i in range(num_episodes):
      action, _states = trained_model.predict(obs)
      next_obs, reward, done, info = env.step(action)
      episode.append((obs, action, reward))
      if done:
        break
      obs = next_obs 
    
    #convert to tuple
    tuple(episode)
    for x in episode:
      all_episode_obs = set([tuple(x[0])])
      for i in range(obs):
        if x[0] == obs:
          first_occurence = x in enumerate(episode)

     # first_occurence = next(i for i, x in enumerate(episode) if x[0] == obs)
      #obtain cumulative return
      reward_sum = sum([x[2] * (discount**i) for i, x in enumerate(episode[first_occurence:])])

      return_sum[tuple(obs)] += reward_sum
      return_occurence = 0
      while x <= num_episodes:
        return_occurence += 1
        if return_occurence == num_episodes:
          final

      return_occurence += 1.0

      Value[tuple(obs)] = return_sum[tuple(obs)] / return_occurence
    
  return Value

prediction = estimate(env, num_episodes=100)


#Add Weighted epsilon-greedy sampling

def weighted_sampling(env, num_episodes, discount = .99):
    """
    Monte Carlo Control Off-Policy Control using Weights for Sampling.
    Finds an optimal greedy policy.
    """
    
    # creates Q dictionary that maps obs to action values
    Q = defaultdict(lambda: np.zeros(env.action_space))
    #dictionary for weights
    C = defaultdict(lambda: np.zeros(env.action_space))
    
    # learn greey policy
    target_policy = env.step(Q)
        
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

        # Generate an episode to be tuple (state, action, reward) tuples
        episode = []
        obs = env.reset()
        for t in range(100):
            # Sample an action from our policy
            action, _states = trained_model.predict(obs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            obs = next_obs
        
        # Sum of discounted returns
        G = 0.0
        # weights for return
        W = 1.0
        for t in range(len(episode))[::-1]:
            obs, action, reward = episode[t]
            G = discount * G + reward
            #  Add weights
            C[obs][action] += W
            # Update policy
            Q[obs][action] += (W / C[obs][action]) * (G - Q[obs][action]
                                                      
            if action !=  np.argmax(target_policy(obs)):
                break
            W = W * 1./behavior_policy(obs)[action]
        
    return Q, target_policy

Q, policy = mc_control_importance_sampling(env, num_episodes=500000)
