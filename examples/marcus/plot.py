import os
import sys
sys.path.append(os.path.realpath("../.."))

import argparse
import gym
import gym_fishing
import matplotlib.pyplot as plt
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=float, default=0.75)
args = parser.parse_args()

env = make_vec_env('fishing-v1', n_envs=4)
model = PPO2.load("sb2_ppo2_recurrent")
traj = [[0 for j in range(101)] for i in range(100)]
reward_list = []
for i in range(100):
    obs = env.reset()
    traj[i][0] = obs.reshape(-1)
    for t in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        traj[i][t+1] = obs.reshape(-1)
#import pdb; pdb.set_trace()
#x = np.linspace(0, 1, 100)
#y = [model.predict(np.array([obs]))[0][0] for obs in x]
plt.plot(np.linspace(0, 100, 101), np.mean(np.mean(traj, axis=0), axis=1))
plt.xlabel("t")
plt.ylabel("Average N")
plt.savefig("plot.png")

