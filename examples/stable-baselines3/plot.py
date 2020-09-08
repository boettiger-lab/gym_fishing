import os
import sys
sys.path.append(os.path.realpath("../.."))

import argparse
import gym
import gym_fishing
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import numpy as np


reps = 100
parser = argparse.ArgumentParser()
parser.add_argument("-i", type=float, default=0.75)
args = parser.parse_args()

env = gym.make('fishing-v1')
model = SAC.load("sb3_sac")
traj = [[0 for j in range(101)] for i in range(reps)]
reward_list = []
for i in range(reps):
    obs = env.reset()
    traj[i][0] = obs[0]
    for t in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        traj[i][t+1] = obs[0]
#import pdb; pdb.set_trace()
#x = np.linspace(0, 1, 100)
#y = [model.predict(np.array([obs]))[0][0] for obs in x]
plt.plot(np.linspace(0, 100, 101), np.mean(traj, axis=0))
plt.xlabel("t")
plt.ylabel("Average N")
plt.savefig("plot.png")

