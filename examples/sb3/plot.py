import os
import sys
sys.path.append(os.path.realpath("../.."))

import argparse
import gym
import gym_fishing
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import numpy as np


reps = 25
parser = argparse.ArgumentParser()
parser.add_argument("-i", type=int, default=0)
args = parser.parse_args()

env = gym.make('fishing-v1')
model = SAC.load(f"models/sb3_sac_{args.i}")
trajs = [] #List to find agg. stats on trajectories
reward_list = [] #To find agg. stats on rewards
for i in range(reps):
    traj = []
    obs = env.reset()
    traj.append(obs[0])
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        traj.append(obs[0])
        if done:
            break
    trajs.append(traj)
    plt.plot(np.linspace(0, len(traj)-1, len(traj)), traj)
plt.xlabel("t")
plt.ylabel("N")
plt.savefig("plot.png")

