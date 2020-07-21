import os
import sys
sys.path.append(os.path.realpath("../.."))

import gym
import gym_fishing
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import numpy as np

env = gym.make('fishing-v1')
model = SAC.load("sb3_sac")
obs = env.reset()
traj = []
for t in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    traj.append([t, obs, env.harvest])
    if done:
        break
import pdb; pdb.set_trace()
traj = np.array(traj)
plt.plot(traj[:, 0], traj[:, 1])
plt.savefig("trash.png")

