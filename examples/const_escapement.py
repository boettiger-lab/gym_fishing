import pandas as pd
import gym
import gym_fishing
import math

r = 0.3
K = 1
env = gym.make("fishing-v1", r=r, K=K, sigma=0.0)

## inits
row = []
rep = 0
env.reset()
obs = env.state

# Note that we must map between the "observed" space (on -1, 1) and model
# space (0, 2K) for both actions and states with the get_* methods

for t in range(env.Tmax):
    fish_population = env.get_fish_population(obs)
    ## The escapement rule
    Q = max(fish_population - K / 2, 0)
    action = env.get_action(Q)
    quota = env.get_quota(action)
    obs, reward, done, info = env.step(action)
    row.append([t, fish_population, quota, reward, int(rep)])


df = pd.DataFrame(row, columns=["time", "state", "action", "reward", "rep"])
df


env.plot(df, "const_escapement.png")
