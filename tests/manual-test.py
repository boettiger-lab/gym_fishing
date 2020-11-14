import pandas as pd
import gym
import gym_fishing

          

r = 0.1
K = 1
env = gym.make('fishing-v0', r = r, K = K, sigma = 0.)

# MSY
msy = r * K / 4
action = env.get_action(msy)

## inits
row = []
rep = 0

## Simulate under MSY
for t in range(env.Tmax):
  obs, reward, done, info = env.step(action)
  fish_population = env.get_fish_population(obs)
  quota =  env.get_quota(action)
  row.append([t, fish_population, quota, reward, int(rep)])
  if done: break
      

df = pd.DataFrame(row, columns=['time', 'state', 'action', 'reward', "rep"])
df


env.plot(df, "msy.png")

