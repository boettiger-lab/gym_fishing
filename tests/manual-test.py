import pandas as pd
import gym
import gym_fishing

          

r = 0.1
K = 1
env = gym.make('fishing-v1', r = r, K = K, sigma = 0.)

# MSY
msy = r * K / 4
action = msy / K - 1

## inits
row = []
rep = 0

## Simulate under MSY
for t in range(env.Tmax):
  obs, reward, done, info = env.step(action)
  fish_population = (obs[0] + 1) * env.K
  quota =  (action + 1) * env.K
  row.append([t, fish_population, action, reward, int(rep)])
  if done: break
      

df = pd.DataFrame(row, columns=['time', 'state', 'action', 'reward', "rep"])
df


env.plot(df, "msy.png")

