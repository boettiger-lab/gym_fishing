import gym
import gym_fishing
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard

env = gym.make('fishing-v1',  file = "results/SAC.csv")
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200000)

## Simulation for visualization purposes
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
env.plot(output = "results/SAC.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("SAC", mean_reward, std_reward)


model.save("results/SAC")
