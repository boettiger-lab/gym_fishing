import gym
import gym_fishing
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard

env = gym.make('fishing-v1')
model = SAC('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=200000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("SAC", mean_reward, std_reward)


model.save("results/SAC")
