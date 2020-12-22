import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_fishing


def test_ppo():
    env = gym.make("fishing-v1")
    check_env(env)
    # takes about 200000 to get a decent policy, about a 12 min test..
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200)

    # Simulate a run with the trained model, visualize result
    df = env.simulate(model)
    env.plot(df, "PPO-test.png")

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
