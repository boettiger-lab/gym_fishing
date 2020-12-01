import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import gym_fishing
from gym_fishing.models.policies import escapement, msy, user_action


def test_discrete():
    env = gym.make("fishing-v0")
    check_env(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "v0_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "v0_escapement.png")


def test_cts():
    env = gym.make("fishing-v1")
    check_env(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "v1_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "v1_escapement.png")


def test_allen():
    env = gym.make("fishing-v5", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "allen_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "allen_escapement.png")


def test_beverton_holt():
    env = gym.make("fishing-v6", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "bh_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "bh_escapement.png")


def test_may():
    env = gym.make("fishing-v7", sigma=0)
    check_env(env)
    #  model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "may_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "may_escapement.png")


def test_myers():
    env = gym.make("fishing-v8", sigma=0)
    check_env(env)
    model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "myers_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "myers_escapement.png")


def test_ricker():
    env = gym.make("fishing-v9", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "ricker_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "ricker_escapement.png")


def test_tipping():
    env = gym.make("fishing-v2", sigma=0, init_state=0.75)
    check_env(env)
    # increases above tipping point
    obs, reward, done, info = env.step(env.get_action(0))
    assert env.get_fish_population(obs) > 0.75

    ## Decreases below the tipping point
    env.init_state = 0.3
    env.reset()
    obs, reward, done, info = env.step(env.get_action(0))
    assert env.get_fish_population(obs) < 0.3

    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "tip_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "tip_escapement.png")


def test_nonstationary():
    env = gym.make("fishing-v10", sigma=0, alpha=-0.007)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "ns_msy.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "ns_escapement.png")


def test_model_uncertainty():
    np.random.seed(0)
    env = gym.make("fishing-v11")
    check_env(env)
    model = user_action(env)
    model = msy(env)
    df = env.simulate(model, reps=10)
    env.plot(df, "mu_msy.png")
    model = escapement(env)
    df = env.simulate(model, reps=10)
    env.plot(df, "mu_escapement.png")
