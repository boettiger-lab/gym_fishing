import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


import gym
import matplotlib
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from collections import defaultdict


from stable_baselines.common.policies import LstmPolicy
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise




from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution

"""
    Policy object that implements actor critic

    :sess: Current TensorFlow session
    :param ob_space: The observation space of env
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run for multiprocessing
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batchs to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

#Custom MLP Policy: Actor has 3 layers of size 128, Critic has 2 layers of size 32
# with a nature_cnn feature extractor

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch, reuse=reuse, scale=True)

        self._probability_distribution = DiagGaussianProbabilityDistribution
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None
        self.n_env = 2
        self.n_steps = 1000
        n_batch = 2000

        with tf.variable_scope("model", reuse=reuse):
            activation = tf.nn.relu

            output = nature_cnn(self.processed_obs, **kwargs)
            output = tf.layers.flatten(output)

            pi_h = output
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = output
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})




env = DummyVecEnv([lambda: gym.make('fishing-v1')])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(CustomPolicy, env, action_noise=action_noise, verbose=1)
# Train the agent
model.learn(total_timesteps=100000)

  
