
## Introduction

This example shows how to train a [DQN (Deep Q
Networks)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
agent on the Cartpole environment using the TF-Agents library.

![Cartpole
environment](https://raw.githubusercontent.com/tensorflow/agents/master/docs/tutorials/images/cartpole.png)

It will walk you through all the components in a Reinforcement Learning
(RL) pipeline for training, evaluation and data collection.

## Setup

``` r
library(reticulate)
reticulate::use_virtualenv("~/.virtualenvs/tf2.0")
#reticulate::py_discover_config()
```

``` python


from __future__ import absolute_import, division, print_function

import gym_fishing

import gym
import base64
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

#### Acitvate compatibility settings 
##tf.compat.v1.enable_v2_behavior()
### change default float
## tf.keras.backend.set_floatx('float64')

## Disable GPU, optional
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


## Grow memory as needed instead of pre-allocating (e.g. allows concurrent GPU usage)
## (TF2 version.  See https://stackoverflow.com/a/59126638/258662)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

tf.version.VERSION
```

    ## '2.1.0'

``` python
## Hyperparameters
num_iterations = 200000             # @param {type:"integer"}
initial_collect_steps = 1000        # @param {type:"integer"} 
collect_steps_per_iteration = 1     # @param {type:"integer"}
replay_buffer_max_length = 1000000  # @param {type:"integer"}
batch_size = 64                     # @param {type:"integer"}
learning_rate = 1e-3                # @param {type:"number"}
log_interval = 200                  # @param {type:"integer"}
num_eval_episodes = 10              # @param {type:"integer"}
eval_interval = 1000                # @param {type:"integer"}
```

## Environment

``` python
env_name = 'fishing-v2'
env = suite_gym.load(env_name)
env.reset()
```

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([0.75]))

The `environment.step` method takes an `action` in the environment and
returns a `TimeStep` tuple containing the next observation of the
environment and the reward for the action.

The `time_step_spec()` method returns the specification for the
`TimeStep` tuple. Its `observation` attribute shows the shape of
observations, the data types, and the ranges of allowed values. The
`reward` attribute shows the same details for the
    reward.

``` python
print('Observation Spec:')
```

    ## Observation Spec:

``` python
print(env.time_step_spec().observation)
```

    ## BoundedArraySpec(shape=(1,), dtype=dtype('float64'), name='observation', minimum=0.0, maximum=2.0)

``` python
print('Reward Spec:')
```

    ## Reward Spec:

``` python
print(env.time_step_spec().reward)
```

    ## ArraySpec(shape=(), dtype=dtype('float32'), name='reward')

The `action_spec()` method returns the shape, data types, and allowed
values of valid
    actions.

``` python
print('Action Spec:')
```

    ## Action Spec:

``` python
print(env.action_spec())
```

    ## BoundedArraySpec(shape=(), dtype=dtype('int64'), name='action', minimum=0, maximum=99)

In the `fishing-v2` environment:

  - `observation` is an array of 1 floating point number: current fish
    density
  - `reward` is a scalar float value
  - `action` is a scalar integer with only three possible values:
  - `0` - “maintain previous harvest”
  - `1` - “increase harvest by 20%”
  - `2` - “decrease harvest by 20%”

<!-- end list -->

``` python
time_step = env.reset()
print('Time step:')
```

    ## Time step:

``` python
print(time_step)
```

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([0.75]))

``` python
action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
```

    ## Next time step:

``` python
print(next_time_step)
```

    ## TimeStep(step_type=array(1, dtype=int32), reward=array(0.01, dtype=float32), discount=array(1., dtype=float32), observation=array([0.77394884]))

Usually two environments are instantiated: one for training and one for
evaluation.

``` python
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
```

Our environment, like most environments, is written in pure Python. This
is converted to TensorFlow using the `TFPyEnvironment` wrapper.

The original environment’s API uses Numpy arrays. The `TFPyEnvironment`
converts these to `Tensors` to make it compatible with Tensorflow agents
and policies.

``` python
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
```

## Agent

The algorithm used to solve an RL problem is represented by an `Agent`.
TF-Agents provides standard implementations of a variety of `Agents`,
including:

  - [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    (used in this
    tutorial)
  - [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
  - [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
  - [TD3](https://arxiv.org/pdf/1802.09477.pdf)
  - [PPO](https://arxiv.org/abs/1707.06347)
  - [SAC](https://arxiv.org/abs/1801.01290).

The DQN agent can be used in any environment which has a discrete action
space. At the heart of a DQN Agent is a `QNetwork`, a neural network
model that can learn to predict `QValues` (expected returns) for all
actions, given an observation from the environment.

Use `tf_agents.networks.q_network` to create a `QNetwork`, passing in
the `observation_spec`, `action_spec`, and a tuple describing the number
and size of the model’s hidden layers.

``` python
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
```

Now use `tf_agents.agents.dqn.dqn_agent` to instantiate a `DqnAgent`. In
addition to the `time_step_spec`, `action_spec` and the QNetwork, the
agent constructor also requires an optimizer (in this case,
`AdamOptimizer`), a loss function, and an integer step
counter.

``` python
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
```

    ## WARNING:tensorflow:Layer QNetwork is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    ## 
    ## If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    ## 
    ## To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    ## 
    ## WARNING:tensorflow:Layer TargetQNetwork is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    ## 
    ## If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    ## 
    ## To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

``` python
agent.initialize()
```

## Policies

A policy defines the way an agent acts in an environment. Typically, the
goal of reinforcement learning is to train the underlying model until
the policy produces the desired outcome.

In this tutorial:

  - The desired outcome is keeping the pole balanced upright over the
    cart.
  - The policy returns an action (left or right) for each `time_step`
    observation.

Agents contain two policies:

  - `agent.policy` — The main policy that is used for evaluation and
    deployment.
  - `agent.collect_policy` — A second policy that is used for data
    collection.

<!-- end list -->

``` python
eval_policy = agent.policy
collect_policy = agent.collect_policy
```

Policies can be created independently of agents. For example, use
`tf_agents.policies.random_tf_policy` to create a policy which will
randomly select an action for each
`time_step`.

``` python
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
```

To get an action from a policy, call the `policy.action(time_step)`
method. The `time_step` contains the observation from the environment.
This method returns a `PolicyStep`, which is a named tuple with three
components:

  - `action` — the action to be taken (in this case, `0`, `1` or `2`)
  - `state` — used for stateful (that is, RNN-based) policies
  - `info` — auxiliary data, such as log probabilities of actions

<!-- end list -->

``` python
example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load(env_name))

time_step = example_environment.reset()
random_policy.action(time_step)
```

    ## PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int64, numpy=array([90])>, state=(), info=())

## Metrics and Evaluation

The most common metric used to evaluate a policy is the average return.
The return is the sum of rewards obtained while running a policy in an
environment for an episode. Several episodes are run, creating an
average return.

The following function computes the average return of a policy, given
the policy, environment, and a number of episodes.

``` python
#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]
```

See also the metrics module for standard implementations of different
metrics:
<https://github.com/tensorflow/agents/tree/master/tf_agents/metrics>

Running this computation on the `random_policy` shows a baseline
performance in the environment.

``` python
compute_avg_return(eval_env, random_policy, num_eval_episodes)
```

    ## 0.7718277

## Replay Buffer

The replay buffer keeps track of data collected from the environment.
This tutorial uses
`tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer`,
as it is the most common.

The constructor requires the specs for the data it will be collecting.
This is available from the agent using the `collect_data_spec` method.
The batch size and maximum buffer length are also required.

``` python
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_max_length)

# For most agents, `collect_data_spec` is a named tuple called 
# `Trajectory`, containing the specs for observations, actions,
# rewards, and other items.
agent.collect_data_spec
```

    ## Trajectory(step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), observation=BoundedTensorSpec(shape=(1,), dtype=tf.float64, name='observation', minimum=array(0.), maximum=array(2.)), action=BoundedTensorSpec(shape=(), dtype=tf.int64, name='action', minimum=array(0), maximum=array(99)), policy_info=(), next_step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), reward=TensorSpec(shape=(), dtype=tf.float32, name='reward'), discount=BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)))

``` python
agent.collect_data_spec._fields
```

    ## ('step_type', 'observation', 'action', 'policy_info', 'next_step_type', 'reward', 'discount')

## Data Collection

Now execute the random policy in the environment for a few steps,
recording the data in the replay buffer.

``` python
#@test {"skip": true}
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=100)
```

This loop is so common in RL, that we provide standard implementations.
For more details see the drivers module.
<https://github.com/tensorflow/agents/blob/master/tf_agents/docs/python/tf_agents/drivers.md>
The replay buffer is now a collection of Trajectories.

For the curious: Uncomment to peel one of these off and inspect it.

``` python
# iter(replay_buffer.as_dataset()).next()
```

The agent needs access to the replay buffer. This is provided by
creating an iterable `tf.data.Dataset` pipeline which will feed data to
the agent.

Each row of the replay buffer only stores a single observation step. But
since the DQN Agent needs both the current and next observation to
compute the loss, the dataset pipeline will sample two adjacent rows for
each item in the batch (`num_steps=2`).

This dataset is also optimized by running parallel calls and prefetching
data.

``` python
# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)


dataset
```

    ## <PrefetchDataset shapes: (Trajectory(step_type=(64, 2), observation=(64, 2, 1), action=(64, 2), policy_info=(), next_step_type=(64, 2), reward=(64, 2), discount=(64, 2)), BufferInfo(ids=(64, 2), probabilities=(64,))), types: (Trajectory(step_type=tf.int32, observation=tf.float64, action=tf.int64, policy_info=(), next_step_type=tf.int32, reward=tf.float32, discount=tf.float32), BufferInfo(ids=tf.int64, probabilities=tf.float32))>

``` python
iterator = iter(dataset)

print(iterator)
```

    ## <tensorflow.python.data.ops.iterator_ops.OwnedIterator object at 0x7fd6300e7208>

For the curious:

Uncomment to see what the dataset iterator is feeding to the agent.
Compare this representation of replay data to the collection of
individual trajectories shown earlier.

``` python
# iterator.next()
```

## Training the agent

Two things must happen during the training loop:

  - collect data from the environment
  - use that data to train the agent’s neural network(s)

This example also periodicially evaluates the policy and prints the
current
score.

``` python
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
```

    ## <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=0>

``` python
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
```

    ## step = 200: loss = 2.199673891067505
    ## step = 400: loss = 103.73721313476562
    ## step = 600: loss = 190.2124786376953
    ## step = 800: loss = 2243.575439453125
    ## step = 1000: loss = 1098.927490234375
    ## step = 1000: Average Return = 0.0
    ## step = 1200: loss = 8778.625
    ## step = 1400: loss = 28637.037109375
    ## step = 1600: loss = 24818.05078125
    ## step = 1800: loss = 10253.701171875
    ## step = 2000: loss = 109817.390625
    ## step = 2000: Average Return = 0.0
    ## step = 2200: loss = 91016.859375
    ## step = 2400: loss = 65640.0
    ## step = 2600: loss = 89303.671875
    ## step = 2800: loss = 101929.8671875
    ## step = 3000: loss = 261762.125
    ## step = 3000: Average Return = 0.0
    ## step = 3200: loss = 36198.34765625
    ## step = 3400: loss = 358648.125
    ## step = 3600: loss = 745571.125
    ## step = 3800: loss = 216055.78125
    ## step = 4000: loss = 556356.625
    ## step = 4000: Average Return = 0.0
    ## step = 4200: loss = 612250.9375
    ## step = 4400: loss = 828404.875
    ## step = 4600: loss = 360389.03125
    ## step = 4800: loss = 632911.875
    ## step = 5000: loss = 1021392.5625
    ## step = 5000: Average Return = 0.0
    ## step = 5200: loss = 827330.5
    ## step = 5400: loss = 3242914.0
    ## step = 5600: loss = 1359218.125
    ## step = 5800: loss = 2616741.5
    ## step = 6000: loss = 3301554.0
    ## step = 6000: Average Return = 0.0
    ## step = 6200: loss = 1835041.25
    ## step = 6400: loss = 1741910.625
    ## step = 6600: loss = 1136261.875
    ## step = 6800: loss = 1032870.5
    ## step = 7000: loss = 1572114.5
    ## step = 7000: Average Return = 0.0
    ## step = 7200: loss = 3264845.5
    ## step = 7400: loss = 2049979.75
    ## step = 7600: loss = 6819020.0
    ## step = 7800: loss = 9035394.0
    ## step = 8000: loss = 4540479.0
    ## step = 8000: Average Return = 0.0
    ## step = 8200: loss = 2819676.0
    ## step = 8400: loss = 7421689.0
    ## step = 8600: loss = 3771877.5
    ## step = 8800: loss = 5360086.5
    ## step = 9000: loss = 6434050.5
    ## step = 9000: Average Return = 0.0
    ## step = 9200: loss = 8097747.0
    ## step = 9400: loss = 5385697.5
    ## step = 9600: loss = 6219741.0
    ## step = 9800: loss = 13774946.0
    ## step = 10000: loss = 11632914.0
    ## step = 10000: Average Return = 0.0
    ## step = 10200: loss = 10319557.0
    ## step = 10400: loss = 5971708.0
    ## step = 10600: loss = 8226798.0
    ## step = 10800: loss = 10899183.0
    ## step = 11000: loss = 23318750.0
    ## step = 11000: Average Return = 0.0
    ## step = 11200: loss = 3884928.0
    ## step = 11400: loss = 15777390.0
    ## step = 11600: loss = 27515076.0
    ## step = 11800: loss = 9590346.0
    ## step = 12000: loss = 16960436.0
    ## step = 12000: Average Return = 0.0
    ## step = 12200: loss = 16255565.0
    ## step = 12400: loss = 12245396.0
    ## step = 12600: loss = 38586124.0
    ## step = 12800: loss = 23068188.0
    ## step = 13000: loss = 34107648.0
    ## step = 13000: Average Return = 0.0
    ## step = 13200: loss = 44387012.0
    ## step = 13400: loss = 40241904.0
    ## step = 13600: loss = 16616557.0
    ## step = 13800: loss = 27322536.0
    ## step = 14000: loss = 9831484.0
    ## step = 14000: Average Return = 0.0
    ## step = 14200: loss = 24219440.0
    ## step = 14400: loss = 15534456.0
    ## step = 14600: loss = 29535244.0
    ## step = 14800: loss = 5475612.0
    ## step = 15000: loss = 29763912.0
    ## step = 15000: Average Return = 0.0
    ## step = 15200: loss = 57501576.0
    ## step = 15400: loss = 39716320.0
    ## step = 15600: loss = 19885264.0
    ## step = 15800: loss = 35268244.0
    ## step = 16000: loss = 16533490.0
    ## step = 16000: Average Return = 0.0
    ## step = 16200: loss = 48501568.0
    ## step = 16400: loss = 23297814.0
    ## step = 16600: loss = 44900660.0
    ## step = 16800: loss = 46429628.0
    ## step = 17000: loss = 55584864.0
    ## step = 17000: Average Return = 0.0
    ## step = 17200: loss = 20006034.0
    ## step = 17400: loss = 59916244.0
    ## step = 17600: loss = 54762208.0
    ## step = 17800: loss = 10430692.0
    ## step = 18000: loss = 45608444.0
    ## step = 18000: Average Return = 0.0
    ## step = 18200: loss = 71467568.0
    ## step = 18400: loss = 59213744.0
    ## step = 18600: loss = 61567880.0
    ## step = 18800: loss = 64041704.0
    ## step = 19000: loss = 30546558.0
    ## step = 19000: Average Return = 0.0
    ## step = 19200: loss = 74718336.0
    ## step = 19400: loss = 18150386.0
    ## step = 19600: loss = 47213488.0
    ## step = 19800: loss = 167422448.0
    ## step = 20000: loss = 72479120.0
    ## step = 20000: Average Return = 0.0
    ## step = 20200: loss = 13713116.0
    ## step = 20400: loss = 89129328.0
    ## step = 20600: loss = 131780312.0
    ## step = 20800: loss = 54701680.0
    ## step = 21000: loss = 29025368.0
    ## step = 21000: Average Return = 0.0
    ## step = 21200: loss = 80229280.0
    ## step = 21400: loss = 151928224.0
    ## step = 21600: loss = 223156144.0
    ## step = 21800: loss = 177817840.0
    ## step = 22000: loss = 165811584.0
    ## step = 22000: Average Return = 0.0
    ## step = 22200: loss = 141308384.0
    ## step = 22400: loss = 74965968.0
    ## step = 22600: loss = 104604032.0
    ## step = 22800: loss = 151345856.0
    ## step = 23000: loss = 148199168.0
    ## step = 23000: Average Return = 0.0
    ## step = 23200: loss = 166690688.0
    ## step = 23400: loss = 103695872.0
    ## step = 23600: loss = 121340328.0
    ## step = 23800: loss = 221455056.0
    ## step = 24000: loss = 195751840.0
    ## step = 24000: Average Return = 0.0
    ## step = 24200: loss = 109027400.0
    ## step = 24400: loss = 146308992.0
    ## step = 24600: loss = 267551648.0
    ## step = 24800: loss = 163786032.0
    ## step = 25000: loss = 342082176.0
    ## step = 25000: Average Return = 0.0
    ## step = 25200: loss = 239810704.0
    ## step = 25400: loss = 178490176.0
    ## step = 25600: loss = 108686472.0
    ## step = 25800: loss = 344716480.0
    ## step = 26000: loss = 205693552.0
    ## step = 26000: Average Return = 0.0
    ## step = 26200: loss = 70003840.0
    ## step = 26400: loss = 61925312.0
    ## step = 26600: loss = 174440464.0
    ## step = 26800: loss = 286901056.0
    ## step = 27000: loss = 350478144.0
    ## step = 27000: Average Return = 0.0
    ## step = 27200: loss = 104863712.0
    ## step = 27400: loss = 504961600.0
    ## step = 27600: loss = 177284480.0
    ## step = 27800: loss = 180933776.0
    ## step = 28000: loss = 366113408.0
    ## step = 28000: Average Return = 0.0
    ## step = 28200: loss = 360626016.0
    ## step = 28400: loss = 305954624.0
    ## step = 28600: loss = 206581024.0
    ## step = 28800: loss = 427547776.0
    ## step = 29000: loss = 62481556.0
    ## step = 29000: Average Return = 0.0
    ## step = 29200: loss = 292962752.0
    ## step = 29400: loss = 335302464.0
    ## step = 29600: loss = 240607328.0
    ## step = 29800: loss = 237014224.0
    ## step = 30000: loss = 226613888.0
    ## step = 30000: Average Return = 0.0
    ## step = 30200: loss = 523124160.0
    ## step = 30400: loss = 486645536.0
    ## step = 30600: loss = 180111504.0
    ## step = 30800: loss = 306757504.0
    ## step = 31000: loss = 218015728.0
    ## step = 31000: Average Return = 0.0
    ## step = 31200: loss = 93542224.0
    ## step = 31400: loss = 410020192.0
    ## step = 31600: loss = 555310720.0
    ## step = 31800: loss = 637886656.0
    ## step = 32000: loss = 198976240.0
    ## step = 32000: Average Return = 0.0
    ## step = 32200: loss = 951879808.0
    ## step = 32400: loss = 641860544.0
    ## step = 32600: loss = 549563904.0
    ## step = 32800: loss = 696396672.0
    ## step = 33000: loss = 520578624.0
    ## step = 33000: Average Return = 0.0
    ## step = 33200: loss = 210056976.0
    ## step = 33400: loss = 286924800.0
    ## step = 33600: loss = 289885696.0
    ## step = 33800: loss = 581803392.0
    ## step = 34000: loss = 943620608.0
    ## step = 34000: Average Return = 0.0
    ## step = 34200: loss = 750681472.0
    ## step = 34400: loss = 374401600.0
    ## step = 34600: loss = 1353087616.0
    ## step = 34800: loss = 393599328.0
    ## step = 35000: loss = 386025792.0
    ## step = 35000: Average Return = 0.0
    ## step = 35200: loss = 432688864.0
    ## step = 35400: loss = 1429039872.0
    ## step = 35600: loss = 220304320.0
    ## step = 35800: loss = 633410240.0
    ## step = 36000: loss = 482099680.0
    ## step = 36000: Average Return = 0.0
    ## step = 36200: loss = 1163629056.0
    ## step = 36400: loss = 313813312.0
    ## step = 36600: loss = 862824704.0
    ## step = 36800: loss = 668592192.0
    ## step = 37000: loss = 426174592.0
    ## step = 37000: Average Return = 0.0
    ## step = 37200: loss = 417246912.0
    ## step = 37400: loss = 217023872.0
    ## step = 37600: loss = 994348352.0
    ## step = 37800: loss = 595601280.0
    ## step = 38000: loss = 324498272.0
    ## step = 38000: Average Return = 0.0
    ## step = 38200: loss = 675225216.0
    ## step = 38400: loss = 840421952.0
    ## step = 38600: loss = 1076596736.0
    ## step = 38800: loss = 804861056.0
    ## step = 39000: loss = 893653888.0
    ## step = 39000: Average Return = 0.0
    ## step = 39200: loss = 468097280.0
    ## step = 39400: loss = 13696508.0
    ## step = 39600: loss = 1134885376.0
    ## step = 39800: loss = 617138176.0
    ## step = 40000: loss = 783955712.0
    ## step = 40000: Average Return = 0.0
    ## step = 40200: loss = 1249622016.0
    ## step = 40400: loss = 657582848.0
    ## step = 40600: loss = 607807936.0
    ## step = 40800: loss = 645411584.0
    ## step = 41000: loss = 2024281344.0
    ## step = 41000: Average Return = 0.0
    ## step = 41200: loss = 681612160.0
    ## step = 41400: loss = 1326368000.0
    ## step = 41600: loss = 560814016.0
    ## step = 41800: loss = 608793088.0
    ## step = 42000: loss = 999707008.0
    ## step = 42000: Average Return = 0.0
    ## step = 42200: loss = 780694720.0
    ## step = 42400: loss = 2008271104.0
    ## step = 42600: loss = 1300031744.0
    ## step = 42800: loss = 929335296.0
    ## step = 43000: loss = 43938368.0
    ## step = 43000: Average Return = 0.0
    ## step = 43200: loss = 1097695744.0
    ## step = 43400: loss = 280833472.0
    ## step = 43600: loss = 811136448.0
    ## step = 43800: loss = 1804574336.0
    ## step = 44000: loss = 1167091712.0
    ## step = 44000: Average Return = 0.0
    ## step = 44200: loss = 1374534144.0
    ## step = 44400: loss = 2004543616.0
    ## step = 44600: loss = 795712448.0
    ## step = 44800: loss = 1862656512.0
    ## step = 45000: loss = 30732196.0
    ## step = 45000: Average Return = 0.0
    ## step = 45200: loss = 1071159808.0
    ## step = 45400: loss = 886357824.0
    ## step = 45600: loss = 1656015232.0
    ## step = 45800: loss = 1049051968.0
    ## step = 46000: loss = 2416714752.0
    ## step = 46000: Average Return = 0.0
    ## step = 46200: loss = 718204160.0
    ## step = 46400: loss = 1618198016.0
    ## step = 46600: loss = 966866880.0
    ## step = 46800: loss = 1922339712.0
    ## step = 47000: loss = 1565707008.0
    ## step = 47000: Average Return = 0.0
    ## step = 47200: loss = 1370854144.0
    ## step = 47400: loss = 524698752.0
    ## step = 47600: loss = 458070240.0
    ## step = 47800: loss = 1470893824.0
    ## step = 48000: loss = 2213808128.0
    ## step = 48000: Average Return = 0.0
    ## step = 48200: loss = 1677917184.0
    ## step = 48400: loss = 2489637120.0
    ## step = 48600: loss = 1309427712.0
    ## step = 48800: loss = 1719073408.0
    ## step = 49000: loss = 1898056576.0
    ## step = 49000: Average Return = 0.0
    ## step = 49200: loss = 1462111488.0
    ## step = 49400: loss = 1596072192.0
    ## step = 49600: loss = 2818180608.0
    ## step = 49800: loss = 2034749952.0
    ## step = 50000: loss = 1624026752.0
    ## step = 50000: Average Return = 0.0
    ## step = 50200: loss = 4362924032.0
    ## step = 50400: loss = 2884927488.0
    ## step = 50600: loss = 2345803008.0
    ## step = 50800: loss = 973765312.0
    ## step = 51000: loss = 4151725568.0
    ## step = 51000: Average Return = 0.0
    ## step = 51200: loss = 1097351296.0
    ## step = 51400: loss = 1818813440.0
    ## step = 51600: loss = 887280832.0
    ## step = 51800: loss = 415584352.0
    ## step = 52000: loss = 2993039104.0
    ## step = 52000: Average Return = 0.0
    ## step = 52200: loss = 3312346112.0
    ## step = 52400: loss = 643567552.0
    ## step = 52600: loss = 3710742528.0
    ## step = 52800: loss = 1575902592.0
    ## step = 53000: loss = 2335956480.0
    ## step = 53000: Average Return = 0.0
    ## step = 53200: loss = 79765568.0
    ## step = 53400: loss = 1986308608.0
    ## step = 53600: loss = 2692083200.0
    ## step = 53800: loss = 66113224.0
    ## step = 54000: loss = 2695820800.0
    ## step = 54000: Average Return = 0.0
    ## step = 54200: loss = 468463872.0
    ## step = 54400: loss = 4250940928.0
    ## step = 54600: loss = 2499123968.0
    ## step = 54800: loss = 594582208.0
    ## step = 55000: loss = 95210088.0
    ## step = 55000: Average Return = 0.0
    ## step = 55200: loss = 4117344256.0
    ## step = 55400: loss = 2409809408.0
    ## step = 55600: loss = 647306560.0
    ## step = 55800: loss = 3681341440.0
    ## step = 56000: loss = 1109577344.0
    ## step = 56000: Average Return = 0.0
    ## step = 56200: loss = 2107815936.0
    ## step = 56400: loss = 2202912768.0
    ## step = 56600: loss = 3035261440.0
    ## step = 56800: loss = 1006375744.0
    ## step = 57000: loss = 3356902912.0
    ## step = 57000: Average Return = 0.0
    ## step = 57200: loss = 5211315712.0
    ## step = 57400: loss = 3111808000.0
    ## step = 57600: loss = 2555298816.0
    ## step = 57800: loss = 4328980480.0
    ## step = 58000: loss = 2998825728.0
    ## step = 58000: Average Return = 0.0
    ## step = 58200: loss = 1404276224.0
    ## step = 58400: loss = 2573123584.0
    ## step = 58600: loss = 2247126016.0
    ## step = 58800: loss = 7988506624.0
    ## step = 59000: loss = 3229075968.0
    ## step = 59000: Average Return = 0.0
    ## step = 59200: loss = 2514169600.0
    ## step = 59400: loss = 4343331328.0
    ## step = 59600: loss = 4464505344.0
    ## step = 59800: loss = 6935249920.0
    ## step = 60000: loss = 2879193344.0
    ## step = 60000: Average Return = 0.0
    ## step = 60200: loss = 3347277312.0
    ## step = 60400: loss = 6602467328.0
    ## step = 60600: loss = 5145861632.0
    ## step = 60800: loss = 3340979968.0
    ## step = 61000: loss = 3632968192.0
    ## step = 61000: Average Return = 0.0
    ## step = 61200: loss = 3650221568.0
    ## step = 61400: loss = 1062186368.0
    ## step = 61600: loss = 2290195968.0
    ## step = 61800: loss = 5638145024.0
    ## step = 62000: loss = 4362870272.0
    ## step = 62000: Average Return = 0.0
    ## step = 62200: loss = 8307301376.0
    ## step = 62400: loss = 8332846080.0
    ## step = 62600: loss = 4793333248.0
    ## step = 62800: loss = 3855002112.0
    ## step = 63000: loss = 2527104000.0
    ## step = 63000: Average Return = 0.0
    ## step = 63200: loss = 6611593216.0
    ## step = 63400: loss = 4328610816.0
    ## step = 63600: loss = 3831912960.0
    ## step = 63800: loss = 6659737600.0
    ## step = 64000: loss = 6276548608.0
    ## step = 64000: Average Return = 0.0
    ## step = 64200: loss = 2627918592.0
    ## step = 64400: loss = 5600144384.0
    ## step = 64600: loss = 5222960640.0
    ## step = 64800: loss = 3299892224.0
    ## step = 65000: loss = 7116032000.0
    ## step = 65000: Average Return = 0.0
    ## step = 65200: loss = 1937961600.0
    ## step = 65400: loss = 1555311744.0
    ## step = 65600: loss = 6204658688.0
    ## step = 65800: loss = 1708470784.0
    ## step = 66000: loss = 9331832832.0
    ## step = 66000: Average Return = 0.0
    ## step = 66200: loss = 6671504896.0
    ## step = 66400: loss = 8335513088.0
    ## step = 66600: loss = 6046582272.0
    ## step = 66800: loss = 6417027072.0
    ## step = 67000: loss = 7294383104.0
    ## step = 67000: Average Return = 0.0
    ## step = 67200: loss = 8206434304.0
    ## step = 67400: loss = 2615333120.0
    ## step = 67600: loss = 6958786048.0
    ## step = 67800: loss = 6629274624.0
    ## step = 68000: loss = 6408281088.0
    ## step = 68000: Average Return = 0.0
    ## step = 68200: loss = 4758253568.0
    ## step = 68400: loss = 9096609792.0
    ## step = 68600: loss = 5708657152.0
    ## step = 68800: loss = 7722753024.0
    ## step = 69000: loss = 9038543872.0
    ## step = 69000: Average Return = 0.0
    ## step = 69200: loss = 6942559744.0
    ## step = 69400: loss = 4505246720.0
    ## step = 69600: loss = 3977304832.0
    ## step = 69800: loss = 7860055040.0
    ## step = 70000: loss = 5419261952.0
    ## step = 70000: Average Return = 0.0
    ## step = 70200: loss = 4750429696.0
    ## step = 70400: loss = 2405319936.0
    ## step = 70600: loss = 3889275904.0
    ## step = 70800: loss = 11051153408.0
    ## step = 71000: loss = 7076102656.0
    ## step = 71000: Average Return = 0.0
    ## step = 71200: loss = 7701137408.0
    ## step = 71400: loss = 7072946688.0
    ## step = 71600: loss = 7183759360.0
    ## step = 71800: loss = 12489697280.0
    ## step = 72000: loss = 2432728576.0
    ## step = 72000: Average Return = 0.0
    ## step = 72200: loss = 12361398272.0
    ## step = 72400: loss = 6689595392.0
    ## step = 72600: loss = 8863902720.0
    ## step = 72800: loss = 11240600576.0
    ## step = 73000: loss = 10102249472.0
    ## step = 73000: Average Return = 0.0
    ## step = 73200: loss = 13172527104.0
    ## step = 73400: loss = 2272565504.0
    ## step = 73600: loss = 4486300672.0
    ## step = 73800: loss = 5495304192.0
    ## step = 74000: loss = 9818251264.0
    ## step = 74000: Average Return = 0.0
    ## step = 74200: loss = 9408398336.0
    ## step = 74400: loss = 7941515264.0
    ## step = 74600: loss = 2637157120.0
    ## step = 74800: loss = 12185094144.0
    ## step = 75000: loss = 3280135680.0
    ## step = 75000: Average Return = 0.0
    ## step = 75200: loss = 154545008.0
    ## step = 75400: loss = 3472931840.0
    ## step = 75600: loss = 9452279808.0
    ## step = 75800: loss = 4634096128.0
    ## step = 76000: loss = 14446415872.0
    ## step = 76000: Average Return = 0.0
    ## step = 76200: loss = 12934041600.0
    ## step = 76400: loss = 11964385280.0
    ## step = 76600: loss = 15362577408.0
    ## step = 76800: loss = 11918375936.0
    ## step = 77000: loss = 5996721664.0
    ## step = 77000: Average Return = 0.0
    ## step = 77200: loss = 7627614208.0
    ## step = 77400: loss = 13679133696.0
    ## step = 77600: loss = 9239745536.0
    ## step = 77800: loss = 14784583680.0
    ## step = 78000: loss = 14648045568.0
    ## step = 78000: Average Return = 0.0
    ## step = 78200: loss = 4814038528.0
    ## step = 78400: loss = 4988324352.0
    ## step = 78600: loss = 12812127232.0
    ## step = 78800: loss = 7720884224.0
    ## step = 79000: loss = 11688919040.0
    ## step = 79000: Average Return = 0.0
    ## step = 79200: loss = 16725122048.0
    ## step = 79400: loss = 14288750592.0
    ## step = 79600: loss = 6425968640.0
    ## step = 79800: loss = 9192247296.0
    ## step = 80000: loss = 14926592000.0
    ## step = 80000: Average Return = 0.0
    ## step = 80200: loss = 7965343232.0
    ## step = 80400: loss = 18593306624.0
    ## step = 80600: loss = 8333208576.0
    ## step = 80800: loss = 6599598080.0
    ## step = 81000: loss = 16270869504.0
    ## step = 81000: Average Return = 0.0
    ## step = 81200: loss = 11843500032.0
    ## step = 81400: loss = 15814321152.0
    ## step = 81600: loss = 17434986496.0
    ## step = 81800: loss = 10668790784.0
    ## step = 82000: loss = 10579509248.0
    ## step = 82000: Average Return = 0.0
    ## step = 82200: loss = 6633719808.0
    ## step = 82400: loss = 16878614528.0
    ## step = 82600: loss = 24251041792.0
    ## step = 82800: loss = 10313808896.0
    ## step = 83000: loss = 11525462016.0
    ## step = 83000: Average Return = 0.0
    ## step = 83200: loss = 7025973248.0
    ## step = 83400: loss = 13473172480.0
    ## step = 83600: loss = 4248977920.0
    ## step = 83800: loss = 11592765440.0
    ## step = 84000: loss = 17203742720.0
    ## step = 84000: Average Return = 0.0
    ## step = 84200: loss = 6113593856.0
    ## step = 84400: loss = 13319336960.0
    ## step = 84600: loss = 8753533952.0
    ## step = 84800: loss = 8580786688.0
    ## step = 85000: loss = 3991901440.0
    ## step = 85000: Average Return = 0.0
    ## step = 85200: loss = 13389507584.0
    ## step = 85400: loss = 22309140480.0
    ## step = 85600: loss = 5015789568.0
    ## step = 85800: loss = 10026891264.0
    ## step = 86000: loss = 14076596224.0
    ## step = 86000: Average Return = 0.0
    ## step = 86200: loss = 10775256064.0
    ## step = 86400: loss = 3599586816.0
    ## step = 86600: loss = 14864858112.0
    ## step = 86800: loss = 23858397184.0
    ## step = 87000: loss = 13934418944.0
    ## step = 87000: Average Return = 0.0
    ## step = 87200: loss = 9334198272.0
    ## step = 87400: loss = 25108152320.0
    ## step = 87600: loss = 12959944704.0
    ## step = 87800: loss = 22223063040.0
    ## step = 88000: loss = 14477839360.0
    ## step = 88000: Average Return = 0.0
    ## step = 88200: loss = 8305678336.0
    ## step = 88400: loss = 9145852928.0
    ## step = 88600: loss = 17707032576.0
    ## step = 88800: loss = 17382512640.0
    ## step = 89000: loss = 11954148352.0
    ## step = 89000: Average Return = 0.0
    ## step = 89200: loss = 9215324160.0
    ## step = 89400: loss = 15202920448.0
    ## step = 89600: loss = 22175805440.0
    ## step = 89800: loss = 14488607744.0
    ## step = 90000: loss = 6140126720.0
    ## step = 90000: Average Return = 0.0
    ## step = 90200: loss = 10232160256.0
    ## step = 90400: loss = 20455933952.0
    ## step = 90600: loss = 11756840960.0
    ## step = 90800: loss = 33905653760.0
    ## step = 91000: loss = 17237237760.0
    ## step = 91000: Average Return = 0.0
    ## step = 91200: loss = 12692696064.0
    ## step = 91400: loss = 23879018496.0
    ## step = 91600: loss = 27568631808.0
    ## step = 91800: loss = 24622778368.0
    ## step = 92000: loss = 10811828224.0
    ## step = 92000: Average Return = 0.0
    ## step = 92200: loss = 15323122688.0
    ## step = 92400: loss = 11804863488.0
    ## step = 92600: loss = 31106842624.0
    ## step = 92800: loss = 30335983616.0
    ## step = 93000: loss = 10852718592.0
    ## step = 93000: Average Return = 0.0
    ## step = 93200: loss = 20160647168.0
    ## step = 93400: loss = 32545710080.0
    ## step = 93600: loss = 20523028480.0
    ## step = 93800: loss = 22321127424.0
    ## step = 94000: loss = 39895687168.0
    ## step = 94000: Average Return = 0.0
    ## step = 94200: loss = 37771608064.0
    ## step = 94400: loss = 23403675648.0
    ## step = 94600: loss = 26515322880.0
    ## step = 94800: loss = 7263790080.0
    ## step = 95000: loss = 22403407872.0
    ## step = 95000: Average Return = 0.0
    ## step = 95200: loss = 16816777216.0
    ## step = 95400: loss = 18153787392.0
    ## step = 95600: loss = 22225561600.0
    ## step = 95800: loss = 26956652544.0
    ## step = 96000: loss = 6685348864.0
    ## step = 96000: Average Return = 0.0
    ## step = 96200: loss = 12230606848.0
    ## step = 96400: loss = 24351600640.0
    ## step = 96600: loss = 18017570816.0
    ## step = 96800: loss = 29229697024.0
    ## step = 97000: loss = 3594561024.0
    ## step = 97000: Average Return = 0.0
    ## step = 97200: loss = 20199487488.0
    ## step = 97400: loss = 3125298176.0
    ## step = 97600: loss = 6584211456.0
    ## step = 97800: loss = 21172111360.0
    ## step = 98000: loss = 4242203136.0
    ## step = 98000: Average Return = 0.0
    ## step = 98200: loss = 24089485312.0
    ## step = 98400: loss = 25779040256.0
    ## step = 98600: loss = 17012228096.0
    ## step = 98800: loss = 38547988480.0
    ## step = 99000: loss = 6177650688.0
    ## step = 99000: Average Return = 0.0
    ## step = 99200: loss = 811021952.0
    ## step = 99400: loss = 21167026176.0
    ## step = 99600: loss = 10869255168.0
    ## step = 99800: loss = 31881742336.0
    ## step = 100000: loss = 26116890624.0
    ## step = 100000: Average Return = 0.0
    ## step = 100200: loss = 9139301376.0
    ## step = 100400: loss = 9711514624.0
    ## step = 100600: loss = 18686840832.0
    ## step = 100800: loss = 15334707200.0
    ## step = 101000: loss = 36784365568.0
    ## step = 101000: Average Return = 0.0
    ## step = 101200: loss = 15915894784.0
    ## step = 101400: loss = 8637095936.0
    ## step = 101600: loss = 15090798592.0
    ## step = 101800: loss = 22089609216.0
    ## step = 102000: loss = 33794828288.0
    ## step = 102000: Average Return = 0.0
    ## step = 102200: loss = 51723362304.0
    ## step = 102400: loss = 10178430976.0
    ## step = 102600: loss = 40448045056.0
    ## step = 102800: loss = 8073037312.0
    ## step = 103000: loss = 34084433920.0
    ## step = 103000: Average Return = 0.0
    ## step = 103200: loss = 7823158272.0
    ## step = 103400: loss = 7858301952.0
    ## step = 103600: loss = 6427041280.0
    ## step = 103800: loss = 37109153792.0
    ## step = 104000: loss = 26489782272.0
    ## step = 104000: Average Return = 0.0
    ## step = 104200: loss = 23647367168.0
    ## step = 104400: loss = 23368833024.0
    ## step = 104600: loss = 48726855680.0
    ## step = 104800: loss = 16680371200.0
    ## step = 105000: loss = 27895429120.0
    ## step = 105000: Average Return = 0.0
    ## step = 105200: loss = 15002530816.0
    ## step = 105400: loss = 20979279872.0
    ## step = 105600: loss = 35711381504.0
    ## step = 105800: loss = 24280096768.0
    ## step = 106000: loss = 30323761152.0
    ## step = 106000: Average Return = 0.0
    ## step = 106200: loss = 70420365312.0
    ## step = 106400: loss = 15035052032.0
    ## step = 106600: loss = 17644666880.0
    ## step = 106800: loss = 42401767424.0
    ## step = 107000: loss = 15728174080.0
    ## step = 107000: Average Return = 0.0
    ## step = 107200: loss = 15522166784.0
    ## step = 107400: loss = 23954227200.0
    ## step = 107600: loss = 43185676288.0
    ## step = 107800: loss = 1176215040.0
    ## step = 108000: loss = 31231938560.0
    ## step = 108000: Average Return = 0.0
    ## step = 108200: loss = 18186975232.0
    ## step = 108400: loss = 10226829312.0
    ## step = 108600: loss = 16990505984.0
    ## step = 108800: loss = 25036546048.0
    ## step = 109000: loss = 38091132928.0
    ## step = 109000: Average Return = 0.0
    ## step = 109200: loss = 35651559424.0
    ## step = 109400: loss = 26864248832.0
    ## step = 109600: loss = 18518142976.0
    ## step = 109800: loss = 5813954048.0
    ## step = 110000: loss = 50525458432.0
    ## step = 110000: Average Return = 0.0
    ## step = 110200: loss = 32433727488.0
    ## step = 110400: loss = 44281782272.0
    ## step = 110600: loss = 29711468544.0
    ## step = 110800: loss = 35012857856.0
    ## step = 111000: loss = 35458662400.0
    ## step = 111000: Average Return = 0.0
    ## step = 111200: loss = 25356818432.0
    ## step = 111400: loss = 42718588928.0
    ## step = 111600: loss = 31399806976.0
    ## step = 111800: loss = 32105476096.0
    ## step = 112000: loss = 94095794176.0
    ## step = 112000: Average Return = 0.0
    ## step = 112200: loss = 27127384064.0
    ## step = 112400: loss = 21514258432.0
    ## step = 112600: loss = 29932531712.0
    ## step = 112800: loss = 27090343936.0
    ## step = 113000: loss = 52568899584.0
    ## step = 113000: Average Return = 0.0
    ## step = 113200: loss = 52477030400.0
    ## step = 113400: loss = 15773344768.0
    ## step = 113600: loss = 25618370560.0
    ## step = 113800: loss = 54636552192.0
    ## step = 114000: loss = 48166064128.0
    ## step = 114000: Average Return = 0.0
    ## step = 114200: loss = 68517339136.0
    ## step = 114400: loss = 48530391040.0
    ## step = 114600: loss = 48058818560.0
    ## step = 114800: loss = 38582837248.0
    ## step = 115000: loss = 69233344512.0
    ## step = 115000: Average Return = 0.0
    ## step = 115200: loss = 36409327616.0
    ## step = 115400: loss = 18697650176.0
    ## step = 115600: loss = 55046373376.0
    ## step = 115800: loss = 25319180288.0
    ## step = 116000: loss = 43891208192.0
    ## step = 116000: Average Return = 0.0
    ## step = 116200: loss = 11522152448.0
    ## step = 116400: loss = 63174287360.0
    ## step = 116600: loss = 43330543616.0
    ## step = 116800: loss = 55726657536.0
    ## step = 117000: loss = 36926767104.0
    ## step = 117000: Average Return = 0.0
    ## step = 117200: loss = 38040027136.0
    ## step = 117400: loss = 76279889920.0
    ## step = 117600: loss = 20662345728.0
    ## step = 117800: loss = 62365171712.0
    ## step = 118000: loss = 82209120256.0
    ## step = 118000: Average Return = 0.0
    ## step = 118200: loss = 20011898880.0
    ## step = 118400: loss = 37692473344.0
    ## step = 118600: loss = 48338886656.0
    ## step = 118800: loss = 41355677696.0
    ## step = 119000: loss = 69395333120.0
    ## step = 119000: Average Return = 0.0
    ## step = 119200: loss = 75304681472.0
    ## step = 119400: loss = 31703756800.0
    ## step = 119600: loss = 11451839488.0
    ## step = 119800: loss = 65441042432.0
    ## step = 120000: loss = 32896299008.0
    ## step = 120000: Average Return = 0.0
    ## step = 120200: loss = 22109263872.0
    ## step = 120400: loss = 59325562880.0
    ## step = 120600: loss = 22604754944.0
    ## step = 120800: loss = 16158601216.0
    ## step = 121000: loss = 33461075968.0
    ## step = 121000: Average Return = 0.0
    ## step = 121200: loss = 63427129344.0
    ## step = 121400: loss = 47472893952.0
    ## step = 121600: loss = 80903110656.0
    ## step = 121800: loss = 63321554944.0
    ## step = 122000: loss = 31959928832.0
    ## step = 122000: Average Return = 0.0
    ## step = 122200: loss = 34715537408.0
    ## step = 122400: loss = 38179971072.0
    ## step = 122600: loss = 53566595072.0
    ## step = 122800: loss = 60500705280.0
    ## step = 123000: loss = 39457333248.0
    ## step = 123000: Average Return = 0.0
    ## step = 123200: loss = 60149792768.0
    ## step = 123400: loss = 67317628928.0
    ## step = 123600: loss = 45658783744.0
    ## step = 123800: loss = 48250372096.0
    ## step = 124000: loss = 51325845504.0
    ## step = 124000: Average Return = 0.0
    ## step = 124200: loss = 27994558464.0
    ## step = 124400: loss = 46104444928.0
    ## step = 124600: loss = 24921628672.0
    ## step = 124800: loss = 46305099776.0
    ## step = 125000: loss = 11058163712.0
    ## step = 125000: Average Return = 0.0
    ## step = 125200: loss = 71834959872.0
    ## step = 125400: loss = 48006569984.0
    ## step = 125600: loss = 41111916544.0
    ## step = 125800: loss = 48253632512.0
    ## step = 126000: loss = 76638797824.0
    ## step = 126000: Average Return = 0.0
    ## step = 126200: loss = 62794809344.0
    ## step = 126400: loss = 25264252928.0
    ## step = 126600: loss = 82104655872.0
    ## step = 126800: loss = 65390723072.0
    ## step = 127000: loss = 130948759552.0
    ## step = 127000: Average Return = 0.0
    ## step = 127200: loss = 43653095424.0
    ## step = 127400: loss = 29917157376.0
    ## step = 127600: loss = 24053743616.0
    ## step = 127800: loss = 42684452864.0
    ## step = 128000: loss = 17164234752.0
    ## step = 128000: Average Return = 0.0
    ## step = 128200: loss = 53701550080.0
    ## step = 128400: loss = 78188298240.0
    ## step = 128600: loss = 159312543744.0
    ## step = 128800: loss = 64481333248.0
    ## step = 129000: loss = 70014468096.0
    ## step = 129000: Average Return = 0.0
    ## step = 129200: loss = 51312934912.0
    ## step = 129400: loss = 86432825344.0
    ## step = 129600: loss = 87390887936.0
    ## step = 129800: loss = 43070844928.0
    ## step = 130000: loss = 23011934208.0
    ## step = 130000: Average Return = 0.0
    ## step = 130200: loss = 28024156160.0
    ## step = 130400: loss = 35992424448.0
    ## step = 130600: loss = 39795417088.0
    ## step = 130800: loss = 82034409472.0
    ## step = 131000: loss = 32508448768.0
    ## step = 131000: Average Return = 0.0
    ## step = 131200: loss = 51596214272.0
    ## step = 131400: loss = 130679635968.0
    ## step = 131600: loss = 30794983424.0
    ## step = 131800: loss = 45711761408.0
    ## step = 132000: loss = 60816539648.0
    ## step = 132000: Average Return = 0.0
    ## step = 132200: loss = 48174989312.0
    ## step = 132400: loss = 47818518528.0
    ## step = 132600: loss = 46036312064.0
    ## step = 132800: loss = 68072456192.0
    ## step = 133000: loss = 111293014016.0
    ## step = 133000: Average Return = 0.0
    ## step = 133200: loss = 87932764160.0
    ## step = 133400: loss = 132721606656.0
    ## step = 133600: loss = 53801381888.0
    ## step = 133800: loss = 23163373568.0
    ## step = 134000: loss = 67594223616.0
    ## step = 134000: Average Return = 0.0
    ## step = 134200: loss = 116428906496.0
    ## step = 134400: loss = 46191747072.0
    ## step = 134600: loss = 46207578112.0
    ## step = 134800: loss = 95537553408.0
    ## step = 135000: loss = 30405371904.0
    ## step = 135000: Average Return = 0.0
    ## step = 135200: loss = 107321827328.0
    ## step = 135400: loss = 26404347904.0
    ## step = 135600: loss = 112338001920.0
    ## step = 135800: loss = 102853632000.0
    ## step = 136000: loss = 28402239488.0
    ## step = 136000: Average Return = 0.0
    ## step = 136200: loss = 81541316608.0
    ## step = 136400: loss = 89039806464.0
    ## step = 136600: loss = 36731084800.0
    ## step = 136800: loss = 78033010688.0
    ## step = 137000: loss = 97377615872.0
    ## step = 137000: Average Return = 0.0
    ## step = 137200: loss = 107265130496.0
    ## step = 137400: loss = 117576040448.0
    ## step = 137600: loss = 58270507008.0
    ## step = 137800: loss = 39985868800.0
    ## step = 138000: loss = 62255104000.0
    ## step = 138000: Average Return = 0.0
    ## step = 138200: loss = 179687391232.0
    ## step = 138400: loss = 44535574528.0
    ## step = 138600: loss = 48819621888.0
    ## step = 138800: loss = 149627502592.0
    ## step = 139000: loss = 139596709888.0
    ## step = 139000: Average Return = 0.0
    ## step = 139200: loss = 23997505536.0
    ## step = 139400: loss = 20791595008.0
    ## step = 139600: loss = 137795190784.0
    ## step = 139800: loss = 64615350272.0
    ## step = 140000: loss = 74097336320.0
    ## step = 140000: Average Return = 0.0
    ## step = 140200: loss = 59275608064.0
    ## step = 140400: loss = 90606190592.0
    ## step = 140600: loss = 183370498048.0
    ## step = 140800: loss = 112836747264.0
    ## step = 141000: loss = 175942647808.0
    ## step = 141000: Average Return = 0.0
    ## step = 141200: loss = 113163182080.0
    ## step = 141400: loss = 29085597696.0
    ## step = 141600: loss = 183668981760.0
    ## step = 141800: loss = 144461692928.0
    ## step = 142000: loss = 186303971328.0
    ## step = 142000: Average Return = 0.0
    ## step = 142200: loss = 123442167808.0
    ## step = 142400: loss = 145365696512.0
    ## step = 142600: loss = 58294075392.0
    ## step = 142800: loss = 174196572160.0
    ## step = 143000: loss = 123722792960.0
    ## step = 143000: Average Return = 0.0
    ## step = 143200: loss = 121341493248.0
    ## step = 143400: loss = 217774686208.0
    ## step = 143600: loss = 28734963712.0
    ## step = 143800: loss = 80755130368.0
    ## step = 144000: loss = 103820804096.0
    ## step = 144000: Average Return = 0.0
    ## step = 144200: loss = 83164979200.0
    ## step = 144400: loss = 101066178560.0
    ## step = 144600: loss = 24440268800.0
    ## step = 144800: loss = 65211297792.0
    ## step = 145000: loss = 87053565952.0
    ## step = 145000: Average Return = 0.0
    ## step = 145200: loss = 48127025152.0
    ## step = 145400: loss = 165720457216.0
    ## step = 145600: loss = 91419918336.0
    ## step = 145800: loss = 95967354880.0
    ## step = 146000: loss = 104865267712.0
    ## step = 146000: Average Return = 0.0
    ## step = 146200: loss = 75742789632.0
    ## step = 146400: loss = 155610759168.0
    ## step = 146600: loss = 146914312192.0
    ## step = 146800: loss = 54648549376.0
    ## step = 147000: loss = 152374558720.0
    ## step = 147000: Average Return = 0.0
    ## step = 147200: loss = 42764423168.0
    ## step = 147400: loss = 92326158336.0
    ## step = 147600: loss = 50763132928.0
    ## step = 147800: loss = 187772796928.0
    ## step = 148000: loss = 127679578112.0
    ## step = 148000: Average Return = 0.0
    ## step = 148200: loss = 85346009088.0
    ## step = 148400: loss = 19293042688.0
    ## step = 148600: loss = 57298092032.0
    ## step = 148800: loss = 108793176064.0
    ## step = 149000: loss = 120393121792.0
    ## step = 149000: Average Return = 0.0
    ## step = 149200: loss = 86627827712.0
    ## step = 149400: loss = 130450055168.0
    ## step = 149600: loss = 107017043968.0
    ## step = 149800: loss = 3421394432.0
    ## step = 150000: loss = 85525217280.0
    ## step = 150000: Average Return = 0.0
    ## step = 150200: loss = 61702705152.0
    ## step = 150400: loss = 74160586752.0
    ## step = 150600: loss = 73997508608.0
    ## step = 150800: loss = 55675125760.0
    ## step = 151000: loss = 41438859264.0
    ## step = 151000: Average Return = 0.0
    ## step = 151200: loss = 156446621696.0
    ## step = 151400: loss = 96200507392.0
    ## step = 151600: loss = 86272974848.0
    ## step = 151800: loss = 149067497472.0
    ## step = 152000: loss = 112469688320.0
    ## step = 152000: Average Return = 0.0
    ## step = 152200: loss = 136328232960.0
    ## step = 152400: loss = 121046712320.0
    ## step = 152600: loss = 120416149504.0
    ## step = 152800: loss = 102737707008.0
    ## step = 153000: loss = 176851943424.0
    ## step = 153000: Average Return = 0.0
    ## step = 153200: loss = 96076505088.0
    ## step = 153400: loss = 182996811776.0
    ## step = 153600: loss = 122856251392.0
    ## step = 153800: loss = 117154152448.0
    ## step = 154000: loss = 75080933376.0
    ## step = 154000: Average Return = 0.0
    ## step = 154200: loss = 78499414016.0
    ## step = 154400: loss = 236723486720.0
    ## step = 154600: loss = 160212844544.0
    ## step = 154800: loss = 129533239296.0
    ## step = 155000: loss = 124540690432.0
    ## step = 155000: Average Return = 0.0
    ## step = 155200: loss = 83954589696.0
    ## step = 155400: loss = 99559530496.0
    ## step = 155600: loss = 151890395136.0
    ## step = 155800: loss = 163642982400.0
    ## step = 156000: loss = 184212668416.0
    ## step = 156000: Average Return = 0.0
    ## step = 156200: loss = 161026260992.0
    ## step = 156400: loss = 80402972672.0
    ## step = 156600: loss = 35419783168.0
    ## step = 156800: loss = 25567248384.0
    ## step = 157000: loss = 50951917568.0
    ## step = 157000: Average Return = 0.0
    ## step = 157200: loss = 219628879872.0
    ## step = 157400: loss = 4322798080.0
    ## step = 157600: loss = 120248352768.0
    ## step = 157800: loss = 41410166784.0
    ## step = 158000: loss = 179662422016.0
    ## step = 158000: Average Return = 0.0
    ## step = 158200: loss = 114324373504.0
    ## step = 158400: loss = 25686753280.0
    ## step = 158600: loss = 145954045952.0
    ## step = 158800: loss = 163440459776.0
    ## step = 159000: loss = 156922888192.0
    ## step = 159000: Average Return = 0.0
    ## step = 159200: loss = 207857860608.0
    ## step = 159400: loss = 78071218176.0
    ## step = 159600: loss = 95733694464.0
    ## step = 159800: loss = 262308724736.0
    ## step = 160000: loss = 103105798144.0
    ## step = 160000: Average Return = 0.0
    ## step = 160200: loss = 79216214016.0
    ## step = 160400: loss = 233903685632.0
    ## step = 160600: loss = 171604475904.0
    ## step = 160800: loss = 306609192960.0
    ## step = 161000: loss = 245299380224.0
    ## step = 161000: Average Return = 0.0
    ## step = 161200: loss = 297703374848.0
    ## step = 161400: loss = 233515352064.0
    ## step = 161600: loss = 98342174720.0
    ## step = 161800: loss = 4529484288.0
    ## step = 162000: loss = 249209765888.0
    ## step = 162000: Average Return = 0.0
    ## step = 162200: loss = 2151915008.0
    ## step = 162400: loss = 65762861056.0
    ## step = 162600: loss = 76095717376.0
    ## step = 162800: loss = 263293730816.0
    ## step = 163000: loss = 209312874496.0
    ## step = 163000: Average Return = 0.0
    ## step = 163200: loss = 97780482048.0
    ## step = 163400: loss = 104065228800.0
    ## step = 163600: loss = 162056814592.0
    ## step = 163800: loss = 37761081344.0
    ## step = 164000: loss = 5741379584.0
    ## step = 164000: Average Return = 0.0
    ## step = 164200: loss = 189178118144.0
    ## step = 164400: loss = 175746514944.0
    ## step = 164600: loss = 170128539648.0
    ## step = 164800: loss = 233802645504.0
    ## step = 165000: loss = 136243019776.0
    ## step = 165000: Average Return = 0.0
    ## step = 165200: loss = 60755124224.0
    ## step = 165400: loss = 188482650112.0
    ## step = 165600: loss = 348828991488.0
    ## step = 165800: loss = 121202311168.0
    ## step = 166000: loss = 53432672256.0
    ## step = 166000: Average Return = 0.0
    ## step = 166200: loss = 30477164544.0
    ## step = 166400: loss = 136058978304.0
    ## step = 166600: loss = 119080591360.0
    ## step = 166800: loss = 237376176128.0
    ## step = 167000: loss = 243507068928.0
    ## step = 167000: Average Return = 0.0
    ## step = 167200: loss = 244255555584.0
    ## step = 167400: loss = 104879243264.0
    ## step = 167600: loss = 60092760064.0
    ## step = 167800: loss = 73507160064.0
    ## step = 168000: loss = 138869325824.0
    ## step = 168000: Average Return = 0.0
    ## step = 168200: loss = 176575479808.0
    ## step = 168400: loss = 56564875264.0
    ## step = 168600: loss = 174636466176.0
    ## step = 168800: loss = 218364903424.0
    ## step = 169000: loss = 101827985408.0
    ## step = 169000: Average Return = 0.0
    ## step = 169200: loss = 113516904448.0
    ## step = 169400: loss = 292279255040.0
    ## step = 169600: loss = 345523552256.0
    ## step = 169800: loss = 125005684736.0
    ## step = 170000: loss = 60236800000.0
    ## step = 170000: Average Return = 0.0
    ## step = 170200: loss = 263745896448.0
    ## step = 170400: loss = 241339465728.0
    ## step = 170600: loss = 157411868672.0
    ## step = 170800: loss = 55451607040.0
    ## step = 171000: loss = 195207315456.0
    ## step = 171000: Average Return = 0.0
    ## step = 171200: loss = 123163934720.0
    ## step = 171400: loss = 253994450944.0
    ## step = 171600: loss = 171011768320.0
    ## step = 171800: loss = 244193132544.0
    ## step = 172000: loss = 272449470464.0
    ## step = 172000: Average Return = 0.0
    ## step = 172200: loss = 153132515328.0
    ## step = 172400: loss = 173038649344.0
    ## step = 172600: loss = 155987083264.0
    ## step = 172800: loss = 205725368320.0
    ## step = 173000: loss = 56170283008.0
    ## step = 173000: Average Return = 0.0
    ## step = 173200: loss = 128308035584.0
    ## step = 173400: loss = 92745121792.0
    ## step = 173600: loss = 318757961728.0
    ## step = 173800: loss = 88796995584.0
    ## step = 174000: loss = 142964244480.0
    ## step = 174000: Average Return = 0.0
    ## step = 174200: loss = 80014303232.0
    ## step = 174400: loss = 201031745536.0
    ## step = 174600: loss = 162222915584.0
    ## step = 174800: loss = 153278218240.0
    ## step = 175000: loss = 241454858240.0
    ## step = 175000: Average Return = 0.0
    ## step = 175200: loss = 363395416064.0
    ## step = 175400: loss = 463193899008.0
    ## step = 175600: loss = 65365598208.0
    ## step = 175800: loss = 323914956800.0
    ## step = 176000: loss = 184773738496.0
    ## step = 176000: Average Return = 0.0
    ## step = 176200: loss = 195855548416.0
    ## step = 176400: loss = 33057677312.0
    ## step = 176600: loss = 86295781376.0
    ## step = 176800: loss = 171079712768.0
    ## step = 177000: loss = 95375024128.0
    ## step = 177000: Average Return = 0.0
    ## step = 177200: loss = 234676125696.0
    ## step = 177400: loss = 433007919104.0
    ## step = 177600: loss = 301966163968.0
    ## step = 177800: loss = 171281907712.0
    ## step = 178000: loss = 145475993600.0
    ## step = 178000: Average Return = 0.0
    ## step = 178200: loss = 173425000448.0
    ## step = 178400: loss = 163538944000.0
    ## step = 178600: loss = 340916338688.0
    ## step = 178800: loss = 232344141824.0
    ## step = 179000: loss = 289545355264.0
    ## step = 179000: Average Return = 0.0
    ## step = 179200: loss = 290372255744.0
    ## step = 179400: loss = 151549706240.0
    ## step = 179600: loss = 379193131008.0
    ## step = 179800: loss = 379850391552.0
    ## step = 180000: loss = 132914618368.0
    ## step = 180000: Average Return = 0.0
    ## step = 180200: loss = 154949632000.0
    ## step = 180400: loss = 224180453376.0
    ## step = 180600: loss = 274802671616.0
    ## step = 180800: loss = 341134082048.0
    ## step = 181000: loss = 130333294592.0
    ## step = 181000: Average Return = 0.0
    ## step = 181200: loss = 175025012736.0
    ## step = 181400: loss = 277670002688.0
    ## step = 181600: loss = 329518284800.0
    ## step = 181800: loss = 386939289600.0
    ## step = 182000: loss = 295930331136.0
    ## step = 182000: Average Return = 0.0
    ## step = 182200: loss = 196463181824.0
    ## step = 182400: loss = 63523618816.0
    ## step = 182600: loss = 394000859136.0
    ## step = 182800: loss = 317409984512.0
    ## step = 183000: loss = 335165620224.0
    ## step = 183000: Average Return = 0.0
    ## step = 183200: loss = 247348461568.0
    ## step = 183400: loss = 373831827456.0
    ## step = 183600: loss = 145633509376.0
    ## step = 183800: loss = 91710627840.0
    ## step = 184000: loss = 247461314560.0
    ## step = 184000: Average Return = 0.0
    ## step = 184200: loss = 181980921856.0
    ## step = 184400: loss = 253215997952.0
    ## step = 184600: loss = 158169923584.0
    ## step = 184800: loss = 72048238592.0
    ## step = 185000: loss = 107393368064.0
    ## step = 185000: Average Return = 0.0
    ## step = 185200: loss = 556987645952.0
    ## step = 185400: loss = 471315906560.0
    ## step = 185600: loss = 328351154176.0
    ## step = 185800: loss = 576276856832.0
    ## step = 186000: loss = 100506746880.0
    ## step = 186000: Average Return = 0.0
    ## step = 186200: loss = 194959474688.0
    ## step = 186400: loss = 199382630400.0
    ## step = 186600: loss = 196453335040.0
    ## step = 186800: loss = 739883941888.0
    ## step = 187000: loss = 276473905152.0
    ## step = 187000: Average Return = 0.0
    ## step = 187200: loss = 173312933888.0
    ## step = 187400: loss = 276716191744.0
    ## step = 187600: loss = 392697774080.0
    ## step = 187800: loss = 44323934208.0
    ## step = 188000: loss = 231313473536.0
    ## step = 188000: Average Return = 0.0
    ## step = 188200: loss = 312113233920.0
    ## step = 188400: loss = 170333028352.0
    ## step = 188600: loss = 293907169280.0
    ## step = 188800: loss = 409247547392.0
    ## step = 189000: loss = 240589438976.0
    ## step = 189000: Average Return = 0.0
    ## step = 189200: loss = 86526697472.0
    ## step = 189400: loss = 104296882176.0
    ## step = 189600: loss = 270764720128.0
    ## step = 189800: loss = 331764563968.0
    ## step = 190000: loss = 195637510144.0
    ## step = 190000: Average Return = 0.0
    ## step = 190200: loss = 265690071040.0
    ## step = 190400: loss = 270256144384.0
    ## step = 190600: loss = 246055354368.0
    ## step = 190800: loss = 233684549632.0
    ## step = 191000: loss = 303019851776.0
    ## step = 191000: Average Return = 0.0
    ## step = 191200: loss = 86058819584.0
    ## step = 191400: loss = 289665515520.0
    ## step = 191600: loss = 165433065472.0
    ## step = 191800: loss = 465793613824.0
    ## step = 192000: loss = 279086596096.0
    ## step = 192000: Average Return = 0.0
    ## step = 192200: loss = 175195258880.0
    ## step = 192400: loss = 246966190080.0
    ## step = 192600: loss = 231328743424.0
    ## step = 192800: loss = 614576422912.0
    ## step = 193000: loss = 156424060928.0
    ## step = 193000: Average Return = 0.0
    ## step = 193200: loss = 314989543424.0
    ## step = 193400: loss = 45410541568.0
    ## step = 193600: loss = 127431196672.0
    ## step = 193800: loss = 280501616640.0
    ## step = 194000: loss = 157283254272.0
    ## step = 194000: Average Return = 0.0
    ## step = 194200: loss = 791207477248.0
    ## step = 194400: loss = 191595479040.0
    ## step = 194600: loss = 220859482112.0
    ## step = 194800: loss = 56777093120.0
    ## step = 195000: loss = 333756104704.0
    ## step = 195000: Average Return = 0.0
    ## step = 195200: loss = 189077176320.0
    ## step = 195400: loss = 462773026816.0
    ## step = 195600: loss = 230204702720.0
    ## step = 195800: loss = 316610543616.0
    ## step = 196000: loss = 345965199360.0
    ## step = 196000: Average Return = 0.0
    ## step = 196200: loss = 265965731840.0
    ## step = 196400: loss = 149379465216.0
    ## step = 196600: loss = 117393760256.0
    ## step = 196800: loss = 427227873280.0
    ## step = 197000: loss = 402703810560.0
    ## step = 197000: Average Return = 0.0
    ## step = 197200: loss = 109493223424.0
    ## step = 197400: loss = 293100552192.0
    ## step = 197600: loss = 261347049472.0
    ## step = 197800: loss = 328722317312.0
    ## step = 198000: loss = 374787899392.0
    ## step = 198000: Average Return = 0.0
    ## step = 198200: loss = 289228455936.0
    ## step = 198400: loss = 18724526080.0
    ## step = 198600: loss = 201227141120.0
    ## step = 198800: loss = 113106485248.0
    ## step = 199000: loss = 368199729152.0
    ## step = 199000: Average Return = 0.0
    ## step = 199200: loss = 213639380992.0
    ## step = 199400: loss = 439899979776.0
    ## step = 199600: loss = 245144485888.0
    ## step = 199800: loss = 165146886144.0
    ## step = 200000: loss = 310037610496.0
    ## step = 200000: Average Return = 0.0

## Visualization

### Plots

Use `matplotlib.pyplot` to chart how the policy improved during
training.

One iteration of `fishing-v2` consists of 1000 time steps. MSY harvest
is `rK/4` = `0.1 * 1 / 4`, so the maximum return for one episode is 25.

``` python
import matplotlib
import matplotlib.pyplot as plt
#@test {"skip": true}

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
```

    ## [<matplotlib.lines.Line2D object at 0x7fd5f450a0b8>]

``` python
plt.ylabel('Average Return')
```

    ## Text(0, 0.5, 'Average Return')

``` python
plt.xlabel('Iterations')
```

    ## Text(0.5, 0, 'Iterations')

``` python
plt.ylim(top=25)
```

    ## (-0.037500000000000006, 25.0)

``` python
plt.show()
```

<img src="dqn_fishing_files/figure-gfm/unnamed-chunk-24-1.png" width="672" />

``` python
plt.savefig('foo.png', bbox_inches='tight')
```
