
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

    ## TimeStep(step_type=array(1, dtype=int32), reward=array(0.01, dtype=float32), discount=array(1., dtype=float32), observation=array([0.68116979]))

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

    ## PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int64, numpy=array([95])>, state=(), info=())

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

    ## 0.7793948

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

    ## <tensorflow.python.data.ops.iterator_ops.OwnedIterator object at 0x7f358c192c88>

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

    ## step = 200: loss = 6.004058837890625
    ## step = 400: loss = 63.32980728149414
    ## step = 600: loss = 486.31219482421875
    ## step = 800: loss = 1392.09765625
    ## step = 1000: loss = 2792.25830078125
    ## step = 1000: Average Return = 0.0
    ## step = 1200: loss = 13056.001953125
    ## step = 1400: loss = 9468.7197265625
    ## step = 1600: loss = 31650.701171875
    ## step = 1800: loss = 49426.54296875
    ## step = 2000: loss = 78091.515625
    ## step = 2000: Average Return = 0.0
    ## step = 2200: loss = 51171.203125
    ## step = 2400: loss = 95697.96875
    ## step = 2600: loss = 174848.3125
    ## step = 2800: loss = 233627.609375
    ## step = 3000: loss = 241551.53125
    ## step = 3000: Average Return = 0.0
    ## step = 3200: loss = 244041.65625
    ## step = 3400: loss = 259416.28125
    ## step = 3600: loss = 541762.4375
    ## step = 3800: loss = 708294.6875
    ## step = 4000: loss = 263286.84375
    ## step = 4000: Average Return = 0.0
    ## step = 4200: loss = 1718019.5
    ## step = 4400: loss = 383976.78125
    ## step = 4600: loss = 1438854.5
    ## step = 4800: loss = 1407011.75
    ## step = 5000: loss = 1319210.75
    ## step = 5000: Average Return = 0.0
    ## step = 5200: loss = 1689381.0
    ## step = 5400: loss = 2074166.125
    ## step = 5600: loss = 1778950.625
    ## step = 5800: loss = 2608825.75
    ## step = 6000: loss = 745896.5
    ## step = 6000: Average Return = 0.0
    ## step = 6200: loss = 1552740.875
    ## step = 6400: loss = 2233800.5
    ## step = 6600: loss = 1893755.875
    ## step = 6800: loss = 4384223.5
    ## step = 7000: loss = 2809323.5
    ## step = 7000: Average Return = 0.0
    ## step = 7200: loss = 663989.8125
    ## step = 7400: loss = 2016350.125
    ## step = 7600: loss = 3089294.5
    ## step = 7800: loss = 4591621.5
    ## step = 8000: loss = 4812686.5
    ## step = 8000: Average Return = 0.0
    ## step = 8200: loss = 6435564.0
    ## step = 8400: loss = 6317555.0
    ## step = 8600: loss = 7270389.5
    ## step = 8800: loss = 5685942.5
    ## step = 9000: loss = 2267163.75
    ## step = 9000: Average Return = 0.0
    ## step = 9200: loss = 1805244.5
    ## step = 9400: loss = 13426098.0
    ## step = 9600: loss = 5924361.0
    ## step = 9800: loss = 18993702.0
    ## step = 10000: loss = 9868625.0
    ## step = 10000: Average Return = 0.0
    ## step = 10200: loss = 12391206.0
    ## step = 10400: loss = 12923254.0
    ## step = 10600: loss = 9321364.0
    ## step = 10800: loss = 2333218.75
    ## step = 11000: loss = 17672796.0
    ## step = 11000: Average Return = 0.0
    ## step = 11200: loss = 23752684.0
    ## step = 11400: loss = 12711278.0
    ## step = 11600: loss = 13949814.0
    ## step = 11800: loss = 19974312.0
    ## step = 12000: loss = 4456232.0
    ## step = 12000: Average Return = 0.0
    ## step = 12200: loss = 27322358.0
    ## step = 12400: loss = 24419160.0
    ## step = 12600: loss = 20798290.0
    ## step = 12800: loss = 21716214.0
    ## step = 13000: loss = 10507326.0
    ## step = 13000: Average Return = 0.0
    ## step = 13200: loss = 16519462.0
    ## step = 13400: loss = 35679560.0
    ## step = 13600: loss = 38072676.0
    ## step = 13800: loss = 30688852.0
    ## step = 14000: loss = 19188636.0
    ## step = 14000: Average Return = 0.0
    ## step = 14200: loss = 30804442.0
    ## step = 14400: loss = 14288790.0
    ## step = 14600: loss = 40224084.0
    ## step = 14800: loss = 47277928.0
    ## step = 15000: loss = 59142992.0
    ## step = 15000: Average Return = 0.0
    ## step = 15200: loss = 35901104.0
    ## step = 15400: loss = 70790296.0
    ## step = 15600: loss = 38225784.0
    ## step = 15800: loss = 28218468.0
    ## step = 16000: loss = 60380792.0
    ## step = 16000: Average Return = 0.0
    ## step = 16200: loss = 68787448.0
    ## step = 16400: loss = 13944332.0
    ## step = 16600: loss = 55985952.0
    ## step = 16800: loss = 39351920.0
    ## step = 17000: loss = 63553916.0
    ## step = 17000: Average Return = 0.0
    ## step = 17200: loss = 33329206.0
    ## step = 17400: loss = 76379720.0
    ## step = 17600: loss = 58350312.0
    ## step = 17800: loss = 10646870.0
    ## step = 18000: loss = 92236544.0
    ## step = 18000: Average Return = 0.0
    ## step = 18200: loss = 110307712.0
    ## step = 18400: loss = 73693616.0
    ## step = 18600: loss = 2612823.5
    ## step = 18800: loss = 23233422.0
    ## step = 19000: loss = 58072392.0
    ## step = 19000: Average Return = 0.0
    ## step = 19200: loss = 104692640.0
    ## step = 19400: loss = 81112912.0
    ## step = 19600: loss = 76938272.0
    ## step = 19800: loss = 81246824.0
    ## step = 20000: loss = 51553348.0
    ## step = 20000: Average Return = 0.0
    ## step = 20200: loss = 39530680.0
    ## step = 20400: loss = 115134824.0
    ## step = 20600: loss = 40880296.0
    ## step = 20800: loss = 29390630.0
    ## step = 21000: loss = 107952792.0
    ## step = 21000: Average Return = 0.0
    ## step = 21200: loss = 63033200.0
    ## step = 21400: loss = 93307440.0
    ## step = 21600: loss = 151428944.0
    ## step = 21800: loss = 31260918.0
    ## step = 22000: loss = 56273736.0
    ## step = 22000: Average Return = 0.0
    ## step = 22200: loss = 81145120.0
    ## step = 22400: loss = 210023072.0
    ## step = 22600: loss = 200559408.0
    ## step = 22800: loss = 28196058.0
    ## step = 23000: loss = 103734680.0
    ## step = 23000: Average Return = 0.0
    ## step = 23200: loss = 57984084.0
    ## step = 23400: loss = 91688152.0
    ## step = 23600: loss = 146167136.0
    ## step = 23800: loss = 183723584.0
    ## step = 24000: loss = 114752624.0
    ## step = 24000: Average Return = 0.0
    ## step = 24200: loss = 127927208.0
    ## step = 24400: loss = 62123024.0
    ## step = 24600: loss = 367040032.0
    ## step = 24800: loss = 169795520.0
    ## step = 25000: loss = 280124288.0
    ## step = 25000: Average Return = 0.0
    ## step = 25200: loss = 143882320.0
    ## step = 25400: loss = 299642304.0
    ## step = 25600: loss = 181319600.0
    ## step = 25800: loss = 198949776.0
    ## step = 26000: loss = 99698272.0
    ## step = 26000: Average Return = 0.0
    ## step = 26200: loss = 379179008.0
    ## step = 26400: loss = 267382560.0
    ## step = 26600: loss = 142731104.0
    ## step = 26800: loss = 106374096.0
    ## step = 27000: loss = 472962560.0
    ## step = 27000: Average Return = 0.0
    ## step = 27200: loss = 111770456.0
    ## step = 27400: loss = 251045552.0
    ## step = 27600: loss = 406932160.0
    ## step = 27800: loss = 152092896.0
    ## step = 28000: loss = 436708992.0
    ## step = 28000: Average Return = 0.0
    ## step = 28200: loss = 314610624.0
    ## step = 28400: loss = 491700032.0
    ## step = 28600: loss = 195134736.0
    ## step = 28800: loss = 391173408.0
    ## step = 29000: loss = 132788912.0
    ## step = 29000: Average Return = 0.0
    ## step = 29200: loss = 662039360.0
    ## step = 29400: loss = 356824608.0
    ## step = 29600: loss = 128481424.0
    ## step = 29800: loss = 399517952.0
    ## step = 30000: loss = 109505304.0
    ## step = 30000: Average Return = 0.0
    ## step = 30200: loss = 286989504.0
    ## step = 30400: loss = 8927697.0
    ## step = 30600: loss = 139586160.0
    ## step = 30800: loss = 418527648.0
    ## step = 31000: loss = 598801792.0
    ## step = 31000: Average Return = 0.0
    ## step = 31200: loss = 93495584.0
    ## step = 31400: loss = 182343200.0
    ## step = 31600: loss = 366677024.0
    ## step = 31800: loss = 559911936.0
    ## step = 32000: loss = 460735552.0
    ## step = 32000: Average Return = 0.0
    ## step = 32200: loss = 748756352.0
    ## step = 32400: loss = 426928416.0
    ## step = 32600: loss = 537372992.0
    ## step = 32800: loss = 285993056.0
    ## step = 33000: loss = 581588352.0
    ## step = 33000: Average Return = 0.0
    ## step = 33200: loss = 477808960.0
    ## step = 33400: loss = 327110912.0
    ## step = 33600: loss = 196003840.0
    ## step = 33800: loss = 389392352.0
    ## step = 34000: loss = 720268288.0
    ## step = 34000: Average Return = 0.0
    ## step = 34200: loss = 488586528.0
    ## step = 34400: loss = 857387008.0
    ## step = 34600: loss = 198460416.0
    ## step = 34800: loss = 493242912.0
    ## step = 35000: loss = 1242438784.0
    ## step = 35000: Average Return = 0.0
    ## step = 35200: loss = 240762240.0
    ## step = 35400: loss = 1117758208.0
    ## step = 35600: loss = 475817792.0
    ## step = 35800: loss = 608035200.0
    ## step = 36000: loss = 303115712.0
    ## step = 36000: Average Return = 0.0
    ## step = 36200: loss = 641606080.0
    ## step = 36400: loss = 1557105664.0
    ## step = 36600: loss = 727103168.0
    ## step = 36800: loss = 1208711296.0
    ## step = 37000: loss = 209439152.0
    ## step = 37000: Average Return = 0.0
    ## step = 37200: loss = 1209117312.0
    ## step = 37400: loss = 603058816.0
    ## step = 37600: loss = 932268928.0
    ## step = 37800: loss = 739938752.0
    ## step = 38000: loss = 980121856.0
    ## step = 38000: Average Return = 0.0
    ## step = 38200: loss = 641711808.0
    ## step = 38400: loss = 448458272.0
    ## step = 38600: loss = 1473852160.0
    ## step = 38800: loss = 1523079168.0
    ## step = 39000: loss = 185834080.0
    ## step = 39000: Average Return = 0.0
    ## step = 39200: loss = 227092064.0
    ## step = 39400: loss = 1219310720.0
    ## step = 39600: loss = 885613760.0
    ## step = 39800: loss = 1323483520.0
    ## step = 40000: loss = 671213952.0
    ## step = 40000: Average Return = 0.0
    ## step = 40200: loss = 934050816.0
    ## step = 40400: loss = 468650240.0
    ## step = 40600: loss = 156077104.0
    ## step = 40800: loss = 983143808.0
    ## step = 41000: loss = 172707424.0
    ## step = 41000: Average Return = 0.0
    ## step = 41200: loss = 1253689600.0
    ## step = 41400: loss = 970762944.0
    ## step = 41600: loss = 1017048768.0
    ## step = 41800: loss = 1185683328.0
    ## step = 42000: loss = 463160608.0
    ## step = 42000: Average Return = 0.0
    ## step = 42200: loss = 1260548096.0
    ## step = 42400: loss = 1423238784.0
    ## step = 42600: loss = 1250397568.0
    ## step = 42800: loss = 1032155008.0
    ## step = 43000: loss = 236114128.0
    ## step = 43000: Average Return = 0.0
    ## step = 43200: loss = 2240583680.0
    ## step = 43400: loss = 1351472640.0
    ## step = 43600: loss = 831684928.0
    ## step = 43800: loss = 505492608.0
    ## step = 44000: loss = 1379219456.0
    ## step = 44000: Average Return = 0.0
    ## step = 44200: loss = 2096915712.0
    ## step = 44400: loss = 1091202560.0
    ## step = 44600: loss = 1513865856.0
    ## step = 44800: loss = 1801782144.0
    ## step = 45000: loss = 1114629376.0
    ## step = 45000: Average Return = 0.0
    ## step = 45200: loss = 179848944.0
    ## step = 45400: loss = 1324599680.0
    ## step = 45600: loss = 1237037312.0
    ## step = 45800: loss = 3401282560.0
    ## step = 46000: loss = 1126417920.0
    ## step = 46000: Average Return = 0.0
    ## step = 46200: loss = 1237741568.0
    ## step = 46400: loss = 814679104.0
    ## step = 46600: loss = 2660635904.0
    ## step = 46800: loss = 2632198912.0
    ## step = 47000: loss = 2292684800.0
    ## step = 47000: Average Return = 0.0
    ## step = 47200: loss = 1976706048.0
    ## step = 47400: loss = 1385483008.0
    ## step = 47600: loss = 1526980608.0
    ## step = 47800: loss = 3449361920.0
    ## step = 48000: loss = 2582724352.0
    ## step = 48000: Average Return = 0.0
    ## step = 48200: loss = 2144857344.0
    ## step = 48400: loss = 1751808000.0
    ## step = 48600: loss = 2046871552.0
    ## step = 48800: loss = 2310003200.0
    ## step = 49000: loss = 2116839424.0
    ## step = 49000: Average Return = 0.0
    ## step = 49200: loss = 1521422848.0
    ## step = 49400: loss = 2026528384.0
    ## step = 49600: loss = 1959329792.0
    ## step = 49800: loss = 3050749184.0
    ## step = 50000: loss = 1868072576.0
    ## step = 50000: Average Return = 0.0
    ## step = 50200: loss = 2148370688.0
    ## step = 50400: loss = 2483182592.0
    ## step = 50600: loss = 668814976.0
    ## step = 50800: loss = 934645120.0
    ## step = 51000: loss = 4447430144.0
    ## step = 51000: Average Return = 0.0
    ## step = 51200: loss = 2305827328.0
    ## step = 51400: loss = 3059207168.0
    ## step = 51600: loss = 4079367168.0
    ## step = 51800: loss = 3009873152.0
    ## step = 52000: loss = 2904854784.0
    ## step = 52000: Average Return = 0.0
    ## step = 52200: loss = 573600832.0
    ## step = 52400: loss = 3378144256.0
    ## step = 52600: loss = 3613702912.0
    ## step = 52800: loss = 3877153792.0
    ## step = 53000: loss = 3455085568.0
    ## step = 53000: Average Return = 0.0
    ## step = 53200: loss = 3611552000.0
    ## step = 53400: loss = 943524736.0
    ## step = 53600: loss = 3926308608.0
    ## step = 53800: loss = 2962593280.0
    ## step = 54000: loss = 2283211264.0
    ## step = 54000: Average Return = 0.0
    ## step = 54200: loss = 4233441280.0
    ## step = 54400: loss = 3124281344.0
    ## step = 54600: loss = 2035464448.0
    ## step = 54800: loss = 1881110016.0
    ## step = 55000: loss = 2704676096.0
    ## step = 55000: Average Return = 0.0
    ## step = 55200: loss = 1565518080.0
    ## step = 55400: loss = 2497509376.0
    ## step = 55600: loss = 2420736000.0
    ## step = 55800: loss = 1554138368.0
    ## step = 56000: loss = 2971306496.0
    ## step = 56000: Average Return = 0.0
    ## step = 56200: loss = 5752718848.0
    ## step = 56400: loss = 2445067776.0
    ## step = 56600: loss = 2364008192.0
    ## step = 56800: loss = 3988699648.0
    ## step = 57000: loss = 2776130560.0
    ## step = 57000: Average Return = 0.0
    ## step = 57200: loss = 3304089600.0
    ## step = 57400: loss = 5755243520.0
    ## step = 57600: loss = 819135552.0
    ## step = 57800: loss = 1402496256.0
    ## step = 58000: loss = 4715043328.0
    ## step = 58000: Average Return = 0.0
    ## step = 58200: loss = 2806916096.0
    ## step = 58400: loss = 3393385984.0
    ## step = 58600: loss = 3010440192.0
    ## step = 58800: loss = 2448553984.0
    ## step = 59000: loss = 1210896896.0
    ## step = 59000: Average Return = 0.0
    ## step = 59200: loss = 3691994112.0
    ## step = 59400: loss = 5034279936.0
    ## step = 59600: loss = 6744753664.0
    ## step = 59800: loss = 3477070080.0
    ## step = 60000: loss = 6840492032.0
    ## step = 60000: Average Return = 0.0
    ## step = 60200: loss = 2478368000.0
    ## step = 60400: loss = 139057888.0
    ## step = 60600: loss = 812913920.0
    ## step = 60800: loss = 2523973632.0
    ## step = 61000: loss = 2500367872.0
    ## step = 61000: Average Return = 0.0
    ## step = 61200: loss = 1293564544.0
    ## step = 61400: loss = 2473095680.0
    ## step = 61600: loss = 3883107584.0
    ## step = 61800: loss = 8678070272.0
    ## step = 62000: loss = 4171427328.0
    ## step = 62000: Average Return = 0.0
    ## step = 62200: loss = 3096714496.0
    ## step = 62400: loss = 4050606848.0
    ## step = 62600: loss = 5684061696.0
    ## step = 62800: loss = 6778661888.0
    ## step = 63000: loss = 3563897344.0
    ## step = 63000: Average Return = 0.0
    ## step = 63200: loss = 7999090176.0
    ## step = 63400: loss = 5430572544.0
    ## step = 63600: loss = 2505304832.0
    ## step = 63800: loss = 4934521856.0
    ## step = 64000: loss = 3208540160.0
    ## step = 64000: Average Return = 0.0
    ## step = 64200: loss = 5169811456.0
    ## step = 64400: loss = 4693942272.0
    ## step = 64600: loss = 5210822656.0
    ## step = 64800: loss = 3327920128.0
    ## step = 65000: loss = 4957561856.0
    ## step = 65000: Average Return = 0.0
    ## step = 65200: loss = 1146233728.0
    ## step = 65400: loss = 3394444800.0
    ## step = 65600: loss = 6101630976.0
    ## step = 65800: loss = 4307191296.0
    ## step = 66000: loss = 5315388416.0
    ## step = 66000: Average Return = 0.0
    ## step = 66200: loss = 3515120896.0
    ## step = 66400: loss = 5504355328.0
    ## step = 66600: loss = 6975621120.0
    ## step = 66800: loss = 3662013952.0
    ## step = 67000: loss = 5145577472.0
    ## step = 67000: Average Return = 0.0
    ## step = 67200: loss = 4535764480.0
    ## step = 67400: loss = 10075088896.0
    ## step = 67600: loss = 3353370112.0
    ## step = 67800: loss = 4696134144.0
    ## step = 68000: loss = 5573437952.0
    ## step = 68000: Average Return = 0.0
    ## step = 68200: loss = 2287329024.0
    ## step = 68400: loss = 9036526592.0
    ## step = 68600: loss = 11563030528.0
    ## step = 68800: loss = 4284041984.0
    ## step = 69000: loss = 6523088384.0
    ## step = 69000: Average Return = 0.0
    ## step = 69200: loss = 2820767232.0
    ## step = 69400: loss = 12567024640.0
    ## step = 69600: loss = 8937991168.0
    ## step = 69800: loss = 6436125696.0
    ## step = 70000: loss = 3960628224.0
    ## step = 70000: Average Return = 0.0
    ## step = 70200: loss = 6924929536.0
    ## step = 70400: loss = 3435011840.0
    ## step = 70600: loss = 5308095488.0
    ## step = 70800: loss = 10748224512.0
    ## step = 71000: loss = 11253222400.0
    ## step = 71000: Average Return = 0.0
    ## step = 71200: loss = 11493587968.0
    ## step = 71400: loss = 8858293248.0
    ## step = 71600: loss = 7957669888.0
    ## step = 71800: loss = 4701466624.0
    ## step = 72000: loss = 8391151616.0
    ## step = 72000: Average Return = 0.0
    ## step = 72200: loss = 10886204416.0
    ## step = 72400: loss = 5166124032.0
    ## step = 72600: loss = 12648646656.0
    ## step = 72800: loss = 11763378176.0
    ## step = 73000: loss = 4005594624.0
    ## step = 73000: Average Return = 0.0
    ## step = 73200: loss = 8674990080.0
    ## step = 73400: loss = 8431045120.0
    ## step = 73600: loss = 7132153344.0
    ## step = 73800: loss = 10381469696.0
    ## step = 74000: loss = 2836098048.0
    ## step = 74000: Average Return = 0.0
    ## step = 74200: loss = 10276550656.0
    ## step = 74400: loss = 5497891840.0
    ## step = 74600: loss = 13710424064.0
    ## step = 74800: loss = 4672469504.0
    ## step = 75000: loss = 10982365184.0
    ## step = 75000: Average Return = 0.0
    ## step = 75200: loss = 12860033024.0
    ## step = 75400: loss = 4306186752.0
    ## step = 75600: loss = 5992761856.0
    ## step = 75800: loss = 11078311936.0
    ## step = 76000: loss = 10167290880.0
    ## step = 76000: Average Return = 0.0
    ## step = 76200: loss = 14758340608.0
    ## step = 76400: loss = 6379677184.0
    ## step = 76600: loss = 6906534400.0
    ## step = 76800: loss = 3333744128.0
    ## step = 77000: loss = 3807286784.0
    ## step = 77000: Average Return = 0.0
    ## step = 77200: loss = 10293604352.0
    ## step = 77400: loss = 12209270784.0
    ## step = 77600: loss = 6985187328.0
    ## step = 77800: loss = 11452712960.0
    ## step = 78000: loss = 17591808000.0
    ## step = 78000: Average Return = 0.0
    ## step = 78200: loss = 16401448960.0
    ## step = 78400: loss = 11531918336.0
    ## step = 78600: loss = 6064710144.0
    ## step = 78800: loss = 3717542656.0
    ## step = 79000: loss = 4506224128.0
    ## step = 79000: Average Return = 0.0
    ## step = 79200: loss = 6117585920.0
    ## step = 79400: loss = 9482250240.0
    ## step = 79600: loss = 9520261120.0
    ## step = 79800: loss = 6608115712.0
    ## step = 80000: loss = 4876337152.0
    ## step = 80000: Average Return = 0.0
    ## step = 80200: loss = 7616079360.0
    ## step = 80400: loss = 5399581696.0
    ## step = 80600: loss = 10433589248.0
    ## step = 80800: loss = 7613854720.0
    ## step = 81000: loss = 15743830016.0
    ## step = 81000: Average Return = 0.0
    ## step = 81200: loss = 11066384384.0
    ## step = 81400: loss = 5151270912.0
    ## step = 81600: loss = 630026048.0
    ## step = 81800: loss = 13769981952.0
    ## step = 82000: loss = 10342252544.0
    ## step = 82000: Average Return = 0.0
    ## step = 82200: loss = 26081662976.0
    ## step = 82400: loss = 20657143808.0
    ## step = 82600: loss = 3325133056.0
    ## step = 82800: loss = 14129317888.0
    ## step = 83000: loss = 11956594688.0
    ## step = 83000: Average Return = 0.0
    ## step = 83200: loss = 18053177344.0
    ## step = 83400: loss = 5564734464.0
    ## step = 83600: loss = 5138516992.0
    ## step = 83800: loss = 7380720128.0
    ## step = 84000: loss = 12045039616.0
    ## step = 84000: Average Return = 0.0
    ## step = 84200: loss = 12172609536.0
    ## step = 84400: loss = 12607772672.0
    ## step = 84600: loss = 6110911488.0
    ## step = 84800: loss = 23789047808.0
    ## step = 85000: loss = 12776253440.0
    ## step = 85000: Average Return = 0.0
    ## step = 85200: loss = 8503729664.0
    ## step = 85400: loss = 15915374592.0
    ## step = 85600: loss = 16825654272.0
    ## step = 85800: loss = 16499722240.0
    ## step = 86000: loss = 18123517952.0
    ## step = 86000: Average Return = 0.0
    ## step = 86200: loss = 14266999808.0
    ## step = 86400: loss = 12811370496.0
    ## step = 86600: loss = 13710317568.0
    ## step = 86800: loss = 9991856128.0
    ## step = 87000: loss = 9337841664.0
    ## step = 87000: Average Return = 0.0
    ## step = 87200: loss = 33902815232.0
    ## step = 87400: loss = 11267389440.0
    ## step = 87600: loss = 12613098496.0
    ## step = 87800: loss = 11517462528.0
    ## step = 88000: loss = 25320968192.0
    ## step = 88000: Average Return = 0.0
    ## step = 88200: loss = 3979222016.0
    ## step = 88400: loss = 24269432832.0
    ## step = 88600: loss = 9934528512.0
    ## step = 88800: loss = 6256761856.0
    ## step = 89000: loss = 12942968832.0
    ## step = 89000: Average Return = 0.0
    ## step = 89200: loss = 12599814144.0
    ## step = 89400: loss = 9761386496.0
    ## step = 89600: loss = 10489716736.0
    ## step = 89800: loss = 24805101568.0
    ## step = 90000: loss = 7167513600.0
    ## step = 90000: Average Return = 0.0
    ## step = 90200: loss = 4978551808.0
    ## step = 90400: loss = 16138322944.0
    ## step = 90600: loss = 15366062080.0
    ## step = 90800: loss = 17260949504.0
    ## step = 91000: loss = 5151558144.0
    ## step = 91000: Average Return = 0.0
    ## step = 91200: loss = 13609530368.0
    ## step = 91400: loss = 26147618816.0
    ## step = 91600: loss = 15670453248.0
    ## step = 91800: loss = 35674697728.0
    ## step = 92000: loss = 17699997696.0
    ## step = 92000: Average Return = 0.0
    ## step = 92200: loss = 8143499776.0
    ## step = 92400: loss = 13178195968.0
    ## step = 92600: loss = 28855511040.0
    ## step = 92800: loss = 15801073664.0
    ## step = 93000: loss = 14297882624.0
    ## step = 93000: Average Return = 0.0
    ## step = 93200: loss = 15791807488.0
    ## step = 93400: loss = 11256544256.0
    ## step = 93600: loss = 28432222208.0
    ## step = 93800: loss = 32657393664.0
    ## step = 94000: loss = 20880553984.0
    ## step = 94000: Average Return = 0.0
    ## step = 94200: loss = 33276465152.0
    ## step = 94400: loss = 25282246656.0
    ## step = 94600: loss = 18742857728.0
    ## step = 94800: loss = 37920559104.0
    ## step = 95000: loss = 30407012352.0
    ## step = 95000: Average Return = 0.0
    ## step = 95200: loss = 24232364032.0
    ## step = 95400: loss = 22852270080.0
    ## step = 95600: loss = 734082880.0
    ## step = 95800: loss = 27626397696.0
    ## step = 96000: loss = 23865935872.0
    ## step = 96000: Average Return = 0.0
    ## step = 96200: loss = 16710455296.0
    ## step = 96400: loss = 26068547584.0
    ## step = 96600: loss = 18579073024.0
    ## step = 96800: loss = 25479159808.0
    ## step = 97000: loss = 20574926848.0
    ## step = 97000: Average Return = 0.0
    ## step = 97200: loss = 4524858368.0
    ## step = 97400: loss = 34206220288.0
    ## step = 97600: loss = 13775183872.0
    ## step = 97800: loss = 27454529536.0
    ## step = 98000: loss = 38165733376.0
    ## step = 98000: Average Return = 0.0
    ## step = 98200: loss = 44604399616.0
    ## step = 98400: loss = 10519021568.0
    ## step = 98600: loss = 9639557120.0
    ## step = 98800: loss = 22973243392.0
    ## step = 99000: loss = 21628841984.0
    ## step = 99000: Average Return = 0.0
    ## step = 99200: loss = 41047506944.0
    ## step = 99400: loss = 37143052288.0
    ## step = 99600: loss = 31820435456.0
    ## step = 99800: loss = 25458302976.0
    ## step = 100000: loss = 28666437632.0
    ## step = 100000: Average Return = 0.0
    ## step = 100200: loss = 27625857024.0
    ## step = 100400: loss = 23093477376.0
    ## step = 100600: loss = 44570886144.0
    ## step = 100800: loss = 18297647104.0
    ## step = 101000: loss = 21785706496.0
    ## step = 101000: Average Return = 0.0
    ## step = 101200: loss = 20426649600.0
    ## step = 101400: loss = 40586432512.0
    ## step = 101600: loss = 45142077440.0
    ## step = 101800: loss = 33149276160.0
    ## step = 102000: loss = 18668328960.0
    ## step = 102000: Average Return = 0.0
    ## step = 102200: loss = 22131585024.0
    ## step = 102400: loss = 31772971008.0
    ## step = 102600: loss = 24936589312.0
    ## step = 102800: loss = 4508985344.0
    ## step = 103000: loss = 50711912448.0
    ## step = 103000: Average Return = 0.0
    ## step = 103200: loss = 19557801984.0
    ## step = 103400: loss = 16058808320.0
    ## step = 103600: loss = 22033723392.0
    ## step = 103800: loss = 26097053696.0
    ## step = 104000: loss = 11202870272.0
    ## step = 104000: Average Return = 0.0
    ## step = 104200: loss = 31728381952.0
    ## step = 104400: loss = 21713694720.0
    ## step = 104600: loss = 39423938560.0
    ## step = 104800: loss = 33551441920.0
    ## step = 105000: loss = 8632909824.0
    ## step = 105000: Average Return = 0.0
    ## step = 105200: loss = 34888417280.0
    ## step = 105400: loss = 41649500160.0
    ## step = 105600: loss = 25756995584.0
    ## step = 105800: loss = 39876919296.0
    ## step = 106000: loss = 29651906560.0
    ## step = 106000: Average Return = 0.0
    ## step = 106200: loss = 32226324480.0
    ## step = 106400: loss = 18073870336.0
    ## step = 106600: loss = 37371760640.0
    ## step = 106800: loss = 59880046592.0
    ## step = 107000: loss = 9315885056.0
    ## step = 107000: Average Return = 0.0
    ## step = 107200: loss = 60524888064.0
    ## step = 107400: loss = 52898906112.0
    ## step = 107600: loss = 29990662144.0
    ## step = 107800: loss = 47948058624.0
    ## step = 108000: loss = 39420141568.0
    ## step = 108000: Average Return = 0.0
    ## step = 108200: loss = 17013323776.0
    ## step = 108400: loss = 22552958976.0
    ## step = 108600: loss = 4959190016.0
    ## step = 108800: loss = 1674969984.0
    ## step = 109000: loss = 29615005696.0
    ## step = 109000: Average Return = 0.0
    ## step = 109200: loss = 35792715776.0
    ## step = 109400: loss = 19887331328.0
    ## step = 109600: loss = 50933260288.0
    ## step = 109800: loss = 44288352256.0
    ## step = 110000: loss = 51758358528.0
    ## step = 110000: Average Return = 0.0
    ## step = 110200: loss = 42607898624.0
    ## step = 110400: loss = 38333054976.0
    ## step = 110600: loss = 13718403072.0
    ## step = 110800: loss = 17792757760.0
    ## step = 111000: loss = 48196104192.0
    ## step = 111000: Average Return = 0.0
    ## step = 111200: loss = 1288504320.0
    ## step = 111400: loss = 31129022464.0
    ## step = 111600: loss = 60600291328.0
    ## step = 111800: loss = 24987080704.0
    ## step = 112000: loss = 21514203136.0
    ## step = 112000: Average Return = 0.0
    ## step = 112200: loss = 35761508352.0
    ## step = 112400: loss = 7342718976.0
    ## step = 112600: loss = 22901571584.0
    ## step = 112800: loss = 77759184896.0
    ## step = 113000: loss = 27752757248.0
    ## step = 113000: Average Return = 0.0
    ## step = 113200: loss = 15666184192.0
    ## step = 113400: loss = 26395705344.0
    ## step = 113600: loss = 78944231424.0
    ## step = 113800: loss = 21239214080.0
    ## step = 114000: loss = 18761809920.0
    ## step = 114000: Average Return = 0.0
    ## step = 114200: loss = 8866086912.0
    ## step = 114400: loss = 68177575936.0
    ## step = 114600: loss = 34091356160.0
    ## step = 114800: loss = 23648935936.0
    ## step = 115000: loss = 26158608384.0
    ## step = 115000: Average Return = 0.0
    ## step = 115200: loss = 54918705152.0
    ## step = 115400: loss = 32619378688.0
    ## step = 115600: loss = 78044315648.0
    ## step = 115800: loss = 65140363264.0
    ## step = 116000: loss = 48381841408.0
    ## step = 116000: Average Return = 0.0
    ## step = 116200: loss = 67014443008.0
    ## step = 116400: loss = 55188529152.0
    ## step = 116600: loss = 89105104896.0
    ## step = 116800: loss = 52385886208.0
    ## step = 117000: loss = 36982423552.0
    ## step = 117000: Average Return = 0.0
    ## step = 117200: loss = 67468914688.0
    ## step = 117400: loss = 30954170368.0
    ## step = 117600: loss = 38530904064.0
    ## step = 117800: loss = 41544208384.0
    ## step = 118000: loss = 53988335616.0
    ## step = 118000: Average Return = 0.0
    ## step = 118200: loss = 41742573568.0
    ## step = 118400: loss = 66925625344.0
    ## step = 118600: loss = 45004750848.0
    ## step = 118800: loss = 92610510848.0
    ## step = 119000: loss = 70215294976.0
    ## step = 119000: Average Return = 0.0
    ## step = 119200: loss = 41617301504.0
    ## step = 119400: loss = 37392375808.0
    ## step = 119600: loss = 21285765120.0
    ## step = 119800: loss = 24109111296.0
    ## step = 120000: loss = 57848201216.0
    ## step = 120000: Average Return = 0.0
    ## step = 120200: loss = 45202300928.0
    ## step = 120400: loss = 20493912064.0
    ## step = 120600: loss = 42593804288.0
    ## step = 120800: loss = 22351212544.0
    ## step = 121000: loss = 22504859648.0
    ## step = 121000: Average Return = 0.0
    ## step = 121200: loss = 66826379264.0
    ## step = 121400: loss = 62422016000.0
    ## step = 121600: loss = 52824068096.0
    ## step = 121800: loss = 60096192512.0
    ## step = 122000: loss = 36907360256.0
    ## step = 122000: Average Return = 0.0
    ## step = 122200: loss = 2253314560.0
    ## step = 122400: loss = 15340802048.0
    ## step = 122600: loss = 37796794368.0
    ## step = 122800: loss = 43241721856.0
    ## step = 123000: loss = 23378530304.0
    ## step = 123000: Average Return = 0.0
    ## step = 123200: loss = 24083912704.0
    ## step = 123400: loss = 83771342848.0
    ## step = 123600: loss = 10189197312.0
    ## step = 123800: loss = 32671705088.0
    ## step = 124000: loss = 64617299968.0
    ## step = 124000: Average Return = 0.0
    ## step = 124200: loss = 37958451200.0
    ## step = 124400: loss = 20561479680.0
    ## step = 124600: loss = 34453803008.0
    ## step = 124800: loss = 65774837760.0
    ## step = 125000: loss = 87565139968.0
    ## step = 125000: Average Return = 0.0
    ## step = 125200: loss = 48202969088.0
    ## step = 125400: loss = 78649327616.0
    ## step = 125600: loss = 45415120896.0
    ## step = 125800: loss = 74011803648.0
    ## step = 126000: loss = 10465900544.0
    ## step = 126000: Average Return = 0.0
    ## step = 126200: loss = 47544156160.0
    ## step = 126400: loss = 52043980800.0
    ## step = 126600: loss = 21856243712.0
    ## step = 126800: loss = 66588188672.0
    ## step = 127000: loss = 66527649792.0
    ## step = 127000: Average Return = 0.0
    ## step = 127200: loss = 52294258688.0
    ## step = 127400: loss = 32925820928.0
    ## step = 127600: loss = 28651233280.0
    ## step = 127800: loss = 67299815424.0
    ## step = 128000: loss = 41961086976.0
    ## step = 128000: Average Return = 0.0
    ## step = 128200: loss = 101584994304.0
    ## step = 128400: loss = 40062349312.0
    ## step = 128600: loss = 60031610880.0
    ## step = 128800: loss = 83366551552.0
    ## step = 129000: loss = 96974839808.0
    ## step = 129000: Average Return = 0.0
    ## step = 129200: loss = 143346434048.0
    ## step = 129400: loss = 25138749440.0
    ## step = 129600: loss = 64599891968.0
    ## step = 129800: loss = 50658246656.0
    ## step = 130000: loss = 37039177728.0
    ## step = 130000: Average Return = 0.0
    ## step = 130200: loss = 89616826368.0
    ## step = 130400: loss = 72572157952.0
    ## step = 130600: loss = 13773340672.0
    ## step = 130800: loss = 136982077440.0
    ## step = 131000: loss = 108028698624.0
    ## step = 131000: Average Return = 0.0
    ## step = 131200: loss = 86120939520.0
    ## step = 131400: loss = 76858925056.0
    ## step = 131600: loss = 74664607744.0
    ## step = 131800: loss = 60085608448.0
    ## step = 132000: loss = 3195093504.0
    ## step = 132000: Average Return = 0.0
    ## step = 132200: loss = 48968458240.0
    ## step = 132400: loss = 18303082496.0
    ## step = 132600: loss = 67955908608.0
    ## step = 132800: loss = 57706606592.0
    ## step = 133000: loss = 89597181952.0
    ## step = 133000: Average Return = 0.0
    ## step = 133200: loss = 62284038144.0
    ## step = 133400: loss = 13645200384.0
    ## step = 133600: loss = 78805458944.0
    ## step = 133800: loss = 82630590464.0
    ## step = 134000: loss = 88328699904.0
    ## step = 134000: Average Return = 0.0
    ## step = 134200: loss = 80338640896.0
    ## step = 134400: loss = 66040659968.0
    ## step = 134600: loss = 64153952256.0
    ## step = 134800: loss = 72595472384.0
    ## step = 135000: loss = 104609775616.0
    ## step = 135000: Average Return = 0.0
    ## step = 135200: loss = 91355029504.0
    ## step = 135400: loss = 61988335616.0
    ## step = 135600: loss = 147154108416.0
    ## step = 135800: loss = 40502042624.0
    ## step = 136000: loss = 70019571712.0
    ## step = 136000: Average Return = 0.0
    ## step = 136200: loss = 16454295552.0
    ## step = 136400: loss = 110282678272.0
    ## step = 136600: loss = 55171694592.0
    ## step = 136800: loss = 40868618240.0
    ## step = 137000: loss = 101625143296.0
    ## step = 137000: Average Return = 0.0
    ## step = 137200: loss = 103507673088.0
    ## step = 137400: loss = 55175405568.0
    ## step = 137600: loss = 57526525952.0
    ## step = 137800: loss = 102622248960.0
    ## step = 138000: loss = 196531830784.0
    ## step = 138000: Average Return = 0.0
    ## step = 138200: loss = 91481047040.0
    ## step = 138400: loss = 101530812416.0
    ## step = 138600: loss = 86967517184.0
    ## step = 138800: loss = 105637183488.0
    ## step = 139000: loss = 196623319040.0
    ## step = 139000: Average Return = 0.0
    ## step = 139200: loss = 105227075584.0
    ## step = 139400: loss = 125782949888.0
    ## step = 139600: loss = 116513071104.0
    ## step = 139800: loss = 81298259968.0
    ## step = 140000: loss = 31415431168.0
    ## step = 140000: Average Return = 0.0
    ## step = 140200: loss = 72719679488.0
    ## step = 140400: loss = 56992686080.0
    ## step = 140600: loss = 57318121472.0
    ## step = 140800: loss = 67710078976.0
    ## step = 141000: loss = 133173420032.0
    ## step = 141000: Average Return = 0.0
    ## step = 141200: loss = 91061641216.0
    ## step = 141400: loss = 126985764864.0
    ## step = 141600: loss = 77374554112.0
    ## step = 141800: loss = 46444650496.0
    ## step = 142000: loss = 115442909184.0
    ## step = 142000: Average Return = 0.0
    ## step = 142200: loss = 40535339008.0
    ## step = 142400: loss = 94268841984.0
    ## step = 142600: loss = 102049710080.0
    ## step = 142800: loss = 129664794624.0
    ## step = 143000: loss = 63938412544.0
    ## step = 143000: Average Return = 0.0
    ## step = 143200: loss = 31207770112.0
    ## step = 143400: loss = 96089292800.0
    ## step = 143600: loss = 100446396416.0
    ## step = 143800: loss = 32305707008.0
    ## step = 144000: loss = 19733520384.0
    ## step = 144000: Average Return = 0.0
    ## step = 144200: loss = 80818405376.0
    ## step = 144400: loss = 104831459328.0
    ## step = 144600: loss = 36600287232.0
    ## step = 144800: loss = 115433275392.0
    ## step = 145000: loss = 57167282176.0
    ## step = 145000: Average Return = 0.0
    ## step = 145200: loss = 41059049472.0
    ## step = 145400: loss = 147927269376.0
    ## step = 145600: loss = 105733144576.0
    ## step = 145800: loss = 155103199232.0
    ## step = 146000: loss = 76222103552.0
    ## step = 146000: Average Return = 0.0
    ## step = 146200: loss = 86348431360.0
    ## step = 146400: loss = 58761674752.0
    ## step = 146600: loss = 41215295488.0
    ## step = 146800: loss = 134789464064.0
    ## step = 147000: loss = 135860248576.0
    ## step = 147000: Average Return = 0.0
    ## step = 147200: loss = 141374930944.0
    ## step = 147400: loss = 93911597056.0
    ## step = 147600: loss = 87718117376.0
    ## step = 147800: loss = 67179659264.0
    ## step = 148000: loss = 74377043968.0
    ## step = 148000: Average Return = 0.0
    ## step = 148200: loss = 89345703936.0
    ## step = 148400: loss = 142819196928.0
    ## step = 148600: loss = 67716628480.0
    ## step = 148800: loss = 89344245760.0
    ## step = 149000: loss = 81822072832.0
    ## step = 149000: Average Return = 0.0
    ## step = 149200: loss = 148236369920.0
    ## step = 149400: loss = 113909129216.0
    ## step = 149600: loss = 80062578688.0
    ## step = 149800: loss = 201338880000.0
    ## step = 150000: loss = 158097473536.0
    ## step = 150000: Average Return = 0.0
    ## step = 150200: loss = 218538360832.0
    ## step = 150400: loss = 86311698432.0
    ## step = 150600: loss = 123045789696.0
    ## step = 150800: loss = 161496711168.0
    ## step = 151000: loss = 136667529216.0
    ## step = 151000: Average Return = 0.0
    ## step = 151200: loss = 121774202880.0
    ## step = 151400: loss = 61959766016.0
    ## step = 151600: loss = 167581761536.0
    ## step = 151800: loss = 43023044608.0
    ## step = 152000: loss = 113967185920.0
    ## step = 152000: Average Return = 0.0
    ## step = 152200: loss = 166219366400.0
    ## step = 152400: loss = 276931608576.0
    ## step = 152600: loss = 120422252544.0
    ## step = 152800: loss = 41305370624.0
    ## step = 153000: loss = 98061959168.0
    ## step = 153000: Average Return = 0.0
    ## step = 153200: loss = 196117168128.0
    ## step = 153400: loss = 140932284416.0
    ## step = 153600: loss = 110268243968.0
    ## step = 153800: loss = 54527352832.0
    ## step = 154000: loss = 149265809408.0
    ## step = 154000: Average Return = 0.0
    ## step = 154200: loss = 196478992384.0
    ## step = 154400: loss = 52247007232.0
    ## step = 154600: loss = 112384425984.0
    ## step = 154800: loss = 172896731136.0
    ## step = 155000: loss = 98709618688.0
    ## step = 155000: Average Return = 0.0
    ## step = 155200: loss = 108691324928.0
    ## step = 155400: loss = 139550490624.0
    ## step = 155600: loss = 234363535360.0
    ## step = 155800: loss = 121348726784.0
    ## step = 156000: loss = 134879010816.0
    ## step = 156000: Average Return = 0.0
    ## step = 156200: loss = 218562101248.0
    ## step = 156400: loss = 98846130176.0
    ## step = 156600: loss = 192317276160.0
    ## step = 156800: loss = 28936208384.0
    ## step = 157000: loss = 89791160320.0
    ## step = 157000: Average Return = 0.0
    ## step = 157200: loss = 46717231104.0
    ## step = 157400: loss = 148502282240.0
    ## step = 157600: loss = 83495067648.0
    ## step = 157800: loss = 154107199488.0
    ## step = 158000: loss = 119185965056.0
    ## step = 158000: Average Return = 0.0
    ## step = 158200: loss = 107902615552.0
    ## step = 158400: loss = 72312102912.0
    ## step = 158600: loss = 135438196736.0
    ## step = 158800: loss = 179484147712.0
    ## step = 159000: loss = 151606198272.0
    ## step = 159000: Average Return = 0.0
    ## step = 159200: loss = 7402707968.0
    ## step = 159400: loss = 49008009216.0
    ## step = 159600: loss = 165129093120.0
    ## step = 159800: loss = 219751612416.0
    ## step = 160000: loss = 137439182848.0
    ## step = 160000: Average Return = 0.0
    ## step = 160200: loss = 77644087296.0
    ## step = 160400: loss = 93248012288.0
    ## step = 160600: loss = 60265328640.0
    ## step = 160800: loss = 79589449728.0
    ## step = 161000: loss = 234106535936.0
    ## step = 161000: Average Return = 0.0
    ## step = 161200: loss = 87843954688.0
    ## step = 161400: loss = 95694364672.0
    ## step = 161600: loss = 60806746112.0
    ## step = 161800: loss = 163592437760.0
    ## step = 162000: loss = 115633455104.0
    ## step = 162000: Average Return = 0.0
    ## step = 162200: loss = 140061900800.0
    ## step = 162400: loss = 97162838016.0
    ## step = 162600: loss = 153065783296.0
    ## step = 162800: loss = 202223304704.0
    ## step = 163000: loss = 139251924992.0
    ## step = 163000: Average Return = 0.0
    ## step = 163200: loss = 112153165824.0
    ## step = 163400: loss = 90672939008.0
    ## step = 163600: loss = 181708816384.0
    ## step = 163800: loss = 135837671424.0
    ## step = 164000: loss = 148502102016.0
    ## step = 164000: Average Return = 0.0
    ## step = 164200: loss = 158263803904.0
    ## step = 164400: loss = 146865881088.0
    ## step = 164600: loss = 259795779584.0
    ## step = 164800: loss = 188397633536.0
    ## step = 165000: loss = 192931266560.0
    ## step = 165000: Average Return = 0.0
    ## step = 165200: loss = 177118445568.0
    ## step = 165400: loss = 166019563520.0
    ## step = 165600: loss = 218578812928.0
    ## step = 165800: loss = 268979716096.0
    ## step = 166000: loss = 197656887296.0
    ## step = 166000: Average Return = 0.0
    ## step = 166200: loss = 180869431296.0
    ## step = 166400: loss = 246011658240.0
    ## step = 166600: loss = 129868087296.0
    ## step = 166800: loss = 211370868736.0
    ## step = 167000: loss = 162427584512.0
    ## step = 167000: Average Return = 0.0
    ## step = 167200: loss = 232252801024.0
    ## step = 167400: loss = 104434753536.0
    ## step = 167600: loss = 87451803648.0
    ## step = 167800: loss = 106601570304.0
    ## step = 168000: loss = 140443189248.0
    ## step = 168000: Average Return = 0.0
    ## step = 168200: loss = 239456075776.0
    ## step = 168400: loss = 215987650560.0
    ## step = 168600: loss = 36345057280.0
    ## step = 168800: loss = 122704625664.0
    ## step = 169000: loss = 101101936640.0
    ## step = 169000: Average Return = 0.0
    ## step = 169200: loss = 98776743936.0
    ## step = 169400: loss = 130176630784.0
    ## step = 169600: loss = 94482325504.0
    ## step = 169800: loss = 156393013248.0
    ## step = 170000: loss = 251764654080.0
    ## step = 170000: Average Return = 0.0
    ## step = 170200: loss = 133836316672.0
    ## step = 170400: loss = 94277296128.0
    ## step = 170600: loss = 286652989440.0
    ## step = 170800: loss = 273388765184.0
    ## step = 171000: loss = 143143878656.0
    ## step = 171000: Average Return = 0.0
    ## step = 171200: loss = 179130138624.0
    ## step = 171400: loss = 222608670720.0
    ## step = 171600: loss = 201591308288.0
    ## step = 171800: loss = 140210831360.0
    ## step = 172000: loss = 302582071296.0
    ## step = 172000: Average Return = 0.0
    ## step = 172200: loss = 185508364288.0
    ## step = 172400: loss = 250614251520.0
    ## step = 172600: loss = 126537326592.0
    ## step = 172800: loss = 286935416832.0
    ## step = 173000: loss = 255115837440.0
    ## step = 173000: Average Return = 0.0
    ## step = 173200: loss = 171036049408.0
    ## step = 173400: loss = 60124033024.0
    ## step = 173600: loss = 56664379392.0
    ## step = 173800: loss = 126925094912.0
    ## step = 174000: loss = 168377532416.0
    ## step = 174000: Average Return = 0.0
    ## step = 174200: loss = 168515665920.0
    ## step = 174400: loss = 205954711552.0
    ## step = 174600: loss = 253699473408.0
    ## step = 174800: loss = 229926731776.0
    ## step = 175000: loss = 3637681152.0
    ## step = 175000: Average Return = 0.0
    ## step = 175200: loss = 351179112448.0
    ## step = 175400: loss = 61474873344.0
    ## step = 175600: loss = 171159240704.0
    ## step = 175800: loss = 203486134272.0
    ## step = 176000: loss = 138972102656.0
    ## step = 176000: Average Return = 0.0
    ## step = 176200: loss = 257922105344.0
    ## step = 176400: loss = 190238212096.0
    ## step = 176600: loss = 203153539072.0
    ## step = 176800: loss = 83717668864.0
    ## step = 177000: loss = 286179753984.0
    ## step = 177000: Average Return = 0.0
    ## step = 177200: loss = 126072741888.0
    ## step = 177400: loss = 268875366400.0
    ## step = 177600: loss = 72975138816.0
    ## step = 177800: loss = 99312648192.0
    ## step = 178000: loss = 196220125184.0
    ## step = 178000: Average Return = 0.0
    ## step = 178200: loss = 166825721856.0
    ## step = 178400: loss = 461591937024.0
    ## step = 178600: loss = 289758609408.0
    ## step = 178800: loss = 340451491840.0
    ## step = 179000: loss = 394570465280.0
    ## step = 179000: Average Return = 0.0
    ## step = 179200: loss = 153910427648.0
    ## step = 179400: loss = 236696305664.0
    ## step = 179600: loss = 189107306496.0
    ## step = 179800: loss = 103162912768.0
    ## step = 180000: loss = 51943849984.0
    ## step = 180000: Average Return = 0.0
    ## step = 180200: loss = 280298192896.0
    ## step = 180400: loss = 279448780800.0
    ## step = 180600: loss = 14276370432.0
    ## step = 180800: loss = 420036378624.0
    ## step = 181000: loss = 68818272256.0
    ## step = 181000: Average Return = 0.0
    ## step = 181200: loss = 217120309248.0
    ## step = 181400: loss = 338278023168.0
    ## step = 181600: loss = 365602963456.0
    ## step = 181800: loss = 178688098304.0
    ## step = 182000: loss = 153911066624.0
    ## step = 182000: Average Return = 0.0
    ## step = 182200: loss = 41211174912.0
    ## step = 182400: loss = 220732751872.0
    ## step = 182600: loss = 236567478272.0
    ## step = 182800: loss = 139737874432.0
    ## step = 183000: loss = 294355861504.0
    ## step = 183000: Average Return = 0.0
    ## step = 183200: loss = 56852881408.0
    ## step = 183400: loss = 300521619456.0
    ## step = 183600: loss = 246903521280.0
    ## step = 183800: loss = 229665161216.0
    ## step = 184000: loss = 152981159936.0
    ## step = 184000: Average Return = 0.0
    ## step = 184200: loss = 170396581888.0
    ## step = 184400: loss = 123301019648.0
    ## step = 184600: loss = 6389574144.0
    ## step = 184800: loss = 553953263616.0
    ## step = 185000: loss = 293156618240.0
    ## step = 185000: Average Return = 0.0
    ## step = 185200: loss = 215935795200.0
    ## step = 185400: loss = 75142840320.0
    ## step = 185600: loss = 305159667712.0
    ## step = 185800: loss = 252292481024.0
    ## step = 186000: loss = 201734094848.0
    ## step = 186000: Average Return = 0.0
    ## step = 186200: loss = 355127885824.0
    ## step = 186400: loss = 131058417664.0
    ## step = 186600: loss = 218237861888.0
    ## step = 186800: loss = 157234364416.0
    ## step = 187000: loss = 399828287488.0
    ## step = 187000: Average Return = 0.0
    ## step = 187200: loss = 295644430336.0
    ## step = 187400: loss = 141441859584.0
    ## step = 187600: loss = 310095052800.0
    ## step = 187800: loss = 178942902272.0
    ## step = 188000: loss = 308575174656.0
    ## step = 188000: Average Return = 0.0
    ## step = 188200: loss = 348394848256.0
    ## step = 188400: loss = 314557202432.0
    ## step = 188600: loss = 59315142656.0
    ## step = 188800: loss = 267197562880.0
    ## step = 189000: loss = 447544754176.0
    ## step = 189000: Average Return = 0.0
    ## step = 189200: loss = 349070295040.0
    ## step = 189400: loss = 417726070784.0
    ## step = 189600: loss = 241111875584.0
    ## step = 189800: loss = 76655493120.0
    ## step = 190000: loss = 190072848384.0
    ## step = 190000: Average Return = 0.0
    ## step = 190200: loss = 209845518336.0
    ## step = 190400: loss = 489785262080.0
    ## step = 190600: loss = 219584053248.0
    ## step = 190800: loss = 467019366400.0
    ## step = 191000: loss = 146895519744.0
    ## step = 191000: Average Return = 0.0
    ## step = 191200: loss = 243651559424.0
    ## step = 191400: loss = 408220139520.0
    ## step = 191600: loss = 106445398016.0
    ## step = 191800: loss = 59411349504.0
    ## step = 192000: loss = 170821435392.0
    ## step = 192000: Average Return = 0.0
    ## step = 192200: loss = 193666449408.0
    ## step = 192400: loss = 389214568448.0
    ## step = 192600: loss = 173780238336.0
    ## step = 192800: loss = 268847087616.0
    ## step = 193000: loss = 390533021696.0
    ## step = 193000: Average Return = 0.0
    ## step = 193200: loss = 157650173952.0
    ## step = 193400: loss = 233039872000.0
    ## step = 193600: loss = 397142097920.0
    ## step = 193800: loss = 539107328000.0
    ## step = 194000: loss = 467118587904.0
    ## step = 194000: Average Return = 0.0
    ## step = 194200: loss = 353231699968.0
    ## step = 194400: loss = 473658097664.0
    ## step = 194600: loss = 203278614528.0
    ## step = 194800: loss = 60514660352.0
    ## step = 195000: loss = 182282256384.0
    ## step = 195000: Average Return = 0.0
    ## step = 195200: loss = 436345241600.0
    ## step = 195400: loss = 223961661440.0
    ## step = 195600: loss = 485479677952.0
    ## step = 195800: loss = 160607043584.0
    ## step = 196000: loss = 351357927424.0
    ## step = 196000: Average Return = 0.0
    ## step = 196200: loss = 215163568128.0
    ## step = 196400: loss = 324896718848.0
    ## step = 196600: loss = 393149251584.0
    ## step = 196800: loss = 404977909760.0
    ## step = 197000: loss = 178602721280.0
    ## step = 197000: Average Return = 0.0
    ## step = 197200: loss = 139643994112.0
    ## step = 197400: loss = 100350099456.0
    ## step = 197600: loss = 139946426368.0
    ## step = 197800: loss = 175790342144.0
    ## step = 198000: loss = 200017920000.0
    ## step = 198000: Average Return = 0.0
    ## step = 198200: loss = 323619192832.0
    ## step = 198400: loss = 398133362688.0
    ## step = 198600: loss = 151602184192.0
    ## step = 198800: loss = 209875419136.0
    ## step = 199000: loss = 184105828352.0
    ## step = 199000: Average Return = 0.0
    ## step = 199200: loss = 397623623680.0
    ## step = 199400: loss = 209021435904.0
    ## step = 199600: loss = 143712157696.0
    ## step = 199800: loss = 383662358528.0
    ## step = 200000: loss = 229625757696.0
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

    ## [<matplotlib.lines.Line2D object at 0x7f3580558080>]

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

    ## (-0.039452713727951054, 25.0)

``` python
plt.show()
```

<img src="dqn_fishing_files/figure-gfm/unnamed-chunk-24-1.png" width="672" />

``` python
plt.savefig('foo.png', bbox_inches='tight')
```
