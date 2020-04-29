
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
reticulate::use_virtualenv("~/.virtualenvs/tf-agents")
#reticulate::py_discover_config()
```

``` python


from __future__ import absolute_import, division, print_function

import gym_fishing
import matplotlib.pyplot as plt

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
replay_buffer_max_length = 2000000  # @param {type:"integer"}
batch_size = 64                     # @param {type:"integer"}
learning_rate = 1e-3                # @param {type:"number"}
log_interval = 500                  # @param {type:"integer"}
num_eval_episodes = 100            # @param {type:"integer"}
eval_interval = 1000                # @param {type:"integer"}
discount = 0.99
```

## Environment

``` python
env_name = 'fishing-v0'
env = suite_gym.load(env_name, discount = discount)
```

    ## /home/cboettig/.virtualenvs/tf-agents/lib/python3.6/site-packages/gym/logger.py:30: UserWarning: [33mWARN: Box bound precision lowered by casting to float32[0m
    ##   warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))

``` python
env.reset()
```

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([0.75], dtype=float32))

The `environment.step` method takes an `action` in the environment and
returns a `TimeStep` tuple containing the next observation of the
environment and the reward for the action.

The `time_step_spec()` method returns the specification for the
`TimeStep` tuple. Its `observation` attribute shows the shape of
observations, the data types, and the ranges of allowed values. The
`reward` attribute shows the same details for the
    reward.

``` python
env.time_step_spec().observation
```

    ## BoundedArraySpec(shape=(1,), dtype=dtype('float32'), name='observation', minimum=0.0, maximum=2.0)

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

    ## BoundedArraySpec(shape=(), dtype=dtype('int64'), name='action', minimum=0, maximum=2)

In the `fishing-v0` environment:

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

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([0.75], dtype=float32))

``` python
action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
```

    ## Next time step:

``` python
print(next_time_step)
```

    ## TimeStep(step_type=array(1, dtype=int32), reward=array(0.015, dtype=float32), discount=array(0.99, dtype=float32), observation=array([0.7544775], dtype=float32))

Usually two environments are instantiated: one for training and one for
evaluation.

``` python
train_py_env = suite_gym.load(env_name, discount = discount)
eval_py_env = suite_gym.load(env_name, discount = discount)
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
    suite_gym.load(env_name, discount = discount))

time_step = example_environment.reset()
random_policy.action(time_step)
```

    ## PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int64, numpy=array([1])>, state=(), info=())

## Metrics and Evaluation

The most common metric used to evaluate a policy is the average return.
The return is the sum of rewards obtained while running a policy in an
environment for an episode. Several episodes are run, creating an
average return.

The following function computes the average return of a policy, given
the policy, environment, and a number of episodes.

``` python
#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=100):

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

    ## 1.7696611

``` python

def simulate(environment, policy):
  total_return = 0.0
  time_step = environment.reset()
  episode_return = 0.0

  output = np.zeros(shape = (1000, 4))
  for it in range(1000):
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
    episode_return += time_step.reward
    output[it] = (it, time_step.observation, action_step.action, episode_return)

  return output
```

``` python
out = simulate(eval_env, random_policy)
```

``` python
plt.plot(out[:,1])
plt.ylabel('state')
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-20-1.png" width="672" />

``` python
plt.plot(out[:,2])
plt.ylabel('action')
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-20-2.png" width="672" />

``` python
plt.plot(out[:,3])
plt.ylabel('reward')
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-20-3.png" width="672" />

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
#agent.collect_data_spec
agent.collect_data_spec.observation
```

    ## BoundedTensorSpec(shape=(1,), dtype=tf.float32, name='observation', minimum=array(0., dtype=float32), maximum=array(2., dtype=float32))

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

Let’s take a look:

``` python
x_i = iter(replay_buffer.as_dataset()).next()
x_i
```

    ## (Trajectory(step_type=<tf.Tensor: shape=(), dtype=int32, numpy=1>, observation=<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.87683916], dtype=float32)>, action=<tf.Tensor: shape=(), dtype=int64, numpy=0>, policy_info=(), next_step_type=<tf.Tensor: shape=(), dtype=int32, numpy=1>, reward=<tf.Tensor: shape=(), dtype=float32, numpy=0.00957359>, discount=<tf.Tensor: shape=(), dtype=float32, numpy=0.99>), BufferInfo(ids=<tf.Tensor: shape=(), dtype=int64, numpy=39>, probabilities=<tf.Tensor: shape=(), dtype=float32, numpy=0.01>))

Wow, but that’s unreadable. Let’s add some whitespace (hardcoded example
below, numeric values may differ):

    (
    Trajectory(
               step_type   = <tf.Tensor: shape=(), dtype=int32, numpy=1>,
               observation = <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.97752917], dtype=float32)>, 
               action      = <tf.Tensor: shape=(), dtype=int64, numpy=0>, 
               policy_info=(), 
            next_step_type = <tf.Tensor: shape=(), dtype=int32, numpy=1>, 
               reward      = <tf.Tensor: shape=(), dtype=float32, numpy=0.00044579292>, 
               discount    = <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
               ), 
    BufferInfo(ids           = <tf.Tensor: shape=(), dtype=int64, numpy=87>, 
               probabilities = <tf.Tensor: shape=(), dtype=float32, numpy=0.01>)
    )

``` python
x_i[0].observation.numpy()[0]
```

    ## 0.87683916

``` python
x_i[0].action.numpy()
```

    ## 0

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
```

``` python
iterator = iter(dataset)
experience, unused_info = next(iterator)
experience.action[:,1]
```

    ## <tf.Tensor: shape=(64,), dtype=int64, numpy=
    ## array([1, 0, 2, 0, 2, 0, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 1,
    ##        2, 2, 0, 0, 2, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 2, 2, 2, 1, 1, 0, 1,
    ##        1, 0, 0, 1, 2, 2, 0, 2, 0, 1, 0, 0, 2, 2, 0, 2, 1, 0, 0, 0])>

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
returns
```

    ## [1.0446321]

``` python
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

    ## step = 500: loss = 0.00041508660069666803
    ## step = 1000: loss = 0.0003100429312326014
    ## step = 1000: Average Return = 12.499878883361816
    ## step = 1500: loss = 0.00045962524018250406
    ## step = 2000: loss = 0.0003571207053028047
    ## step = 2000: Average Return = 1.0857243537902832
    ## step = 2500: loss = 0.0002469278406351805
    ## step = 3000: loss = 0.0007014636066742241
    ## step = 3000: Average Return = 23.97899627685547
    ## step = 3500: loss = 0.0006873199017718434
    ## step = 4000: loss = 0.0003172701981384307
    ## step = 4000: Average Return = 0.05000000819563866
    ## step = 4500: loss = 0.0026111360639333725
    ## step = 5000: loss = 0.002444352488964796
    ## step = 5000: Average Return = 24.097021102905273
    ## step = 5500: loss = 0.0048406003043055534
    ## step = 6000: loss = 0.0008520166738890111
    ## step = 6000: Average Return = 24.4843807220459
    ## step = 6500: loss = 0.009111703373491764
    ## step = 7000: loss = 0.0007441607885994017
    ## step = 7000: Average Return = 24.576662063598633
    ## step = 7500: loss = 0.0044975061900913715
    ## step = 8000: loss = 0.0037520267069339752
    ## step = 8000: Average Return = 24.638261795043945
    ## step = 8500: loss = 0.0006257598288357258
    ## step = 9000: loss = 0.002574161160737276
    ## step = 9000: Average Return = 24.563007354736328
    ## step = 9500: loss = 0.002109786495566368
    ## step = 10000: loss = 0.0541391558945179
    ## step = 10000: Average Return = 23.952800750732422
    ## step = 10500: loss = 0.0016952529549598694
    ## step = 11000: loss = 0.0017114244401454926
    ## step = 11000: Average Return = 12.499878883361816
    ## step = 11500: loss = 0.0007175914943218231
    ## step = 12000: loss = 0.0008064503781497478
    ## step = 12000: Average Return = 23.281761169433594
    ## step = 12500: loss = 0.0021274457685649395
    ## step = 13000: loss = 0.0023901257663965225
    ## step = 13000: Average Return = 24.22067642211914
    ## step = 13500: loss = 0.002071435796096921
    ## step = 14000: loss = 0.0006992188282310963
    ## step = 14000: Average Return = 23.375890731811523
    ## step = 14500: loss = 0.0019326854962855577
    ## step = 15000: loss = 0.11230131983757019
    ## step = 15000: Average Return = 22.64626693725586
    ## step = 15500: loss = 0.0010921249631792307
    ## step = 16000: loss = 0.0024257516488432884
    ## step = 16000: Average Return = 12.499878883361816
    ## step = 16500: loss = 0.0026362603530287743
    ## step = 17000: loss = 0.0025039450265467167
    ## step = 17000: Average Return = 12.499878883361816
    ## step = 17500: loss = 0.0015444150194525719
    ## step = 18000: loss = 0.0013929157285019755
    ## step = 18000: Average Return = 23.941160202026367
    ## step = 18500: loss = 0.0018550183158367872
    ## step = 19000: loss = 0.002491926308721304
    ## step = 19000: Average Return = 23.331892013549805
    ## step = 19500: loss = 0.1133069396018982
    ## step = 20000: loss = 0.06870725750923157
    ## step = 20000: Average Return = 22.677600860595703
    ## step = 20500: loss = 0.0012968267546966672
    ## step = 21000: loss = 0.0010041336063295603
    ## step = 21000: Average Return = 23.57443618774414
    ## step = 21500: loss = 0.0019583709072321653
    ## step = 22000: loss = 0.0008648406947031617
    ## step = 22000: Average Return = 24.035869598388672
    ## step = 22500: loss = 0.001881257863715291
    ## step = 23000: loss = 0.0007317801937460899
    ## step = 23000: Average Return = 23.322893142700195
    ## step = 23500: loss = 0.0022862893529236317
    ## step = 24000: loss = 0.0019839266315102577
    ## step = 24000: Average Return = 23.907472610473633
    ## step = 24500: loss = 0.0010284977033734322
    ## step = 25000: loss = 0.0014464892446994781
    ## step = 25000: Average Return = 23.963359832763672
    ## step = 25500: loss = 0.00207173777744174
    ## step = 26000: loss = 0.0007686762255616486
    ## step = 26000: Average Return = 22.896099090576172
    ## step = 26500: loss = 0.000689235981553793
    ## step = 27000: loss = 0.001352743012830615
    ## step = 27000: Average Return = 23.916484832763672
    ## step = 27500: loss = 0.0017216941341757774
    ## step = 28000: loss = 0.002174260327592492
    ## step = 28000: Average Return = 0.05000000819563866
    ## step = 28500: loss = 0.0011158485431224108
    ## step = 29000: loss = 0.002727989573031664
    ## step = 29000: Average Return = 12.499878883361816
    ## step = 29500: loss = 0.0014539591502398252
    ## step = 30000: loss = 0.0010088577400892973
    ## step = 30000: Average Return = 24.257810592651367
    ## step = 30500: loss = 0.0019294697558507323
    ## step = 31000: loss = 0.0011954533401876688
    ## step = 31000: Average Return = 24.177080154418945
    ## step = 31500: loss = 0.0012926488416269422
    ## step = 32000: loss = 0.0021092293318361044
    ## step = 32000: Average Return = 24.419897079467773
    ## step = 32500: loss = 0.0005495784571394324
    ## step = 33000: loss = 0.0009829235495999455
    ## step = 33000: Average Return = 23.908123016357422
    ## step = 33500: loss = 0.0028383079916238785
    ## step = 34000: loss = 0.025130387395620346
    ## step = 34000: Average Return = 0.23749984800815582
    ## step = 34500: loss = 0.0012390261981636286
    ## step = 35000: loss = 0.0008529928745701909
    ## step = 35000: Average Return = 12.499878883361816
    ## step = 35500: loss = 0.0006981366313993931
    ## step = 36000: loss = 0.00042605274938978255
    ## step = 36000: Average Return = 24.4475154876709
    ## step = 36500: loss = 0.0015102621400728822
    ## step = 37000: loss = 0.0018020719289779663
    ## step = 37000: Average Return = 24.6562557220459
    ## step = 37500: loss = 0.0012079320149496198
    ## step = 38000: loss = 0.0016517348121851683
    ## step = 38000: Average Return = 23.942304611206055
    ## step = 38500: loss = 0.0007965233526192605
    ## step = 39000: loss = 0.001434275764040649
    ## step = 39000: Average Return = 12.499878883361816
    ## step = 39500: loss = 0.0016145206755027175
    ## step = 40000: loss = 0.0007919894414953887
    ## step = 40000: Average Return = 24.534889221191406
    ## step = 40500: loss = 0.0010598384542390704
    ## step = 41000: loss = 0.0023714578710496426
    ## step = 41000: Average Return = 24.322364807128906
    ## step = 41500: loss = 0.0015405595768243074
    ## step = 42000: loss = 0.0012135350843891501
    ## step = 42000: Average Return = 23.470539093017578
    ## step = 42500: loss = 0.0033284735400229692
    ## step = 43000: loss = 0.0009658304043114185
    ## step = 43000: Average Return = 24.613750457763672
    ## step = 43500: loss = 0.003259408287703991
    ## step = 44000: loss = 0.001667981268838048
    ## step = 44000: Average Return = 24.495868682861328
    ## step = 44500: loss = 0.0008697114535607398
    ## step = 45000: loss = 0.0007782280445098877
    ## step = 45000: Average Return = 23.00552749633789
    ## step = 45500: loss = 0.001645440119318664
    ## step = 46000: loss = 0.0033663662616163492
    ## step = 46000: Average Return = 24.668066024780273
    ## step = 46500: loss = 0.002454116242006421
    ## step = 47000: loss = 0.0037452499382197857
    ## step = 47000: Average Return = 24.497560501098633
    ## step = 47500: loss = 0.001642104354687035
    ## step = 48000: loss = 0.0011847235728055239
    ## step = 48000: Average Return = 24.794679641723633
    ## step = 48500: loss = 0.001703873509541154
    ## step = 49000: loss = 0.0010246098972856998
    ## step = 49000: Average Return = 24.001466751098633
    ## step = 49500: loss = 0.0007467406685464084
    ## step = 50000: loss = 0.0011871617753058672
    ## step = 50000: Average Return = 24.433368682861328
    ## step = 50500: loss = 0.001141777029260993
    ## step = 51000: loss = 0.0018943912582471967
    ## step = 51000: Average Return = 0.05000000819563866
    ## step = 51500: loss = 0.0007053549634292722
    ## step = 52000: loss = 0.0020065929275006056
    ## step = 52000: Average Return = 24.488788604736328
    ## step = 52500: loss = 0.0028973943553864956
    ## step = 53000: loss = 0.001222761464305222
    ## step = 53000: Average Return = 12.499878883361816
    ## step = 53500: loss = 0.0009642557706683874
    ## step = 54000: loss = 0.0006680078222416341
    ## step = 54000: Average Return = 12.499878883361816
    ## step = 54500: loss = 0.0014373557642102242
    ## step = 55000: loss = 0.0020868512801826
    ## step = 55000: Average Return = 24.8217716217041
    ## step = 55500: loss = 0.0005827780114486814
    ## step = 56000: loss = 0.0029314772691577673
    ## step = 56000: Average Return = 24.414817810058594
    ## step = 56500: loss = 0.00070133653935045
    ## step = 57000: loss = 0.0013856401201337576
    ## step = 57000: Average Return = 24.415916442871094
    ## step = 57500: loss = 0.0011387438280507922
    ## step = 58000: loss = 0.0007883461657911539
    ## step = 58000: Average Return = 0.05000000819563866
    ## step = 58500: loss = 0.0020059444941580296
    ## step = 59000: loss = 0.0019461287884041667
    ## step = 59000: Average Return = 24.673683166503906
    ## step = 59500: loss = 0.0007808437803760171
    ## step = 60000: loss = 0.15805964171886444
    ## step = 60000: Average Return = 12.499878883361816
    ## step = 60500: loss = 0.0009230725700035691
    ## step = 61000: loss = 0.13862673938274384
    ## step = 61000: Average Return = 12.499878883361816
    ## step = 61500: loss = 0.0010418272577226162
    ## step = 62000: loss = 0.0018658728804439306
    ## step = 62000: Average Return = 24.666040420532227
    ## step = 62500: loss = 0.0005268298555165529
    ## step = 63000: loss = 0.0018147025257349014
    ## step = 63000: Average Return = 24.766359329223633
    ## step = 63500: loss = 0.0006347064627334476
    ## step = 64000: loss = 0.001104823313653469
    ## step = 64000: Average Return = 24.594560623168945
    ## step = 64500: loss = 0.0005988309858366847
    ## step = 65000: loss = 0.002288350835442543
    ## step = 65000: Average Return = 24.97031021118164
    ## step = 65500: loss = 0.001057108398526907
    ## step = 66000: loss = 0.0026814229786396027
    ## step = 66000: Average Return = 24.854005813598633
    ## step = 66500: loss = 0.002001985441893339
    ## step = 67000: loss = 0.0013093978632241488
    ## step = 67000: Average Return = 15.00012493133545
    ## step = 67500: loss = 0.0009836886310949922
    ## step = 68000: loss = 0.0008613409590907395
    ## step = 68000: Average Return = 21.908353805541992
    ## step = 68500: loss = 0.001574514084495604
    ## step = 69000: loss = 0.0007803082698956132
    ## step = 69000: Average Return = 24.90892791748047
    ## step = 69500: loss = 0.0014158978592604399
    ## step = 70000: loss = 0.0023685693740844727
    ## step = 70000: Average Return = 0.7283428311347961
    ## step = 70500: loss = 0.0005451845936477184
    ## step = 71000: loss = 0.0019404151244089007
    ## step = 71000: Average Return = 24.566816329956055
    ## step = 71500: loss = 0.0016526788240298629
    ## step = 72000: loss = 0.0032083417754620314
    ## step = 72000: Average Return = 24.757160186767578
    ## step = 72500: loss = 0.001163874170742929
    ## step = 73000: loss = 0.0019504799274727702
    ## step = 73000: Average Return = 22.641273498535156
    ## step = 73500: loss = 0.001493633957579732
    ## step = 74000: loss = 0.0012867712648585439
    ## step = 74000: Average Return = 24.203046798706055
    ## step = 74500: loss = 0.0014282844495028257
    ## step = 75000: loss = 0.0010855314321815968
    ## step = 75000: Average Return = 24.768722534179688
    ## step = 75500: loss = 0.001221624668687582
    ## step = 76000: loss = 0.16168083250522614
    ## step = 76000: Average Return = 0.05000000819563866
    ## step = 76500: loss = 0.0025346071925014257
    ## step = 77000: loss = 0.0021043396554887295
    ## step = 77000: Average Return = 24.449316024780273
    ## step = 77500: loss = 0.0020438723731786013
    ## step = 78000: loss = 0.0013651662738993764
    ## step = 78000: Average Return = 24.68740463256836
    ## step = 78500: loss = 0.0017298063030466437
    ## step = 79000: loss = 0.001740701962262392
    ## step = 79000: Average Return = 24.483884811401367
    ## step = 79500: loss = 0.002240682952105999
    ## step = 80000: loss = 0.0028687790036201477
    ## step = 80000: Average Return = 12.499878883361816
    ## step = 80500: loss = 0.002428340259939432
    ## step = 81000: loss = 0.002036465099081397
    ## step = 81000: Average Return = 24.288719177246094
    ## step = 81500: loss = 0.001750859897583723
    ## step = 82000: loss = 0.0010374935809522867
    ## step = 82000: Average Return = 12.499878883361816
    ## step = 82500: loss = 0.0012735207565128803
    ## step = 83000: loss = 0.001558380899950862
    ## step = 83000: Average Return = 24.659812927246094
    ## step = 83500: loss = 0.0017408154672011733
    ## step = 84000: loss = 0.002461859490722418
    ## step = 84000: Average Return = 23.431316375732422
    ## step = 84500: loss = 0.0014912886545062065
    ## step = 85000: loss = 0.0010015391744673252
    ## step = 85000: Average Return = 24.792648315429688
    ## step = 85500: loss = 0.0018041752045974135
    ## step = 86000: loss = 0.11566281318664551
    ## step = 86000: Average Return = 23.068744659423828
    ## step = 86500: loss = 0.000960618257522583
    ## step = 87000: loss = 0.0017759379697963595
    ## step = 87000: Average Return = 24.598852157592773
    ## step = 87500: loss = 0.0013599786907434464
    ## step = 88000: loss = 0.0011967081809416413
    ## step = 88000: Average Return = 22.94436264038086
    ## step = 88500: loss = 0.0013749924255535007
    ## step = 89000: loss = 0.0015936684794723988
    ## step = 89000: Average Return = 0.33749985694885254
    ## step = 89500: loss = 0.001171695999801159
    ## step = 90000: loss = 0.0012711319141089916
    ## step = 90000: Average Return = 24.208168029785156
    ## step = 90500: loss = 0.113337941467762
    ## step = 91000: loss = 0.0010065138339996338
    ## step = 91000: Average Return = 23.021974563598633
    ## step = 91500: loss = 0.0006901763845235109
    ## step = 92000: loss = 0.0005000691162422299
    ## step = 92000: Average Return = 12.499878883361816
    ## step = 92500: loss = 0.0020809066481888294
    ## step = 93000: loss = 0.00141172728035599
    ## step = 93000: Average Return = 24.348657608032227
    ## step = 93500: loss = 0.0009063492761924863
    ## step = 94000: loss = 0.0006433433154597878
    ## step = 94000: Average Return = 23.217613220214844
    ## step = 94500: loss = 0.0013690830674022436
    ## step = 95000: loss = 0.00301115564070642
    ## step = 95000: Average Return = 23.068744659423828
    ## step = 95500: loss = 0.0019484101794660091
    ## step = 96000: loss = 0.0012039359426125884
    ## step = 96000: Average Return = 24.527612686157227
    ## step = 96500: loss = 0.001388483215123415
    ## step = 97000: loss = 0.001824430306442082
    ## step = 97000: Average Return = 24.479272842407227
    ## step = 97500: loss = 0.030001293867826462
    ## step = 98000: loss = 0.0011858772486448288
    ## step = 98000: Average Return = 21.667991638183594
    ## step = 98500: loss = 0.0013810497475787997
    ## step = 99000: loss = 0.0013781798770651221
    ## step = 99000: Average Return = 22.136850357055664
    ## step = 99500: loss = 0.001993674784898758
    ## step = 100000: loss = 0.0009244412649422884
    ## step = 100000: Average Return = 20.663965225219727
    ## step = 100500: loss = 0.0016106369439512491
    ## step = 101000: loss = 0.0014276073779910803
    ## step = 101000: Average Return = 24.656469345092773
    ## step = 101500: loss = 0.05771505832672119
    ## step = 102000: loss = 0.0032097126822918653
    ## step = 102000: Average Return = 12.499878883361816
    ## step = 102500: loss = 0.0015859772684052587
    ## step = 103000: loss = 0.0018281295197084546
    ## step = 103000: Average Return = 23.102516174316406
    ## step = 103500: loss = 0.003692534752190113
    ## step = 104000: loss = 0.03185390681028366
    ## step = 104000: Average Return = 23.068744659423828
    ## step = 104500: loss = 0.14538243412971497
    ## step = 105000: loss = 0.0018489975482225418
    ## step = 105000: Average Return = 24.635906219482422
    ## step = 105500: loss = 0.0034957951866090298
    ## step = 106000: loss = 0.0009628172847442329
    ## step = 106000: Average Return = 0.05000000819563866
    ## step = 106500: loss = 0.0017360970377922058
    ## step = 107000: loss = 0.0028112437576055527
    ## step = 107000: Average Return = 23.75021743774414
    ## step = 107500: loss = 0.0010898669715970755
    ## step = 108000: loss = 0.001935088774189353
    ## step = 108000: Average Return = 24.63637924194336
    ## step = 108500: loss = 0.0017934847855940461
    ## step = 109000: loss = 0.0016757887788116932
    ## step = 109000: Average Return = 24.662330627441406
    ## step = 109500: loss = 0.0012860032729804516
    ## step = 110000: loss = 0.006968035362660885
    ## step = 110000: Average Return = 24.057764053344727
    ## step = 110500: loss = 0.0018478911370038986
    ## step = 111000: loss = 0.0016492744907736778
    ## step = 111000: Average Return = 22.066984176635742
    ## step = 111500: loss = 0.0008572880178689957
    ## step = 112000: loss = 0.0007642944110557437
    ## step = 112000: Average Return = 22.67727279663086
    ## step = 112500: loss = 0.001718263840302825
    ## step = 113000: loss = 0.0015434310771524906
    ## step = 113000: Average Return = 23.103839874267578
    ## step = 113500: loss = 0.0016405591741204262
    ## step = 114000: loss = 0.0015028638299554586
    ## step = 114000: Average Return = 23.986839294433594
    ## step = 114500: loss = 0.0016712601063773036
    ## step = 115000: loss = 0.003324308432638645
    ## step = 115000: Average Return = 23.011070251464844
    ## step = 115500: loss = 0.0010597012005746365
    ## step = 116000: loss = 0.003954798448830843
    ## step = 116000: Average Return = 0.17500005662441254
    ## step = 116500: loss = 0.001802136772312224
    ## step = 117000: loss = 0.0023441212251782417
    ## step = 117000: Average Return = 24.572973251342773
    ## step = 117500: loss = 0.08703397959470749
    ## step = 118000: loss = 0.0009637754992581904
    ## step = 118000: Average Return = 23.875566482543945
    ## step = 118500: loss = 0.0010542471427470446
    ## step = 119000: loss = 0.00196454138495028
    ## step = 119000: Average Return = 24.88526153564453
    ## step = 119500: loss = 0.00174346670974046
    ## step = 120000: loss = 0.0023205396719276905
    ## step = 120000: Average Return = 24.659692764282227
    ## step = 120500: loss = 0.0013377222931012511
    ## step = 121000: loss = 0.1488901525735855
    ## step = 121000: Average Return = 24.46141815185547
    ## step = 121500: loss = 0.0019017598824575543
    ## step = 122000: loss = 0.0011854117037728429
    ## step = 122000: Average Return = 23.21778106689453
    ## step = 122500: loss = 0.0015896904515102506
    ## step = 123000: loss = 0.0018932335078716278
    ## step = 123000: Average Return = 24.736791610717773
    ## step = 123500: loss = 0.0009756524232216179
    ## step = 124000: loss = 0.0023199315182864666
    ## step = 124000: Average Return = 12.499878883361816
    ## step = 124500: loss = 0.0018946817144751549
    ## step = 125000: loss = 0.0015421346761286259
    ## step = 125000: Average Return = 23.186521530151367
    ## step = 125500: loss = 0.0017277220031246543
    ## step = 126000: loss = 0.0010864341165870428
    ## step = 126000: Average Return = 23.270126342773438
    ## step = 126500: loss = 0.003513420233502984
    ## step = 127000: loss = 0.0017658004071563482
    ## step = 127000: Average Return = 24.216411590576172
    ## step = 127500: loss = 0.0047722626477479935
    ## step = 128000: loss = 0.0013978471979498863
    ## step = 128000: Average Return = 24.163251876831055
    ## step = 128500: loss = 0.0017469001468271017
    ## step = 129000: loss = 0.001035639550536871
    ## step = 129000: Average Return = 12.499878883361816
    ## step = 129500: loss = 0.0022403053008019924
    ## step = 130000: loss = 0.12657256424427032
    ## step = 130000: Average Return = 24.27541732788086
    ## step = 130500: loss = 0.1041143462061882
    ## step = 131000: loss = 0.001675488892942667
    ## step = 131000: Average Return = 24.850828170776367
    ## step = 131500: loss = 0.0013296394608914852
    ## step = 132000: loss = 0.0013747100019827485
    ## step = 132000: Average Return = 24.925050735473633
    ## step = 132500: loss = 0.001445523463189602
    ## step = 133000: loss = 0.0015364966820925474
    ## step = 133000: Average Return = 21.21546173095703
    ## step = 133500: loss = 0.002381454687565565
    ## step = 134000: loss = 0.0014609606005251408
    ## step = 134000: Average Return = 24.720678329467773
    ## step = 134500: loss = 0.0008742882637307048
    ## step = 135000: loss = 0.00275331805460155
    ## step = 135000: Average Return = 24.917892456054688
    ## step = 135500: loss = 0.002373700961470604
    ## step = 136000: loss = 0.0014137268299236894
    ## step = 136000: Average Return = 12.499878883361816
    ## step = 136500: loss = 0.0018515045521780849
    ## step = 137000: loss = 0.0011184120085090399
    ## step = 137000: Average Return = 12.499878883361816
    ## step = 137500: loss = 0.15589609742164612
    ## step = 138000: loss = 0.0009425695170648396
    ## step = 138000: Average Return = 23.897052764892578
    ## step = 138500: loss = 0.0020760525949299335
    ## step = 139000: loss = 0.0013235677033662796
    ## step = 139000: Average Return = 12.499878883361816
    ## step = 139500: loss = 0.0011408296413719654
    ## step = 140000: loss = 0.0019467558013275266
    ## step = 140000: Average Return = 23.217613220214844
    ## step = 140500: loss = 0.0009020722936838865
    ## step = 141000: loss = 0.0014157078694552183
    ## step = 141000: Average Return = 12.499878883361816
    ## step = 141500: loss = 0.0011313599534332752
    ## step = 142000: loss = 0.0009682426461949944
    ## step = 142000: Average Return = 24.88768768310547
    ## step = 142500: loss = 0.0010173209011554718
    ## step = 143000: loss = 0.001593532506376505
    ## step = 143000: Average Return = 23.890138626098633
    ## step = 143500: loss = 0.001666176482103765
    ## step = 144000: loss = 0.0017350866692140698
    ## step = 144000: Average Return = 3.6327123641967773
    ## step = 144500: loss = 0.0028234373312443495
    ## step = 145000: loss = 0.0013936024624854326
    ## step = 145000: Average Return = 24.348657608032227
    ## step = 145500: loss = 0.003336002118885517
    ## step = 146000: loss = 0.0021965554915368557
    ## step = 146000: Average Return = 21.763784408569336
    ## step = 146500: loss = 0.0014535461086779833
    ## step = 147000: loss = 0.0014615445397794247
    ## step = 147000: Average Return = 23.19400405883789
    ## step = 147500: loss = 0.002070794813334942
    ## step = 148000: loss = 0.00212331535294652
    ## step = 148000: Average Return = 23.907472610473633
    ## step = 148500: loss = 0.00222582183778286
    ## step = 149000: loss = 0.0014638301217928529
    ## step = 149000: Average Return = 24.76636505126953
    ## step = 149500: loss = 0.0017339731566607952
    ## step = 150000: loss = 0.0034603741951286793
    ## step = 150000: Average Return = 21.818960189819336
    ## step = 150500: loss = 0.0016914079897105694
    ## step = 151000: loss = 0.0020620212890207767
    ## step = 151000: Average Return = 12.499878883361816
    ## step = 151500: loss = 0.0017451737076044083
    ## step = 152000: loss = 0.0011207468342036009
    ## step = 152000: Average Return = 12.499878883361816
    ## step = 152500: loss = 0.001201326958835125
    ## step = 153000: loss = 0.001349169760942459
    ## step = 153000: Average Return = 24.464210510253906
    ## step = 153500: loss = 0.06012139469385147
    ## step = 154000: loss = 0.002921522594988346
    ## step = 154000: Average Return = 8.374741554260254
    ## step = 154500: loss = 0.0014553777873516083
    ## step = 155000: loss = 0.12990161776542664
    ## step = 155000: Average Return = 22.78116226196289
    ## step = 155500: loss = 0.001768162939697504
    ## step = 156000: loss = 0.010235851630568504
    ## step = 156000: Average Return = 22.925376892089844
    ## step = 156500: loss = 0.0036828271113336086
    ## step = 157000: loss = 0.0009424451855011284
    ## step = 157000: Average Return = 23.9689884185791
    ## step = 157500: loss = 0.12624122202396393
    ## step = 158000: loss = 0.003787602297961712
    ## step = 158000: Average Return = 8.374741554260254
    ## step = 158500: loss = 0.0012279785005375743
    ## step = 159000: loss = 0.005035948473960161
    ## step = 159000: Average Return = 0.05000000819563866
    ## step = 159500: loss = 0.0017544608563184738
    ## step = 160000: loss = 0.001276625320315361
    ## step = 160000: Average Return = 0.05000000819563866
    ## step = 160500: loss = 0.0026733537670224905
    ## step = 161000: loss = 0.0018760606180876493
    ## step = 161000: Average Return = 24.751462936401367
    ## step = 161500: loss = 0.0022215750068426132
    ## step = 162000: loss = 0.0015457123517990112
    ## step = 162000: Average Return = 23.881101608276367
    ## step = 162500: loss = 0.0036647252272814512
    ## step = 163000: loss = 0.0027172775007784367
    ## step = 163000: Average Return = 0.05000000819563866
    ## step = 163500: loss = 0.0010625631548464298
    ## step = 164000: loss = 0.001604217803105712
    ## step = 164000: Average Return = 24.020971298217773
    ## step = 164500: loss = 0.00254416442476213
    ## step = 165000: loss = 0.00234542740508914
    ## step = 165000: Average Return = 24.801494598388672
    ## step = 165500: loss = 0.0015240770298987627
    ## step = 166000: loss = 0.0010399322491139174
    ## step = 166000: Average Return = 24.672685623168945
    ## step = 166500: loss = 0.0019482786301523447
    ## step = 167000: loss = 0.001480692415498197
    ## step = 167000: Average Return = 23.77857208251953
    ## step = 167500: loss = 0.002701353747397661
    ## step = 168000: loss = 0.001258020754903555
    ## step = 168000: Average Return = 24.600732803344727
    ## step = 168500: loss = 0.0024581649340689182
    ## step = 169000: loss = 0.002413245150819421
    ## step = 169000: Average Return = 3.609834909439087
    ## step = 169500: loss = 0.002204182092100382
    ## step = 170000: loss = 0.11487281322479248
    ## step = 170000: Average Return = 19.06114959716797
    ## step = 170500: loss = 0.004222566727548838
    ## step = 171000: loss = 0.0023671183735132217
    ## step = 171000: Average Return = 2.2665109634399414
    ## step = 171500: loss = 0.0013719217386096716
    ## step = 172000: loss = 0.0016447152011096478
    ## step = 172000: Average Return = 12.499878883361816
    ## step = 172500: loss = 0.0015501469606533647
    ## step = 173000: loss = 0.0019972571171820164
    ## step = 173000: Average Return = 5.45211935043335
    ## step = 173500: loss = 0.002576058730483055
    ## step = 174000: loss = 0.0015407395549118519
    ## step = 174000: Average Return = 23.518224716186523
    ## step = 174500: loss = 0.0011119106784462929
    ## step = 175000: loss = 0.001127536641433835
    ## step = 175000: Average Return = 11.85009479522705
    ## step = 175500: loss = 0.002150424523279071
    ## step = 176000: loss = 0.0012533459812402725
    ## step = 176000: Average Return = 0.05000000819563866
    ## step = 176500: loss = 0.0014263923512771726
    ## step = 177000: loss = 0.00530939269810915
    ## step = 177000: Average Return = 24.657617568969727
    ## step = 177500: loss = 0.0019776923581957817
    ## step = 178000: loss = 0.0020992718636989594
    ## step = 178000: Average Return = 23.16164779663086
    ## step = 178500: loss = 0.0013499849010258913
    ## step = 179000: loss = 0.0017918956000357866
    ## step = 179000: Average Return = 24.587970733642578
    ## step = 179500: loss = 0.000783001771196723
    ## step = 180000: loss = 0.0019216248765587807
    ## step = 180000: Average Return = 20.8128604888916
    ## step = 180500: loss = 0.0026488183066248894
    ## step = 181000: loss = 0.0017163543961942196
    ## step = 181000: Average Return = 3.609834909439087
    ## step = 181500: loss = 0.0017749075777828693
    ## step = 182000: loss = 0.001127393334172666
    ## step = 182000: Average Return = 9.643050193786621
    ## step = 182500: loss = 0.002401479287073016
    ## step = 183000: loss = 0.013057274743914604
    ## step = 183000: Average Return = 24.302085876464844
    ## step = 183500: loss = 0.0011202181922271848
    ## step = 184000: loss = 0.00199119676835835
    ## step = 184000: Average Return = 12.499878883361816
    ## step = 184500: loss = 0.002167629776522517
    ## step = 185000: loss = 0.00334335258230567
    ## step = 185000: Average Return = 23.795923233032227
    ## step = 185500: loss = 0.0015002465806901455
    ## step = 186000: loss = 0.0009076636633835733
    ## step = 186000: Average Return = 24.12253189086914
    ## step = 186500: loss = 0.0024910862557590008
    ## step = 187000: loss = 0.001185152679681778
    ## step = 187000: Average Return = 12.499878883361816
    ## step = 187500: loss = 0.11247730255126953
    ## step = 188000: loss = 0.03999009355902672
    ## step = 188000: Average Return = 24.231691360473633
    ## step = 188500: loss = 0.0023495859932154417
    ## step = 189000: loss = 0.0012150041293352842
    ## step = 189000: Average Return = 4.820977210998535
    ## step = 189500: loss = 0.0015169491525739431
    ## step = 190000: loss = 0.0014286770019680262
    ## step = 190000: Average Return = 23.164897918701172
    ## step = 190500: loss = 0.0021451907232403755
    ## step = 191000: loss = 0.0013144081458449364
    ## step = 191000: Average Return = 6.033745765686035
    ## step = 191500: loss = 0.0011917767114937305
    ## step = 192000: loss = 0.11634081602096558
    ## step = 192000: Average Return = 4.820977210998535
    ## step = 192500: loss = 0.0017537525855004787
    ## step = 193000: loss = 0.003804608481004834
    ## step = 193000: Average Return = 21.220243453979492
    ## step = 193500: loss = 0.0034137361217290163
    ## step = 194000: loss = 0.001838475582189858
    ## step = 194000: Average Return = 13.926555633544922
    ## step = 194500: loss = 0.0008581238216720521
    ## step = 195000: loss = 0.0071532088331878185
    ## step = 195000: Average Return = 22.248506546020508
    ## step = 195500: loss = 0.00160334596876055
    ## step = 196000: loss = 0.005312785040587187
    ## step = 196000: Average Return = 23.121686935424805
    ## step = 196500: loss = 0.004101564176380634
    ## step = 197000: loss = 0.0028868913650512695
    ## step = 197000: Average Return = 22.970701217651367
    ## step = 197500: loss = 0.001310832565650344
    ## step = 198000: loss = 0.002464821096509695
    ## step = 198000: Average Return = 23.411027908325195
    ## step = 198500: loss = 0.0018913637613877654
    ## step = 199000: loss = 0.0016743526794016361
    ## step = 199000: Average Return = 22.64868927001953
    ## step = 199500: loss = 0.0013189068995416164
    ## step = 200000: loss = 0.0007211819174699485
    ## step = 200000: Average Return = 23.065916061401367

## Visualization

### Plots

``` python
out = simulate(eval_env, agent.policy)
```

``` python
plt.plot(out[:,1])
```

    ## [<matplotlib.lines.Line2D object at 0x7f4718617cf8>]

``` python
plt.ylabel('state')
```

    ## Text(0, 0.5, 'state')

``` python
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-30-1.png" width="672" />

``` python
plt.plot(out[:,2])
```

    ## [<matplotlib.lines.Line2D object at 0x7f47185e1278>]

``` python
plt.ylabel('action')
```

    ## Text(0, 0.5, 'action')

``` python
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-30-2.png" width="672" />

``` python
plt.plot(out[:,3])
```

    ## [<matplotlib.lines.Line2D object at 0x7f47185a75c0>]

``` python
plt.ylabel('reward')
```

    ## Text(0, 0.5, 'reward')

``` python
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-30-3.png" width="672" />

Use `matplotlib.pyplot` to chart how the policy improved during
training.

One iteration of `fishing-v0` consists of 1000 time steps. MSY harvest
is `rK/4` = `0.1 * 1 / 4`, so the optimal return for one episode is
close to 25.

``` python
#@test {"skip": true}

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
```

    ## [<matplotlib.lines.Line2D object at 0x7f4718636198>]

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

    ## (-1.1960155019536616, 25.0)

``` python
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-31-1.png" width="672" />

``` python
plt.savefig('foo.png', bbox_inches='tight')
```
