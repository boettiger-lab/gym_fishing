
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

##import gym_fishing
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

tf.version.VERSION
```

    ## '2.1.0'

``` python
## Hyperparameters
num_iterations = 20000             # @param {type:"integer"}
initial_collect_steps = 1000       # @param {type:"integer"} 
collect_steps_per_iteration = 1    # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 64                    # @param {type:"integer"}
learning_rate = 1e-3               # @param {type:"number"}
log_interval = 200                 # @param {type:"integer"}
num_eval_episodes = 10             # @param {type:"integer"}
eval_interval = 1000               # @param {type:"integer"}

## Environment
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env.reset()
```

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([ 0.02651513, -0.0126955 , -0.03012531, -0.03375835], dtype=float32))

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

    ## BoundedArraySpec(shape=(4,), dtype=dtype('float32'), name='observation', minimum=[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], maximum=[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38])

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

    ## BoundedArraySpec(shape=(), dtype=dtype('int64'), name='action', minimum=0, maximum=1)

In the Cartpole environment:

  - `observation` is an array of 4 floats:
  - the position and velocity of the cart
  - the angular position and velocity of the pole
  - `reward` is a scalar float value
  - `action` is a scalar integer with only two possible values:
  - `0` — “move left”
  - `1` — “move right”

<!-- end list -->

``` python
time_step = env.reset()
print('Time step:')
```

    ## Time step:

``` python
print(time_step)
```

    ## TimeStep(step_type=array(0, dtype=int32), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([ 0.01614731,  0.03317719,  0.00815116, -0.0369195 ], dtype=float32))

``` python
action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
```

    ## Next time step:

``` python
print(next_time_step)
```

    ## TimeStep(step_type=array(1, dtype=int32), reward=array(1., dtype=float32), discount=array(1., dtype=float32), observation=array([ 0.01681085,  0.2281813 ,  0.00741277, -0.32701954], dtype=float32))

Usually two environments are instantiated: one for training and one for
evaluation.

``` python
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
```

The Cartpole environment, like most environments, is written in pure
Python. This is converted to TensorFlow using the `TFPyEnvironment`
wrapper.

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

# Policies can be created independently of agents. For example,

# use `tf_agents.policies.random_tf_policy` to create a policy

# which will randomly select an action for each `time_step`.

``` python
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
```

To get an action from a policy, call the `policy.action(time_step)`
method. The `time_step` contains the observation from the environment.
This method returns a `PolicyStep`, which is a named tuple with three
components:

  - `action` — the action to be taken (in this case, `0` or `1`)
  - `state` — used for stateful (that is, RNN-based) policies
  - `info` — auxiliary data, such as log probabilities of actions

<!-- end list -->

``` python
example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load(env_name))

time_step = example_environment.reset()
random_policy.action(time_step)
```

    ## PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int64, numpy=array([0])>, state=(), info=())

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

    ## 23.3

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

    ## Trajectory(step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), observation=BoundedTensorSpec(shape=(4,), dtype=tf.float32, name='observation', minimum=array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],
    ##       dtype=float32), maximum=array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38],
    ##       dtype=float32)), action=BoundedTensorSpec(shape=(), dtype=tf.int64, name='action', minimum=array(0), maximum=array(1)), policy_info=(), next_step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'), reward=TensorSpec(shape=(), dtype=tf.float32, name='reward'), discount=BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)))

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
The replay buffer is now a collection of Trajectories. \`\`\`

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

    ## <PrefetchDataset shapes: (Trajectory(step_type=(64, 2), observation=(64, 2, 4), action=(64, 2), policy_info=(), next_step_type=(64, 2), reward=(64, 2), discount=(64, 2)), BufferInfo(ids=(64, 2), probabilities=(64,))), types: (Trajectory(step_type=tf.int32, observation=tf.float32, action=tf.int64, policy_info=(), next_step_type=tf.int32, reward=tf.float32, discount=tf.float32), BufferInfo(ids=tf.int64, probabilities=tf.float32))>

``` python
iterator = iter(dataset)

print(iterator)
```

    ## <tensorflow.python.data.ops.iterator_ops.OwnedIterator object at 0x7f8d95aa5b00>

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
current score. The following will take ~5 minutes to
run.

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

    ## step = 200: loss = 11.370645523071289
    ## step = 400: loss = 3.9576056003570557
    ## step = 600: loss = 3.5838136672973633
    ## step = 800: loss = 10.748034477233887
    ## step = 1000: loss = 9.833311080932617
    ## step = 1000: Average Return = 17.299999237060547
    ## step = 1200: loss = 9.130208015441895
    ## step = 1400: loss = 21.797657012939453
    ## step = 1600: loss = 20.00463104248047
    ## step = 1800: loss = 43.34418487548828
    ## step = 2000: loss = 14.112831115722656
    ## step = 2000: Average Return = 44.79999923706055
    ## step = 2200: loss = 12.525673866271973
    ## step = 2400: loss = 22.84939193725586
    ## step = 2600: loss = 10.977899551391602
    ## step = 2800: loss = 55.397621154785156
    ## step = 3000: loss = 70.82565307617188
    ## step = 3000: Average Return = 96.30000305175781
    ## step = 3200: loss = 52.766441345214844
    ## step = 3400: loss = 8.240467071533203
    ## step = 3600: loss = 70.84109497070312
    ## step = 3800: loss = 167.63023376464844
    ## step = 4000: loss = 75.20069122314453
    ## step = 4000: Average Return = 147.5
    ## step = 4200: loss = 151.7181396484375
    ## step = 4400: loss = 55.634910583496094
    ## step = 4600: loss = 66.52213287353516
    ## step = 4800: loss = 254.00601196289062
    ## step = 5000: loss = 57.388023376464844
    ## step = 5000: Average Return = 179.0
    ## step = 5200: loss = 163.36148071289062
    ## step = 5400: loss = 174.94406127929688
    ## step = 5600: loss = 8.123729705810547
    ## step = 5800: loss = 12.239727020263672
    ## step = 6000: loss = 24.866798400878906
    ## step = 6000: Average Return = 165.1999969482422
    ## step = 6200: loss = 159.75485229492188
    ## step = 6400: loss = 95.36395263671875
    ## step = 6600: loss = 167.65089416503906
    ## step = 6800: loss = 102.50139617919922
    ## step = 7000: loss = 493.4981689453125
    ## step = 7000: Average Return = 178.3000030517578
    ## step = 7200: loss = 407.5428771972656
    ## step = 7400: loss = 245.0514678955078
    ## step = 7600: loss = 271.0803527832031
    ## step = 7800: loss = 15.289068222045898
    ## step = 8000: loss = 544.4300537109375
    ## step = 8000: Average Return = 199.89999389648438
    ## step = 8200: loss = 484.2497863769531
    ## step = 8400: loss = 273.6162109375
    ## step = 8600: loss = 107.59264373779297
    ## step = 8800: loss = 346.23138427734375
    ## step = 9000: loss = 24.986705780029297
    ## step = 9000: Average Return = 196.1999969482422
    ## step = 9200: loss = 289.8091125488281
    ## step = 9400: loss = 284.9351806640625
    ## step = 9600: loss = 20.34059715270996
    ## step = 9800: loss = 214.0171661376953
    ## step = 10000: loss = 14.123146057128906
    ## step = 10000: Average Return = 200.0
    ## step = 10200: loss = 16.764244079589844
    ## step = 10400: loss = 86.82066345214844
    ## step = 10600: loss = 833.0789794921875
    ## step = 10800: loss = 23.74228858947754
    ## step = 11000: loss = 680.4890747070312
    ## step = 11000: Average Return = 199.60000610351562
    ## step = 11200: loss = 18.95240592956543
    ## step = 11400: loss = 546.3292236328125
    ## step = 11600: loss = 23.312122344970703
    ## step = 11800: loss = 28.655170440673828
    ## step = 12000: loss = 477.88916015625
    ## step = 12000: Average Return = 200.0
    ## step = 12200: loss = 18.769699096679688
    ## step = 12400: loss = 341.620849609375
    ## step = 12600: loss = 23.182086944580078
    ## step = 12800: loss = 862.5274047851562
    ## step = 13000: loss = 35.15269088745117
    ## step = 13000: Average Return = 200.0
    ## step = 13200: loss = 74.97379302978516
    ## step = 13400: loss = 874.2750854492188
    ## step = 13600: loss = 24.998014450073242
    ## step = 13800: loss = 431.86328125
    ## step = 14000: loss = 1278.161865234375
    ## step = 14000: Average Return = 200.0
    ## step = 14200: loss = 16.576778411865234
    ## step = 14400: loss = 1202.00146484375
    ## step = 14600: loss = 665.1270141601562
    ## step = 14800: loss = 209.93783569335938
    ## step = 15000: loss = 698.7433471679688
    ## step = 15000: Average Return = 200.0
    ## step = 15200: loss = 1577.43701171875
    ## step = 15400: loss = 157.58953857421875
    ## step = 15600: loss = 33.06439208984375
    ## step = 15800: loss = 1479.5399169921875
    ## step = 16000: loss = 671.3092651367188
    ## step = 16000: Average Return = 200.0
    ## step = 16200: loss = 49.79524612426758
    ## step = 16400: loss = 59.117740631103516
    ## step = 16600: loss = 40.34868621826172
    ## step = 16800: loss = 1678.249267578125
    ## step = 17000: loss = 32.240604400634766
    ## step = 17000: Average Return = 197.0
    ## step = 17200: loss = 38.503562927246094
    ## step = 17400: loss = 54.595855712890625
    ## step = 17600: loss = 1032.6385498046875
    ## step = 17800: loss = 96.58526611328125
    ## step = 18000: loss = 39.7320556640625
    ## step = 18000: Average Return = 200.0
    ## step = 18200: loss = 354.571044921875
    ## step = 18400: loss = 51.646942138671875
    ## step = 18600: loss = 43.54009246826172
    ## step = 18800: loss = 1472.5869140625
    ## step = 19000: loss = 671.4873046875
    ## step = 19000: Average Return = 200.0
    ## step = 19200: loss = 43.111289978027344
    ## step = 19400: loss = 69.58483123779297
    ## step = 19600: loss = 28.29844093322754
    ## step = 19800: loss = 30.780902862548828
    ## step = 20000: loss = 69.9782485961914
    ## step = 20000: Average Return = 200.0

## Visualization

``` python
# ### Plots
# 
# Use `matplotlib.pyplot` to chart how the policy improved during training.
# 
# One iteration of `Cartpole-v0` consists of 200 time steps. The environment gives a reward of `+1` for each step the pole stays up, so the maximum return for one episode is 200. The charts shows the return increasing towards that maximum each time it is evaluated during training. (It may be a little unstable and not increase monotonically each time.)

# In[ ]:


#@test {"skip": true}

#iterations = range(0, num_iterations + 1, eval_interval)
#plt.plot(iterations, returns)
#plt.ylabel('Average Return')
#plt.xlabel('Iterations')
#plt.ylim(top=250)
```
