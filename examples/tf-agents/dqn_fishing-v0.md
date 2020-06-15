

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

    ## '2.2.0'

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

    ## TimeStep(step_type=array(1, dtype=int32), reward=array(0.015, dtype=float32), discount=array(0.99, dtype=float32), observation=array([0.7513822], dtype=float32))

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

    ## 1.5747936

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

Let’s take a
    look:

``` python
x_i = iter(replay_buffer.as_dataset()).next()
```

    ## WARNING:tensorflow:AutoGraph could not transform <function TFUniformReplayBuffer._as_dataset.<locals>.get_next at 0x7f31e006f2f0> and will run it as-is.
    ## Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    ## Cause: Bad argument number for Name: 4, expecting 3
    ## To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

``` python
x_i
```

    ## (Trajectory(step_type=<tf.Tensor: shape=(), dtype=int32, numpy=1>, observation=<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.7995297], dtype=float32)>, action=<tf.Tensor: shape=(), dtype=int64, numpy=2>, policy_info=(), next_step_type=<tf.Tensor: shape=(), dtype=int32, numpy=1>, reward=<tf.Tensor: shape=(), dtype=float32, numpy=0.017232463>, discount=<tf.Tensor: shape=(), dtype=float32, numpy=0.99>), BufferInfo(ids=<tf.Tensor: shape=(), dtype=int64, numpy=34>, probabilities=<tf.Tensor: shape=(), dtype=float32, numpy=0.01>))

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

    ## 0.7995297

``` python
x_i[0].action.numpy()
```

    ## 2

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

    ## WARNING:tensorflow:AutoGraph could not transform <function TFUniformReplayBuffer._as_dataset.<locals>.get_next at 0x7f31e006f2f0> and will run it as-is.
    ## Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    ## Cause: Bad argument number for Name: 4, expecting 3
    ## To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

``` python
iterator = iter(dataset)
experience, unused_info = next(iterator)
experience.action[:,1]
```

    ## <tf.Tensor: shape=(64,), dtype=int64, numpy=
    ## array([2, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 0,
    ##        1, 0, 0, 1, 0, 0, 2, 2, 1, 2, 1, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 0,
    ##        1, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 2])>

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

    ## [12.499879]

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

    ## step = 500: loss = 6.960943574085832e-05
    ## step = 1000: loss = 0.0003346058656461537
    ## step = 1000: Average Return = 1.7991993427276611
    ## step = 1500: loss = 0.00026385701494291425
    ## step = 2000: loss = 0.0004057092301081866
    ## step = 2000: Average Return = 12.499878883361816
    ## step = 2500: loss = 0.00039787578862160444
    ## step = 3000: loss = 0.0005966126918792725
    ## step = 3000: Average Return = 17.967788696289062
    ## step = 3500: loss = 0.00021358413505367935
    ## step = 4000: loss = 0.01260442566126585
    ## step = 4000: Average Return = 12.499878883361816
    ## step = 4500: loss = 0.0006850471254438162
    ## step = 5000: loss = 0.003131887875497341
    ## step = 5000: Average Return = 1.5622117519378662
    ## step = 5500: loss = 0.0632232278585434
    ## step = 6000: loss = 0.0015548327937722206
    ## step = 6000: Average Return = 14.168335914611816
    ## step = 6500: loss = 0.0397910475730896
    ## step = 7000: loss = 0.002429888118058443
    ## step = 7000: Average Return = 21.895418167114258
    ## step = 7500: loss = 0.0036471737548708916
    ## step = 8000: loss = 0.00342706311494112
    ## step = 8000: Average Return = 0.05000000819563866
    ## step = 8500: loss = 0.002731693210080266
    ## step = 9000: loss = 0.031400009989738464
    ## step = 9000: Average Return = 24.550920486450195
    ## step = 9500: loss = 0.003345740493386984
    ## step = 10000: loss = 0.0025665219873189926
    ## step = 10000: Average Return = 24.517189025878906
    ## step = 10500: loss = 0.035212621092796326
    ## step = 11000: loss = 0.04573190212249756
    ## step = 11000: Average Return = 24.353254318237305
    ## step = 11500: loss = 0.003191737923771143
    ## step = 12000: loss = 0.0018810254987329245
    ## step = 12000: Average Return = 24.027814865112305
    ## step = 12500: loss = 0.012322998605668545
    ## step = 13000: loss = 0.003426768584176898
    ## step = 13000: Average Return = 24.494970321655273
    ## step = 13500: loss = 0.06176411733031273
    ## step = 14000: loss = 0.0029339324682950974
    ## step = 14000: Average Return = 24.129823684692383
    ## step = 14500: loss = 0.08277386426925659
    ## step = 15000: loss = 0.0017064164858311415
    ## step = 15000: Average Return = 22.0033016204834
    ## step = 15500: loss = 0.0023202761076390743
    ## step = 16000: loss = 0.04066009819507599
    ## step = 16000: Average Return = 23.704544067382812
    ## step = 16500: loss = 0.002448881044983864
    ## step = 17000: loss = 0.002002434805035591
    ## step = 17000: Average Return = 0.05000000819563866
    ## step = 17500: loss = 0.0019301209831610322
    ## step = 18000: loss = 0.003434403333812952
    ## step = 18000: Average Return = 24.510595321655273
    ## step = 18500: loss = 0.0017949139000847936
    ## step = 19000: loss = 0.0025183542165905237
    ## step = 19000: Average Return = 23.18878173828125
    ## step = 19500: loss = 0.0009860062273219228
    ## step = 20000: loss = 0.001765379449352622
    ## step = 20000: Average Return = 12.499878883361816
    ## step = 20500: loss = 0.0027524675242602825
    ## step = 21000: loss = 0.0016532223671674728
    ## step = 21000: Average Return = 23.049707412719727
    ## step = 21500: loss = 0.001883740071207285
    ## step = 22000: loss = 0.0016414348501712084
    ## step = 22000: Average Return = 23.8951473236084
    ## step = 22500: loss = 0.001800329890102148
    ## step = 23000: loss = 0.001396878738887608
    ## step = 23000: Average Return = 23.232694625854492
    ## step = 23500: loss = 0.040137771517038345
    ## step = 24000: loss = 0.001473290496505797
    ## step = 24000: Average Return = 24.395160675048828
    ## step = 24500: loss = 0.001268986496143043
    ## step = 25000: loss = 0.0025312365032732487
    ## step = 25000: Average Return = 24.362014770507812
    ## step = 25500: loss = 0.0018823022255674005
    ## step = 26000: loss = 0.0015599466860294342
    ## step = 26000: Average Return = 8.654509544372559
    ## step = 26500: loss = 0.001294163055717945
    ## step = 27000: loss = 0.00189693842548877
    ## step = 27000: Average Return = 12.499878883361816
    ## step = 27500: loss = 0.11706840246915817
    ## step = 28000: loss = 0.0012503082398325205
    ## step = 28000: Average Return = 24.1502628326416
    ## step = 28500: loss = 0.0012833056971430779
    ## step = 29000: loss = 0.0029122927226126194
    ## step = 29000: Average Return = 24.1785831451416
    ## step = 29500: loss = 0.0013046406675130129
    ## step = 30000: loss = 0.0017146688187494874
    ## step = 30000: Average Return = 22.878767013549805
    ## step = 30500: loss = 0.0015373651403933764
    ## step = 31000: loss = 0.0009326132712885737
    ## step = 31000: Average Return = 24.198389053344727
    ## step = 31500: loss = 0.0016149968141689897
    ## step = 32000: loss = 0.0015824781730771065
    ## step = 32000: Average Return = 24.19631576538086
    ## step = 32500: loss = 0.002079019322991371
    ## step = 33000: loss = 0.0014705507783219218
    ## step = 33000: Average Return = 24.14284896850586
    ## step = 33500: loss = 0.0033984901383519173
    ## step = 34000: loss = 0.0012816519010812044
    ## step = 34000: Average Return = 23.414207458496094
    ## step = 34500: loss = 0.002221539383754134
    ## step = 35000: loss = 0.001690099947154522
    ## step = 35000: Average Return = 12.499878883361816
    ## step = 35500: loss = 0.002240530215203762
    ## step = 36000: loss = 0.0024067454505711794
    ## step = 36000: Average Return = 24.285839080810547
    ## step = 36500: loss = 0.0012827485334128141
    ## step = 37000: loss = 0.0023016375489532948
    ## step = 37000: Average Return = 12.499878883361816
    ## step = 37500: loss = 0.0020799331832677126
    ## step = 38000: loss = 0.004822168964892626
    ## step = 38000: Average Return = 23.153797149658203
    ## step = 38500: loss = 0.0022917226888239384
    ## step = 39000: loss = 0.002376849763095379
    ## step = 39000: Average Return = 23.138591766357422
    ## step = 39500: loss = 0.0029322123154997826
    ## step = 40000: loss = 0.002222490729764104
    ## step = 40000: Average Return = 22.417917251586914
    ## step = 40500: loss = 0.00231295102275908
    ## step = 41000: loss = 0.0014212329406291246
    ## step = 41000: Average Return = 23.740114212036133
    ## step = 41500: loss = 0.0013603608822450042
    ## step = 42000: loss = 0.001885314006358385
    ## step = 42000: Average Return = 15.3951997756958
    ## step = 42500: loss = 0.0016994556644931436
    ## step = 43000: loss = 0.0019472715212032199
    ## step = 43000: Average Return = 23.442243576049805
    ## step = 43500: loss = 0.0033058275002986193
    ## step = 44000: loss = 0.0022192979231476784
    ## step = 44000: Average Return = 24.27454376220703
    ## step = 44500: loss = 0.0020815161988139153
    ## step = 45000: loss = 0.0025033652782440186
    ## step = 45000: Average Return = 23.963003158569336
    ## step = 45500: loss = 0.0016913069412112236
    ## step = 46000: loss = 0.0018869752530008554
    ## step = 46000: Average Return = 24.295879364013672
    ## step = 46500: loss = 0.0032983445562422276
    ## step = 47000: loss = 0.0024603777565062046
    ## step = 47000: Average Return = 23.259218215942383
    ## step = 47500: loss = 0.002715588081628084
    ## step = 48000: loss = 0.00186703831423074
    ## step = 48000: Average Return = 12.717358589172363
    ## step = 48500: loss = 0.002383404178544879
    ## step = 49000: loss = 0.001482099061831832
    ## step = 49000: Average Return = 24.321640014648438
    ## step = 49500: loss = 0.0025202271062880754
    ## step = 50000: loss = 0.0711849182844162
    ## step = 50000: Average Return = 21.645179748535156
    ## step = 50500: loss = 0.004564264789223671
    ## step = 51000: loss = 0.05127006024122238
    ## step = 51000: Average Return = 24.57032012939453
    ## step = 51500: loss = 0.0049144430086016655
    ## step = 52000: loss = 0.00153950450476259
    ## step = 52000: Average Return = 24.072912216186523
    ## step = 52500: loss = 0.0023052264004945755
    ## step = 53000: loss = 0.002431764965876937
    ## step = 53000: Average Return = 23.62970542907715
    ## step = 53500: loss = 0.004394222982227802
    ## step = 54000: loss = 0.00232695322483778
    ## step = 54000: Average Return = 23.307268142700195
    ## step = 54500: loss = 0.001633591833524406
    ## step = 55000: loss = 0.0038477894850075245
    ## step = 55000: Average Return = 23.67131996154785
    ## step = 55500: loss = 0.0020371088758111
    ## step = 56000: loss = 0.002989820670336485
    ## step = 56000: Average Return = 24.314468383789062
    ## step = 56500: loss = 0.0020031388849020004
    ## step = 57000: loss = 0.001899197930470109
    ## step = 57000: Average Return = 24.309255599975586
    ## step = 57500: loss = 0.0036532417871057987
    ## step = 58000: loss = 0.0012085451744496822
    ## step = 58000: Average Return = 12.349228858947754
    ## step = 58500: loss = 0.0018590829567983747
    ## step = 59000: loss = 0.0014581666328012943
    ## step = 59000: Average Return = 23.941791534423828
    ## step = 59500: loss = 0.0026398920454084873
    ## step = 60000: loss = 0.0026768757961690426
    ## step = 60000: Average Return = 14.782742500305176
    ## step = 60500: loss = 0.0018407958559691906
    ## step = 61000: loss = 0.002269869204610586
    ## step = 61000: Average Return = 23.481752395629883
    ## step = 61500: loss = 0.0027964101172983646
    ## step = 62000: loss = 0.00210211961530149
    ## step = 62000: Average Return = 24.32384490966797
    ## step = 62500: loss = 0.0022723390720784664
    ## step = 63000: loss = 0.0017205774784088135
    ## step = 63000: Average Return = 23.8562068939209
    ## step = 63500: loss = 0.002978093223646283
    ## step = 64000: loss = 0.0020317467860877514
    ## step = 64000: Average Return = 23.240659713745117
    ## step = 64500: loss = 0.0029556602239608765
    ## step = 65000: loss = 0.0016632538754492998
    ## step = 65000: Average Return = 23.219022750854492
    ## step = 65500: loss = 0.0022328412160277367
    ## step = 66000: loss = 0.001849478343501687
    ## step = 66000: Average Return = 24.412734985351562
    ## step = 66500: loss = 0.002564862137660384
    ## step = 67000: loss = 0.001818111166357994
    ## step = 67000: Average Return = 24.003271102905273
    ## step = 67500: loss = 0.0019572610035538673
    ## step = 68000: loss = 0.002366637112572789
    ## step = 68000: Average Return = 21.375593185424805
    ## step = 68500: loss = 0.002700387267395854
    ## step = 69000: loss = 0.0017344595398753881
    ## step = 69000: Average Return = 24.3016300201416
    ## step = 69500: loss = 0.0017578080296516418
    ## step = 70000: loss = 0.0021952176466584206
    ## step = 70000: Average Return = 22.397258758544922
    ## step = 70500: loss = 0.0022558935452252626
    ## step = 71000: loss = 0.0019254365470260382
    ## step = 71000: Average Return = 24.418638229370117
    ## step = 71500: loss = 0.0025878348387777805
    ## step = 72000: loss = 0.0015328079462051392
    ## step = 72000: Average Return = 24.199087142944336
    ## step = 72500: loss = 0.04199357330799103
    ## step = 73000: loss = 0.0028852333780378103
    ## step = 73000: Average Return = 16.029579162597656
    ## step = 73500: loss = 0.0012658365303650498
    ## step = 74000: loss = 0.002347125206142664
    ## step = 74000: Average Return = 11.0618314743042
    ## step = 74500: loss = 0.00241171196103096
    ## step = 75000: loss = 0.0018654624000191689
    ## step = 75000: Average Return = 8.124253273010254
    ## step = 75500: loss = 0.0028662923723459244
    ## step = 76000: loss = 0.002271117642521858
    ## step = 76000: Average Return = 24.040447235107422
    ## step = 76500: loss = 0.0026968445163220167
    ## step = 77000: loss = 0.05532240495085716
    ## step = 77000: Average Return = 12.499878883361816
    ## step = 77500: loss = 0.0013840701431035995
    ## step = 78000: loss = 0.0018577890004962683
    ## step = 78000: Average Return = 24.21249771118164
    ## step = 78500: loss = 0.0018713331082835793
    ## step = 79000: loss = 0.004011246375739574
    ## step = 79000: Average Return = 2.450878858566284
    ## step = 79500: loss = 0.002720488468185067
    ## step = 80000: loss = 0.00203898036852479
    ## step = 80000: Average Return = 24.57616424560547
    ## step = 80500: loss = 0.002380420221015811
    ## step = 81000: loss = 0.0010486366227269173
    ## step = 81000: Average Return = 24.51650619506836
    ## step = 81500: loss = 0.0017046260181814432
    ## step = 82000: loss = 0.002336360514163971
    ## step = 82000: Average Return = 12.499878883361816
    ## step = 82500: loss = 0.0019417835865169764
    ## step = 83000: loss = 0.023561744019389153
    ## step = 83000: Average Return = 24.12091064453125
    ## step = 83500: loss = 0.0030159049201756716
    ## step = 84000: loss = 0.0035291267558932304
    ## step = 84000: Average Return = 12.499878883361816
    ## step = 84500: loss = 0.0030847396701574326
    ## step = 85000: loss = 0.002077337121590972
    ## step = 85000: Average Return = 0.05643290653824806
    ## step = 85500: loss = 0.002636495279148221
    ## step = 86000: loss = 0.0016988138668239117
    ## step = 86000: Average Return = 23.470226287841797
    ## step = 86500: loss = 0.0021175395231693983
    ## step = 87000: loss = 0.002725707832723856
    ## step = 87000: Average Return = 23.822118759155273
    ## step = 87500: loss = 0.002816966036334634
    ## step = 88000: loss = 0.0019313242519274354
    ## step = 88000: Average Return = 24.495525360107422
    ## step = 88500: loss = 0.0032312264665961266
    ## step = 89000: loss = 0.0028234804049134254
    ## step = 89000: Average Return = 9.30917739868164
    ## step = 89500: loss = 0.002474929206073284
    ## step = 90000: loss = 0.0017532119527459145
    ## step = 90000: Average Return = 24.351276397705078
    ## step = 90500: loss = 0.002045935718342662
    ## step = 91000: loss = 0.0032735918648540974
    ## step = 91000: Average Return = 19.37778091430664
    ## step = 91500: loss = 0.0031389354262501
    ## step = 92000: loss = 0.002123619429767132
    ## step = 92000: Average Return = 22.923051834106445
    ## step = 92500: loss = 0.00242174556478858
    ## step = 93000: loss = 0.0024287693668156862
    ## step = 93000: Average Return = 23.336633682250977
    ## step = 93500: loss = 0.0019354848191142082
    ## step = 94000: loss = 0.0020452013704925776
    ## step = 94000: Average Return = 22.84050750732422
    ## step = 94500: loss = 0.004118088632822037
    ## step = 95000: loss = 0.0020050734747201204
    ## step = 95000: Average Return = 23.317855834960938
    ## step = 95500: loss = 0.0021844396833330393
    ## step = 96000: loss = 0.0013830284588038921
    ## step = 96000: Average Return = 23.446754455566406
    ## step = 96500: loss = 0.00297419261187315
    ## step = 97000: loss = 0.0018307535210624337
    ## step = 97000: Average Return = 12.499878883361816
    ## step = 97500: loss = 0.00250423070974648
    ## step = 98000: loss = 0.001597237423993647
    ## step = 98000: Average Return = 24.124948501586914
    ## step = 98500: loss = 0.0021255265455693007
    ## step = 99000: loss = 0.001945774769410491
    ## step = 99000: Average Return = 12.499878883361816
    ## step = 99500: loss = 0.0019103139638900757
    ## step = 100000: loss = 0.0021419133991003036
    ## step = 100000: Average Return = 24.0690975189209
    ## step = 100500: loss = 0.0012633863370865583
    ## step = 101000: loss = 0.00138920359313488
    ## step = 101000: Average Return = 24.256633758544922
    ## step = 101500: loss = 0.0022368882782757282
    ## step = 102000: loss = 0.002301693195477128
    ## step = 102000: Average Return = 12.499878883361816
    ## step = 102500: loss = 0.0026783663779497147
    ## step = 103000: loss = 0.00282998476177454
    ## step = 103000: Average Return = 24.395505905151367
    ## step = 103500: loss = 0.0022046566009521484
    ## step = 104000: loss = 0.0026974251959472895
    ## step = 104000: Average Return = 24.58257484436035
    ## step = 104500: loss = 0.0024766610004007816
    ## step = 105000: loss = 0.0021306562703102827
    ## step = 105000: Average Return = 23.335886001586914
    ## step = 105500: loss = 0.004375539254397154
    ## step = 106000: loss = 0.003945633303374052
    ## step = 106000: Average Return = 24.557415008544922
    ## step = 106500: loss = 0.0018900242866948247
    ## step = 107000: loss = 0.002511614467948675
    ## step = 107000: Average Return = 22.95953941345215
    ## step = 107500: loss = 0.0018358853412792087
    ## step = 108000: loss = 0.002706239465624094
    ## step = 108000: Average Return = 24.426759719848633
    ## step = 108500: loss = 0.0020809147972613573
    ## step = 109000: loss = 0.0027761058881878853
    ## step = 109000: Average Return = 12.499878883361816
    ## step = 109500: loss = 0.05125170946121216
    ## step = 110000: loss = 0.11899405717849731
    ## step = 110000: Average Return = 23.400163650512695
    ## step = 110500: loss = 0.0027095372788608074
    ## step = 111000: loss = 0.002594682853668928
    ## step = 111000: Average Return = 24.281631469726562
    ## step = 111500: loss = 0.0022172448225319386
    ## step = 112000: loss = 0.0016598895890638232
    ## step = 112000: Average Return = 23.873109817504883
    ## step = 112500: loss = 0.0022938461042940617
    ## step = 113000: loss = 0.002217374974861741
    ## step = 113000: Average Return = 23.15121841430664
    ## step = 113500: loss = 0.0029563927091658115
    ## step = 114000: loss = 0.0019659423269331455
    ## step = 114000: Average Return = 22.43023109436035
    ## step = 114500: loss = 0.0029522371478378773
    ## step = 115000: loss = 0.0035853739827871323
    ## step = 115000: Average Return = 17.173297882080078
    ## step = 115500: loss = 0.00393294170498848
    ## step = 116000: loss = 0.1156022921204567
    ## step = 116000: Average Return = 12.499878883361816
    ## step = 116500: loss = 0.001629287376999855
    ## step = 117000: loss = 0.0030465025920420885
    ## step = 117000: Average Return = 12.499878883361816
    ## step = 117500: loss = 0.0017558397958055139
    ## step = 118000: loss = 0.0015611447161063552
    ## step = 118000: Average Return = 23.89456558227539
    ## step = 118500: loss = 0.003923683427274227
    ## step = 119000: loss = 0.0019226272124797106
    ## step = 119000: Average Return = 24.359437942504883
    ## step = 119500: loss = 0.0020535639487206936
    ## step = 120000: loss = 0.0026524567510932684
    ## step = 120000: Average Return = 24.417627334594727
    ## step = 120500: loss = 0.09009591490030289
    ## step = 121000: loss = 0.002195111010223627
    ## step = 121000: Average Return = 24.54535675048828
    ## step = 121500: loss = 0.02482246235013008
    ## step = 122000: loss = 0.00431407755240798
    ## step = 122000: Average Return = 24.603713989257812
    ## step = 122500: loss = 0.003379952162504196
    ## step = 123000: loss = 0.001979199703782797
    ## step = 123000: Average Return = 12.499878883361816
    ## step = 123500: loss = 0.002169213257730007
    ## step = 124000: loss = 0.0017541679553687572
    ## step = 124000: Average Return = 24.4118595123291
    ## step = 124500: loss = 0.0022501908242702484
    ## step = 125000: loss = 0.0024668555706739426
    ## step = 125000: Average Return = 23.52536392211914
    ## step = 125500: loss = 0.0024187928065657616
    ## step = 126000: loss = 0.03184082359075546
    ## step = 126000: Average Return = 24.0469913482666
    ## step = 126500: loss = 0.0019840397872030735
    ## step = 127000: loss = 0.002440035343170166
    ## step = 127000: Average Return = 20.736865997314453
    ## step = 127500: loss = 0.0018023555167019367
    ## step = 128000: loss = 0.09810958802700043
    ## step = 128000: Average Return = 23.77143096923828
    ## step = 128500: loss = 0.0023196241818368435
    ## step = 129000: loss = 0.0033128124196082354
    ## step = 129000: Average Return = 22.955490112304688
    ## step = 129500: loss = 0.002059041988104582
    ## step = 130000: loss = 0.001666074967943132
    ## step = 130000: Average Return = 23.443986892700195
    ## step = 130500: loss = 0.0018736697966232896
    ## step = 131000: loss = 0.0017094267532229424
    ## step = 131000: Average Return = 24.08051109313965
    ## step = 131500: loss = 0.001860095770098269
    ## step = 132000: loss = 0.0018689847784116864
    ## step = 132000: Average Return = 24.092418670654297
    ## step = 132500: loss = 0.0027925893664360046
    ## step = 133000: loss = 0.002503241878002882
    ## step = 133000: Average Return = 22.616418838500977
    ## step = 133500: loss = 0.0016334839165210724
    ## step = 134000: loss = 0.002385316649451852
    ## step = 134000: Average Return = 23.8905086517334
    ## step = 134500: loss = 0.003056012559682131
    ## step = 135000: loss = 0.001664512325078249
    ## step = 135000: Average Return = 24.173776626586914
    ## step = 135500: loss = 0.0016157447826117277
    ## step = 136000: loss = 0.00256741838529706
    ## step = 136000: Average Return = 21.138229370117188
    ## step = 136500: loss = 0.003459764178842306
    ## step = 137000: loss = 0.002324303612112999
    ## step = 137000: Average Return = 23.998424530029297
    ## step = 137500: loss = 0.0029305648058652878
    ## step = 138000: loss = 0.002868757350370288
    ## step = 138000: Average Return = 1.274078130722046
    ## step = 138500: loss = 0.0021460738498717546
    ## step = 139000: loss = 0.0037967583630234003
    ## step = 139000: Average Return = 23.652515411376953
    ## step = 139500: loss = 0.002066377317532897
    ## step = 140000: loss = 0.001557470066472888
    ## step = 140000: Average Return = 24.343852996826172
    ## step = 140500: loss = 0.002140594646334648
    ## step = 141000: loss = 0.005599288735538721
    ## step = 141000: Average Return = 23.921836853027344
    ## step = 141500: loss = 0.001423787442035973
    ## step = 142000: loss = 0.0024896326940506697
    ## step = 142000: Average Return = 24.126708984375
    ## step = 142500: loss = 0.0011291488772258162
    ## step = 143000: loss = 0.004421811085194349
    ## step = 143000: Average Return = 20.583820343017578
    ## step = 143500: loss = 0.0030754576437175274
    ## step = 144000: loss = 0.0023673302493989468
    ## step = 144000: Average Return = 23.894424438476562
    ## step = 144500: loss = 0.001768697751685977
    ## step = 145000: loss = 0.0017617729026824236
    ## step = 145000: Average Return = 24.061128616333008
    ## step = 145500: loss = 0.003866611048579216
    ## step = 146000: loss = 0.0019588505383580923
    ## step = 146000: Average Return = 24.011796951293945
    ## step = 146500: loss = 0.0014088393654674292
    ## step = 147000: loss = 0.0016749724745750427
    ## step = 147000: Average Return = 22.97748565673828
    ## step = 147500: loss = 0.000897538207937032
    ## step = 148000: loss = 0.0029092938639223576
    ## step = 148000: Average Return = 23.040264129638672
    ## step = 148500: loss = 0.0017148602055385709
    ## step = 149000: loss = 0.002970378380268812
    ## step = 149000: Average Return = 12.499878883361816
    ## step = 149500: loss = 0.004234926775097847
    ## step = 150000: loss = 0.0019949483685195446
    ## step = 150000: Average Return = 22.865705490112305
    ## step = 150500: loss = 0.00228943582624197
    ## step = 151000: loss = 0.0018540716264396906
    ## step = 151000: Average Return = 6.300189018249512
    ## step = 151500: loss = 0.0027697591576725245
    ## step = 152000: loss = 0.0022259254474192858
    ## step = 152000: Average Return = 23.644004821777344
    ## step = 152500: loss = 0.0032178231049329042
    ## step = 153000: loss = 0.0030190791003406048
    ## step = 153000: Average Return = 23.447654724121094
    ## step = 153500: loss = 0.002774879802018404
    ## step = 154000: loss = 0.0015903718303889036
    ## step = 154000: Average Return = 23.545921325683594
    ## step = 154500: loss = 0.0027362708933651447
    ## step = 155000: loss = 0.0020304168574512005
    ## step = 155000: Average Return = 24.293813705444336
    ## step = 155500: loss = 0.0027716662734746933
    ## step = 156000: loss = 0.0015506665222346783
    ## step = 156000: Average Return = 24.562637329101562
    ## step = 156500: loss = 0.002411467023193836
    ## step = 157000: loss = 0.0037283075507730246
    ## step = 157000: Average Return = 12.499878883361816
    ## step = 157500: loss = 0.0396391898393631
    ## step = 158000: loss = 0.022329095751047134
    ## step = 158000: Average Return = 7.697067737579346
    ## step = 158500: loss = 0.0029433881863951683
    ## step = 159000: loss = 0.0023421449586749077
    ## step = 159000: Average Return = 18.816953659057617
    ## step = 159500: loss = 0.0017204124014824629
    ## step = 160000: loss = 0.0034742425195872784
    ## step = 160000: Average Return = 12.499878883361816
    ## step = 160500: loss = 0.0027203315403312445
    ## step = 161000: loss = 0.0012261738302186131
    ## step = 161000: Average Return = 24.378015518188477
    ## step = 161500: loss = 0.003955556079745293
    ## step = 162000: loss = 0.0035003749653697014
    ## step = 162000: Average Return = 23.431547164916992
    ## step = 162500: loss = 0.0017709831008687615
    ## step = 163000: loss = 0.003192152362316847
    ## step = 163000: Average Return = 24.278512954711914
    ## step = 163500: loss = 0.0028073585126549006
    ## step = 164000: loss = 0.00317647703923285
    ## step = 164000: Average Return = 24.331892013549805
    ## step = 164500: loss = 0.002726835198700428
    ## step = 165000: loss = 0.0021442347206175327
    ## step = 165000: Average Return = 19.590251922607422
    ## step = 165500: loss = 0.002447953913360834
    ## step = 166000: loss = 0.0033967618364840746
    ## step = 166000: Average Return = 18.41465187072754
    ## step = 166500: loss = 0.0054232049733400345
    ## step = 167000: loss = 0.002577102743089199
    ## step = 167000: Average Return = 23.454185485839844
    ## step = 167500: loss = 0.0024566352367401123
    ## step = 168000: loss = 0.0026963679119944572
    ## step = 168000: Average Return = 24.412019729614258
    ## step = 168500: loss = 0.0022292647045105696
    ## step = 169000: loss = 0.0027833229396492243
    ## step = 169000: Average Return = 21.779870986938477
    ## step = 169500: loss = 0.003148446325212717
    ## step = 170000: loss = 0.002481854520738125
    ## step = 170000: Average Return = 23.490930557250977
    ## step = 170500: loss = 0.002615545876324177
    ## step = 171000: loss = 0.0008539333357475698
    ## step = 171000: Average Return = 24.39011001586914
    ## step = 171500: loss = 0.0020922960247844458
    ## step = 172000: loss = 0.002076215110719204
    ## step = 172000: Average Return = 24.189369201660156
    ## step = 172500: loss = 0.0013901207130402327
    ## step = 173000: loss = 0.003808391047641635
    ## step = 173000: Average Return = 4.41131591796875
    ## step = 173500: loss = 0.002113030757755041
    ## step = 174000: loss = 0.0019228467717766762
    ## step = 174000: Average Return = 24.03076171875
    ## step = 174500: loss = 0.0022913715802133083
    ## step = 175000: loss = 0.0386987030506134
    ## step = 175000: Average Return = 24.43450164794922
    ## step = 175500: loss = 0.0025567233096808195
    ## step = 176000: loss = 0.002011496340855956
    ## step = 176000: Average Return = 18.003074645996094
    ## step = 176500: loss = 0.00328332232311368
    ## step = 177000: loss = 0.002995830960571766
    ## step = 177000: Average Return = 12.129262924194336
    ## step = 177500: loss = 0.0020443028770387173
    ## step = 178000: loss = 0.002947250148281455
    ## step = 178000: Average Return = 19.055912017822266
    ## step = 178500: loss = 0.001664321986027062
    ## step = 179000: loss = 0.0032067897263914347
    ## step = 179000: Average Return = 23.66114044189453
    ## step = 179500: loss = 0.00376994744874537
    ## step = 180000: loss = 0.043896257877349854
    ## step = 180000: Average Return = 23.0062198638916
    ## step = 180500: loss = 0.0031649149022996426
    ## step = 181000: loss = 0.0055377548560500145
    ## step = 181000: Average Return = 23.61642837524414
    ## step = 181500: loss = 0.0022749314084649086
    ## step = 182000: loss = 0.002518555847927928
    ## step = 182000: Average Return = 23.175756454467773
    ## step = 182500: loss = 0.0025766806211322546
    ## step = 183000: loss = 0.001826119376346469
    ## step = 183000: Average Return = 12.499878883361816
    ## step = 183500: loss = 0.002507748082280159
    ## step = 184000: loss = 0.003408570308238268
    ## step = 184000: Average Return = 23.14629554748535
    ## step = 184500: loss = 0.0025242618285119534
    ## step = 185000: loss = 0.0030317390337586403
    ## step = 185000: Average Return = 22.45562171936035
    ## step = 185500: loss = 0.0032762535847723484
    ## step = 186000: loss = 0.0025391788221895695
    ## step = 186000: Average Return = 21.919946670532227
    ## step = 186500: loss = 0.0024304636754095554
    ## step = 187000: loss = 0.0022409860976040363
    ## step = 187000: Average Return = 24.413677215576172
    ## step = 187500: loss = 0.0036490566562861204
    ## step = 188000: loss = 0.002428510459139943
    ## step = 188000: Average Return = 24.29560089111328
    ## step = 188500: loss = 0.003214897820726037
    ## step = 189000: loss = 0.001507685286924243
    ## step = 189000: Average Return = 22.388914108276367
    ## step = 189500: loss = 0.11967220157384872
    ## step = 190000: loss = 0.0038840484339743853
    ## step = 190000: Average Return = 24.186939239501953
    ## step = 190500: loss = 0.002838718704879284
    ## step = 191000: loss = 0.0015374424401670694
    ## step = 191000: Average Return = 12.499878883361816
    ## step = 191500: loss = 0.036385081708431244
    ## step = 192000: loss = 0.003425517352297902
    ## step = 192000: Average Return = 15.030292510986328
    ## step = 192500: loss = 0.001563523430377245
    ## step = 193000: loss = 0.0025782627053558826
    ## step = 193000: Average Return = 12.499878883361816
    ## step = 193500: loss = 0.0020376406610012054
    ## step = 194000: loss = 0.0018474889220669866
    ## step = 194000: Average Return = 24.15179443359375
    ## step = 194500: loss = 0.003105614334344864
    ## step = 195000: loss = 0.0021034833043813705
    ## step = 195000: Average Return = 23.469032287597656
    ## step = 195500: loss = 0.0022092568688094616
    ## step = 196000: loss = 0.0022078831680119038
    ## step = 196000: Average Return = 23.16680908203125
    ## step = 196500: loss = 0.0023575350642204285
    ## step = 197000: loss = 0.002780814189463854
    ## step = 197000: Average Return = 22.604990005493164
    ## step = 197500: loss = 0.10678347200155258
    ## step = 198000: loss = 0.005033146124333143
    ## step = 198000: Average Return = 24.37579345703125
    ## step = 198500: loss = 0.004543536342680454
    ## step = 199000: loss = 0.0021745634730905294
    ## step = 199000: Average Return = 23.166160583496094
    ## step = 199500: loss = 0.001186963403597474
    ## step = 200000: loss = 0.0020312340930104256
    ## step = 200000: Average Return = 23.40517807006836

## Visualization

### Plots

``` python
out = simulate(eval_env, agent.policy)
```

``` python
plt.plot(out[:,1])
```

    ## [<matplotlib.lines.Line2D object at 0x7f3164682c50>]

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

    ## [<matplotlib.lines.Line2D object at 0x7f316464c4a8>]

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

    ## [<matplotlib.lines.Line2D object at 0x7f3164615748>]

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

    ## [<matplotlib.lines.Line2D object at 0x7f31646a5e48>]

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

    ## (-1.1776856908574702, 25.0)

``` python
plt.show()
```

<img src="dqn_fishing-v0_files/figure-gfm/unnamed-chunk-31-1.png" width="672" />

``` python
plt.savefig('foo.png', bbox_inches='tight')
```
