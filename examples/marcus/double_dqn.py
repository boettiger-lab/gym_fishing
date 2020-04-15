import gym
import gym_fishing
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from math import floor
from itertools import count
import argparse

env = gym.make('fishing-v0')

parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, default=100, help='This is the batch size.')
parser.add_argument("-g", type=float, default=1, help='This is gamma or discount factor.')
parser.add_argument('-es', type=float, default=0.99999, help='This is the max of epsilon (starting point).')
parser.add_argument('-ee', type=float, default=0.01, help='This is the min of epsilon (end point).')
parser.add_argument('-ed', type=float, default=1e5, help='This is the decay time of epsilon.')
parser.add_argument('-t', type=int, default=1e2, help="The number of episodes before updating the target network.")
parser.add_argument('-l', type=int, default=1000, help="The width of the neural network")
parser.add_argument('-n', type=int, default=int(1e4), help='The number of total episodes.')
parser.add_argument('-m', type=int, default=1000, help='The length of memory')
parser.add_argument('-T', type=float, default=1000, help='The temperature for Soft Max selection.')
parser.add_argument('-sm', type=int, default=0, help='Flag for softmax activation.')


args = parser.parse_args()

BATCH_SIZE = args.b
SOFT_MAX_FLAG = args.sm
TEMP = args.T
GAMMA = args.g
EPS_START = args.es
EPS_END = args.ee
EPS_DECAY = args.ed
TARGET_UPDATE = args.t
LAYER_WIDTH = args.l
N_ACTIONS = len(env.action_space)
num_episodes = args.n
MEMORY_LENGTH = args.m
steps_done = 0
loss_history = []


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_quotas):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_quotas)
    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        out = self.fc3(out)
        
        return out



policy_net = DQN(1, LAYER_WIDTH, N_ACTIONS).to(device).requires_grad_(requires_grad=True)
target_net = DQN(1, LAYER_WIDTH, N_ACTIONS).to(device).requires_grad_(requires_grad=False)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
memory = ReplayMemory(MEMORY_LENGTH)



def select_action(state):
    global steps_done, SOFT_MAX_FLAG, TEMP
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # So here I find the index of the max Q(S,A)
            # I map the indices of my output to the indices of 
            # action space, so I use this index to access the 
            # corresponding action
            max_index = policy_net(state).max(0)[1]

            return torch.tensor([[env.action_space[max_index]]], device=device, dtype=torch.float)
    # Returning a random action
    elif SOFT_MAX_FLAG:
        "Sample from e^(Q(s,a)) / Z"
        q_values = policy_net(state)
        
        Z = sum([np.exp(float(q_values[i].detach()) / TEMP) for i,_ in enumerate(env.action_space)])
        dist = {}
        for i, action in enumerate(env.action_space):
            dist[action] = np.exp(q_values[i].detach().numpy() / TEMP) / Z

        return torch.tensor([[random.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]]],\
                            device=device, dtype=torch.float)
    else:
        return torch.tensor([[env.action_space[random.randrange(N_ACTIONS)]]],\
                            device=device, dtype=torch.float)


def optimize_model():
    global loss_history
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # THIS DOES NOT MATTER NOW BUT SHOULD EVENTUALLY EDIT THIS
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    non_final_next_states = non_final_next_states.unsqueeze(1)
    state_batch = torch.cat(batch.state)
    state_batch = state_batch.unsqueeze(1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, ((N_ACTIONS-1)*action_batch).type(torch.long))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # Double Q implementation
    _, max_indices = policy_net(non_final_next_states).max(1, keepdim=True)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, max_indices).squeeze(1)
    
    ## Compute the expected Q values
    #import pdb; pdb.set_trace()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Normalizing Q values
    max_Q_prime = torch.mean(expected_state_action_values) 
    expected_state_action_values = (expected_state_action_values - torch.mean(expected_state_action_values)) \
                                    / torch.std(expected_state_action_values)
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss_history.append(loss)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()


episode_durations = []
loss_history = []
traj_history = []
for i_episode in range(num_episodes):
    ratio = steps_done / EPS_DECAY
    if i_episode % TARGET_UPDATE == 0: print(str(i_episode) + f"({ratio:.4f}) ", end='', flush=True)
    # Initialize the environment and state
    env.reset()
    state = torch.tensor([env.fish_population], device=device, dtype=torch.float)
    traj_history.append(state)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        fish_population, reward, done, _ = env.step(action)
        reward = torch.Tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = torch.tensor([fish_population], \
                                      device=device, dtype=torch.float)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        traj_history.append(state)

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    torch.save(policy_net.state_dict(), './big_search_model.pth')

print('Complete')