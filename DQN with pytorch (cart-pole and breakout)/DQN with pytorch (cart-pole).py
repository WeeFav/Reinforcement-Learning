# https://www.youtube.com/watch?v=NP8pXZdU-5U&ab_channel=brthor

from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x)
    
    # select action in q-learning
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0)) # get a return from forward method, which gives q_values for each action (output of the neural network)
                                            # returns a list

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item() # turn tensor into integer

        return action # number between 0 and (env.action_space.n - 1)

env = gym.make('CartPole-v0')
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100) # reward of all episode
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initalize replay buffer
obs = env.reset()

for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample() # take action
    new_obs, rew, done, _ = env.step(action) # get new observation
    transition = (obs, action, rew, done, new_obs) # get tuple to put into replay_buffer
    replay_buffer.append(transition)
    obs = new_obs # updating observation for next step

    if done:
        obs = env.reset()

# Main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) # starting from 100% random action to 2% random action
    rnd_sample = random.random()

    # take random action
    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    # take action according to online network
    else:
        action = online_net.act(obs)
    
    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs # updating observation for next step

    episode_reward += rew # update episode reward for each step taken in the episode

    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0
    
    # After done, watch it play
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 195: # max reward is 200
            while True:
                action = online_net.act(obs)
 
                obs, _, done, _ = env.step(action)
                env.render()
                if done:
                    env.reset()


    # Start gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE) # take random batch from replay buffer

    # seperating the tuple
    # putting it into a np.array is faster than python array
    obses = np.array([t[0] for t in transitions])
    actions = np.array([t[1] for t in transitions])
    rews = np.array([t[2] for t in transitions])
    dones = np.array([t[3] for t in transitions])
    new_obses = np.array([t[4] for t in transitions])

    # turning into tensor
    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute targets
    target_q_values = target_net(new_obses_t) # gives us q_values computed from the TARGET network for each observation in the batch
                                              # gives us a list of q_values for each of the observations
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # take the max q_value for eac of the observations
                                                                     # since q_values are in dim=1
                                                                     # keepdim means keep the batched dim even though there is only 1 value left (max_q) 
                                                                     # [0] because max returns a tuple --> (max, argmax)
    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values # this uses temporal difference learning to produce 'target/actual' q_values

    # Compute loss
    q_values = online_net(obses_t) # gives us q_values computed from the ONLINE network for each observation in the batch
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) # get q_values for the actual action that we took 

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network parameters
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Reward', np.mean(rew_buffer))



    