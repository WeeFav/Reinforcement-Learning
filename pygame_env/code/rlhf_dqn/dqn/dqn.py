##############################################################################
# Batch processing during optimization step.
# Only compute loss in q_values according to the action taken instead of every q_values in that state.
##############################################################################

from environment import Grid
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import sys

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, 128, dtype=torch.float64)   # first fully connected layer
        self.out = nn.Linear(128, out_actions, dtype=torch.float64) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, env):
        self.env = env
        self.num_states = env.obs_space
        self.num_actions = len(env.action_space)

        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.001         # learning rate (alpha)
        self.discount_factor_g = 0.9         # discount rate (gamma)    
        self.network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 1000       # size of replay memory
        self.mini_batch_size = 32            # size of the training data set sampled from the replay memory
        self.epsilon = 1                     # 1 = 100% random actions

        self.memory = ReplayMemory(self.replay_memory_size)

        # Neural Network
        self.policy_dqn = DQN(in_states=self.num_states, out_actions=self.num_actions)
        self.target_dqn = DQN(in_states=self.num_states, out_actions=self.num_actions)
        self.policy_dqn.cuda()
        self.target_dqn.cuda()
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict()) # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

    def train(self, episodes):
        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0
            
        for i in range(episodes):
            print(i)
            state = self.env.reset()     # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select action based on epsilon-greedy
                if random.random() < self.epsilon:
                    action = random.choice(self.env.action_space) # select random action
                else:
                    # select best action            
                    with torch.no_grad():
                        action = self.policy_dqn(torch.from_numpy(state).cuda()).detach().argmax().item()

                # Execute action
                new_state, reward, terminated, truncated = self.env.step(action)

                # Save experience into memory
                self.memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(self.memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = self.memory.sample(self.mini_batch_size)
                self.optimize(mini_batch)        

                # Decay epsilon
                self.epsilon = max(self.epsilon - 1/episodes, 0)
                epsilon_history.append(self.epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count = 0

        # Save policy
        torch.save(self.policy_dqn.state_dict(), "model.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('graph.png')

    # Optimize policy network
    def optimize(self, mini_batch):
        state_batch = torch.from_numpy(np.array([t[0] for t in mini_batch]))
        action_batch = torch.from_numpy(np.array([t[1] for t in mini_batch], dtype=np.int64)).unsqueeze(1).cuda()
        new_state_batch = torch.from_numpy(np.array([t[2] for t in mini_batch]))
        reward_batch = torch.from_numpy(np.array([t[3] for t in mini_batch])).cuda()
        terminated_batch = np.array([t[4] for t in mini_batch])

        target_action_values = self.target_dqn(new_state_batch.cuda())
        target_q_values = reward_batch + torch.from_numpy((1 - terminated_batch)).cuda() * self.discount_factor_g * target_action_values.max(dim=1)[0]

        current_action_values = self.policy_dqn(state_batch.cuda())
        current_q_values = torch.gather(input=current_action_values, dim=1, index=action_batch).squeeze()

        # Compute loss for the whole minibatch
        loss = self.loss_fn(target_q_values, current_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Run the FrozeLake environment with the learned policy
    def test(self, episodes):
        self.env.show_render = True

        # Load learned policy
        policy_dqn = DQN(in_states=self.num_states, out_actions=self.num_actions) 
        policy_dqn.load_state_dict(torch.load("model.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            print(i)
            state = self.env.reset()     # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(torch.from_numpy(state)).argmax().item()

                # Execute action
                state, reward, terminated, truncated = self.env.step(action)

if __name__ == '__main__':
    print("GPU:", torch.cuda.is_available())
    maze = 'maze1'
    env = Grid(maze=maze, show_render=False)
    agent = Agent(env)
    agent.train(3000)
    agent.test(10)