# https://dilithjay.com/blog/ddqn

from custom_environment import Grid
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

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
        self.tau = 0.001

        self.memory = ReplayMemory(self.replay_memory_size)

        # Neural Network
        self.policy_dqn = DQN(in_states=self.num_states, out_actions=self.num_actions)
        self.target_dqn = DQN(in_states=self.num_states, out_actions=self.num_actions)
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
                if random.random() < epsilon:
                    action = random.choice(self.env.action_space) # select random action
                else:
                    # select best action            
                    with torch.no_grad():
                        action = self.policy_dqn(torch.from_numpy(state)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated = env.step(action)

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
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0


        # Save policy
        torch.save(policy_dqn.state_dict(), "model.pt")

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
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = reward + self.discount_factor_g * target_dqn(torch.from_numpy(new_state)).max()

            # Get the current set of Q values
            current_q = policy_dqn(torch.from_numpy(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(torch.from_numpy(state)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update target network parameters with polyak averaging
        for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, maze):
        # Create FrozenLake instance
        env = Grid(maze=maze, show_render=True)
        num_states = env.obs_space
        num_actions = len(env.action_space)

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=32, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("model.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        # print('Policy (trained):')
        # self.print_dqn(policy_dqn)

        for i in range(episodes):
            print(i)
            state = env.reset()     # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(torch.from_numpy(state)).argmax().item()

                # Execute action
                state,reward,terminated,truncated = env.step(action)

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':
    maze = 'maze1'
    env = Grid(maze=maze, show_render=False)
    agent = Agent(env=env)
    agent.train(3000, render=False, maze=maze)
    # agent.test(10, maze=maze)