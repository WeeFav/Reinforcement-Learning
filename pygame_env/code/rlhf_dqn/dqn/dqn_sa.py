# https://www.youtube.com/watch?v=EUrWGTCGzlA&t=1088s&ab_channel=JohnnyCode

from environment import Grid
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, in_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states + in_actions, 128, dtype=torch.float64)   # first fully connected layer
        self.out = nn.Linear(128, 1, dtype=torch.float64) # ouptut layer w

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x
    
    def get_feature(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
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

# FrozeLake Deep Q-Learning
class Agent():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['U','D','L','R']

    def select_action(self, dqn, state, num_actions):
        state_tensor = torch.from_numpy(state)
        action_values = []
        for a in range(num_actions):
            action_tensor = torch.zeros(num_actions, dtype=torch.float64)
            action_tensor[a] = 1
            action_values.append(dqn(state_tensor, action_tensor))
        action_values = torch.tensor(action_values)
        return action_values


    # Train the FrozeLake environment
    def train(self, episodes, maze, render=False):
        # Create FrozenLake instance
        env = Grid(maze=maze, show_render=render)
        num_states = env.obs_space
        num_actions = len(env.action_space)
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, in_actions=num_actions)
        target_dqn = DQN(in_states=num_states, in_actions=num_actions)
 
        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            print(i)
            state = env.reset()  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    action = random.choice(env.action_space) # select random action
                else:
                    # select best action            
                    with torch.no_grad():
                        action_values = self.select_action(policy_dqn, state, num_actions)
                        action = action_values.argmax().item()

                # Execute action
                new_state,reward,terminated,truncated = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

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
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            state_tensor = torch.from_numpy(state)
            action_tensor = torch.zeros(4, dtype=torch.float64)
            action_tensor[action] = 1

            if terminated:
                target = torch.tensor(reward)
            else:
                next_action_values = self.select_action(target_dqn, new_state, 4)
                target = reward + self.discount_factor_g * next_action_values.max()

            current_q = policy_dqn(state_tensor, action_tensor)
            current_q_list.append(current_q)

            target_q = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list).squeeze(), torch.stack(target_q_list).squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, maze):
        # Create FrozenLake instance
        env = Grid(maze=maze, show_render=True)
        num_states = env.obs_space
        num_actions = len(env.action_space)

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, in_actions=num_actions) 
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
                    action_values = self.select_action(policy_dqn, state, num_actions)
                    action = action_values.argmax().item()

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
    agent = Agent()
    agent.train(3000, render=False, maze=maze)
    agent.test(10, maze=maze)