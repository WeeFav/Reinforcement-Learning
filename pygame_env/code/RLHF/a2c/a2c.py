##############################################################################
# Init
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import sys  

from custom_environment.environment import Grid

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()

        # actor
        self.actor_fc1 = nn.Linear(num_states, 128, dtype=torch.float64)   
        self.actor_fc2 = nn.Linear(128, 128, dtype=torch.float64)   
        self.actor_out = nn.Linear(128, num_actions, dtype=torch.float64)
        # critic
        self.critic_fc1 = nn.Linear(num_states, 128, dtype=torch.float64)   
        self.critic_fc2 = nn.Linear(128, 128, dtype=torch.float64)   
        self.critic_out = nn.Linear(128, 1, dtype=torch.float64)

    def forward(self, state):
        # actor
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        x = F.softmax(self.actor_out(x), dim=-1)
        # critic
        y = F.relu(self.critic_fc1(state))
        y = F.relu(self.critic_fc2(y))
        y = self.critic_out(y)    
        return x, y

class Agent():
    def __init__(self, env):
        self.env = env
        self.num_states = env.obs_space
        self.num_actions = len(env.action_space)

        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.0003        
        self.discount_factor_g = 0.9           

        # Neural Network
        self.a2c = ActorCritic(num_states=self.num_states, num_actions=self.num_actions)
        # put model onto GPU
        self.a2c.cuda()

        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr=self.learning_rate_a)

    def train(self, episodes):
        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for i in range(episodes):
            log_probs = []
            values = []
            rewards = []
            dones = []

            # print(i)
            state = self.env.reset()     
            terminated = False      
            truncated = False

            while(not terminated and not truncated):
                action_probs, value = self.a2c(torch.from_numpy(state).cuda())
                action = np.random.choice(self.num_actions, p=action_probs.detach().cpu().numpy())
                log_prob = torch.log(action_probs[action])
                entropy = -torch.sum(torch.mean(action_probs) * torch.log(action_probs))
                new_state, reward, terminated, truncated = self.env.step(action)

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(terminated)
                entropy_term += entropy
                state = new_state

            
            returns = []
            _, value = self.a2c(torch.from_numpy(new_state).cuda())
            G = value
            for idx in reversed(range(len(rewards))):
                G = rewards[idx] + self.discount_factor_g * G * (1 - dones[idx])
                returns.insert(0, G)


            returns = torch.stack(returns)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)

            advantage = returns - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            # loss = actor_loss + critic_loss + 0.001 * entropy_term
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), 0.5)
            self.optimizer.step()

            print(i, sum(rewards))
            all_rewards.append(sum(rewards)) 

        # Save policy
        torch.save(self.a2c.state_dict(), "a2c/model.pt")

        # Create new graph 
        plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            # sum_rewards[x] = np.sum(all_rewards[max(0, x-100):(x+1)])
            sum_rewards[x] = all_rewards[x]
        plt.plot(sum_rewards)
        plt.savefig('a2c/graph.png')

    def test(self, episodes):
        self.env.show_render = True

        # Load learned policy
        self.a2c.load_state_dict(torch.load("a2c/model.pt"))
        self.a2c.eval().cuda()    # switch model to evaluation mode

        for i in range(episodes):
            state = self.env.reset()     
            terminated = False     
            truncated = False
            rewards = 0             

            while(not terminated and not truncated):  
                action_probs, _ = self.a2c(torch.from_numpy(state).cuda())
                action_probs = action_probs.detach().cpu().numpy()
                action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))

                new_state, reward, terminated, truncated = self.env.step(action)
                state = new_state
                rewards += reward

    def print_probs(self):
        self.env.show_render = True

        # Load learned policy
        self.a2c.load_state_dict(torch.load("a2c/model.pt"))
        self.a2c.eval().cuda()    # switch model to evaluation mode

        valid_pos = self.env.get_valid_pos()
        for (row, col) in valid_pos:
            state = self.env.reset(row, col)
            action_probs, _ = self.a2c(torch.from_numpy(state).cuda())
            print(f"row: {row}, col: {col}, probs: {action_probs.detach().cpu().numpy()}")

if __name__ == '__main__':
    print("GPU:", torch.cuda.is_available())
    maze = 'maze1'
    env = Grid(maze=maze, show_render=False)
    agent = Agent(env)

    if sys.argv[1] == 'train':
        agent.train(int(sys.argv[2]))
    elif sys.argv[1] == 'test':
        agent.test(10)
    elif sys.argv[1] == 'probs':
        agent.print_probs()

          


