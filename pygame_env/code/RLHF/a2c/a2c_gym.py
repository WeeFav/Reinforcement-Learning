import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import sys  
import gym

from custom_environment.environment import Grid

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()

        # actor
        self.actor_fc1 = nn.Linear(num_states, 256, dtype=torch.float64)   
        self.actor_out = nn.Linear(256, num_actions, dtype=torch.float64)
        # critic
        self.critic_fc1 = nn.Linear(num_states, 256, dtype=torch.float64)   
        self.critic_out = nn.Linear(256, 1, dtype=torch.float64)

    def forward(self, state):
        # actor
        x = F.relu(self.actor_fc1(state))
        x = F.softmax(self.actor_out(x))
        # critic
        y = F.relu(self.critic_fc1(state))
        y = self.critic_out(y)    
        return x, y

class Agent():
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        # Hyperparameters (adjustable)
        self.learning_rate_a = 3e-4         
        self.discount_factor_g = 0.99           

        # Neural Network
        self.a2c = ActorCritic(num_states=self.num_states, num_actions=self.num_actions)
        # put model onto GPU
        self.a2c.float().cuda()

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

            print(i)
            state = self.env.reset()[0]     
            terminated = False      
            truncated = False

            while(not terminated and not truncated):
                action_probs, value = self.a2c(torch.from_numpy(state).cuda())
                action_probs = action_probs.detach().cpu().numpy()
                value = value.detach().cpu().numpy()

                action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
                log_prob = np.log(action_probs[action])

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                state = np.array(new_state)
            
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + self.discount_factor_g * G
                returns.insert(0, G)

            returns = torch.tensor(returns, requires_grad=True)
            values = torch.tensor(values, requires_grad=True)
            log_probs = torch.tensor(log_probs, requires_grad=True)

            advantage = returns - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_rewards.append(sum(rewards)) 

        # Save policy
        torch.save(self.a2c.state_dict(), "a2c/model.pt")

        # Create new graph 
        plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(all_rewards[max(0, x-100):(x+1)])
        plt.plot(sum_rewards)
        plt.savefig('a2c/graph.png')

    def test(self, episodes):
        self.env.show_render = True

        # Load learned policy
        a2c = ActorCritic(num_states=self.num_states, num_actions=self.num_actions) 
        a2c.load_state_dict(torch.load("a2c/model.pt"))
        a2c.eval()    # switch model to evaluation mode

        for i in range(episodes):
            print(i)
            state = self.env.reset()     
            terminated = False     
            truncated = False               

            while(not terminated and not truncated):  
                action_probs, _ = self.a2c(torch.from_numpy(state).cuda())
                action_probs = action_probs.detach().cpu().numpy()
                action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))

                new_state, _, terminated, truncated = self.env.step(action)
                state = new_state

    def test_all_state(self):
        self.env.show_render = True

        # Load learned policy
        a2c = ActorCritic(num_states=self.num_states, num_actions=self.num_actions) 
        a2c.load_state_dict(torch.load("a2c/model.pt"))
        a2c.eval()    # switch model to evaluation mode

        valid_pos = self.env.get_valid_pos()
        for (row, col) in valid_pos:
            state = self.env.reset(row, col)
            action_probs, _ = self.a2c(torch.from_numpy(state).cuda())
            print(f"row: {row}, col: {col}, probs: {action_probs.detach().cpu().numpy()}")

if __name__ == '__main__':
    print("GPU:", torch.cuda.is_available())
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    agent.train(3000)
          


