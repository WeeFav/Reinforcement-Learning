import random
from custom_environment import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import itertools
from dqn.dqn_sa import DQN
import torch

class Agent():
    def __init__(self, env):
        self.env = env
        self.num_actions = len(self.env.action_space)
        self.model = DQN(in_states=self.env.obs_space, in_actions=self.num_actions)
        self.model.load_state_dict(torch.load(f"model_maze{self.env.maze[4]}.pt"))
        self.model.eval()

    def phi(self, s, a):
        feature = self.model.get_feature(s, a)
        return feature.detach().numpy()

    def reward(self, parameter, s, a):
        return np.dot(parameter, self.phi(s, a))
    
    def get_one_hot_action(self, action):
        one_hot_action = torch.zeros(self.num_actions, dtype=torch.float64)
        one_hot_action[action] = 1
        return one_hot_action

    def mle_loss(self, parameter, D):
        loss = 0
        for i in range(len(D)):
            for j in range(4):
                state_tensor = torch.from_numpy(D[i][0])
                action_tensor = torch.zeros(4)
                action = D[i][1][j]
                action_tensor[action] = 1
                t1 = math.exp(self.reward(parameter, state_tensor, action_tensor))
                
                t2 = 0
                for k in range(j, 4):
                    action_tensor = torch.zeros(4)
                    action = D[i][1][k]
                    action_tensor[action] = 1
                    t2 += math.exp(self.reward(parameter, state_tensor, action_tensor))

                curr_loss = math.log(t1/t2)
                loss += curr_loss
        
        return -1 / len(D) * loss 

    def solve(self, D, init_parameter, save, result_filename):
        result = optimize.minimize(self.mle_loss, init_parameter, args=D)

        if save:
            with open(f'{result_filename}.pkl', 'wb') as fp:
                pickle.dump(result, fp)
                print(f'saved {result_filename}.pkl')
    
        print(result.x) 

    def train(self, D, result_filename):
        init_parameter = [0] * 64
        self.solve(D, init_parameter, save=True, result_filename=result_filename)

    def test(self, parameter):
        self.env.show_render = True

        valid_pos = self.env.get_valid_pos()
        for (row, col) in valid_pos:
            if (row, col) == (3, 4):
                continue
            
            state = self.env.reset(row, col)

            print(row, col)
            r_list = []
            state_tensor = torch.from_numpy(state)
            for i in range(self.num_actions):
                action_tensor = self.get_one_hot_action(i)
                r = self.reward(parameter, state_tensor, action_tensor)
                r_list.append(r)
            print(r_list)


            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select best action according to reward  
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state)
                    r_list = []
                    for i in range(self.num_actions):
                        action_tensor = self.get_one_hot_action(i)
                        r = self.reward(parameter, state_tensor, action_tensor)
                        r_list.append(r)
                    
                action = np.array(r_list).argmax()
                # Execute action
                state, reward, terminated, truncated = self.env.step(action)

def create_D(env):
    env.reset()
    D = []

    valid_pos = env.get_valid_pos()
    no_sample_list = []
    for (row, col) in valid_pos:
        if (row, col) not in no_sample_list:
            # query human for ranking
            rank, state = env.query(row, col)
            D.append((state, rank))

    with open(f'D_{env.maze}.pkl', 'wb') as fp:
        pickle.dump(D, fp)
        print(f'saved D_{env.maze}.pkl')

    return D

def copy_D(D, num_copy):
    copy_list = D.copy()
    for _ in range(num_copy - 1):
        D.extend(copy_list)
    
    return D


if __name__ == '__main__':
    maze = 'maze3'
    env = Grid(maze=maze, show_render=False)
    agent = Agent(env)

    # # create dataset
    # create_D(env)

    # with open(f'D_{maze}.pkl', 'rb') as fp:
    #     D = pickle.load(fp)

    # D = copy_D(D, 5)
    # print(len(D))

    # # train
    # agent.train(D, f'result_{maze}')

    # testing
    result_filename = f'result_{maze}.pkl'
    with open(result_filename, 'rb') as fp:
        result = pickle.load(fp)
    agent.test(result.x)