import random
from environment import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import itertools
from dqn_claude import DQN
import torch

def allowed_actions(x, y):
    actions = [0, 1, 2, 3]

    if (y == 200):
        actions.remove(1)
    if (x == 0):
        actions.remove(2)
    if (x == 200):
        actions.remove(3)
    if (y == 0):
        actions.remove(0)

    return actions

def set_synthetic_reward():
    synthetic_reward = []
    for s in range(25):
        x, y = det_coord(s)
        action_list = allowed_actions(x, y)
        for a in range(4):
            if a not in action_list:
                synthetic_reward.append(-10)
            elif (x == 150 and y == 0 and a == 3):
                synthetic_reward.append(10)
            elif (x == 200 and y == 50 and a == 0):
                synthetic_reward.append(10)
            else:
                synthetic_reward.append(0)
    return synthetic_reward

def set_prob_matrix(synthetic_reward):
    prob_matrix = dict()
    permutation = list(itertools.permutations([0,1,2,3]))
    for s in range(25):
        permutation_list = []
        action_list = allowed_actions(*det_coord(s))
        for p in range(len(permutation)):
            prob = 1
            for i in range(len(action_list)):
                t1 = math.exp(reward(synthetic_reward, s, permutation[p][i]))
                t2 = 0
                for j in range(i, len(action_list)):
                    t2 += math.exp(reward(synthetic_reward, s, permutation[p][j]))
                prob *= t1 / t2
            permutation_list.append(prob)
        prob_matrix[s] = permutation_list

    return prob_matrix

def permutation_lookup(index):
    permutation = list(itertools.permutations([0,1,2,3]))
    return permutation[index]

def find_max_prob_list(prob_list):
    m = max(prob_list)
    max_prob_list = [i for i, j in enumerate(prob_list) if j == m]
    return max_prob_list

def create_D(maze):
    env = Grid(maze=maze, show_render=False)
    D = []
    
    valid_pos = env.get_valid_pos()
    for (row, col) in valid_pos:
        # query human for ranking
        rank, state = env.query(row, col)
        D.append((state, rank))

    with open(f'D_{maze}.pkl', 'wb') as fp:
        pickle.dump(D, fp)
        print(f'D_{maze} saved successfully to file')

    return D

def phi(s, a):
    env = Grid(maze='maze1', show_render=False)
    model = DQN(in_states=env.obs_space, in_actions=len(env.action_space))
    model.load_state_dict(torch.load("model.pt"))
    feature = model.get_feature(s, a)
    return feature.detach().numpy()

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(parameter, D):
    loss = 0

    for i in range(len(D)):
        for j in range(4):
            state_tensor = torch.from_numpy(D[i][0])
            action_tensor = torch.zeros(4)
            action = D[i][1][j]
            action_tensor[action] = 1
            t1 = math.exp(reward(parameter, state_tensor, action_tensor))
            
            t2 = 0
            for k in range(j, 4):
                action_tensor = torch.zeros(4)
                action = D[i][1][k]
                action_tensor[action] = 1
                t2 += math.exp(reward(parameter, state_tensor, action_tensor))

            curr_loss = math.log(t1/t2)
            loss += curr_loss
    
    return -1 / len(D) * loss 

def solve(D, init_parameter, save, result_filename):
    result = optimize.minimize(mle_loss, init_parameter, args=D)

    if save:
        with open(f'{result_filename}.pkl', 'wb') as fp:
            pickle.dump(result, fp)
            print(f'{result_filename} saved successfully to file')
  
    print(result.x) 

def train(D, result_filename):
    init_parameter = [0] * 128
    solve(D, init_parameter, save=True, result_filename=result_filename)

def test(result):
    env = Grid()
    while True:
        done = False
        env.reset()
        while not done:
            row, col = env.get_state()
            s = det_s(row, col)
            reward_s = result.x[s*4:s*4+4]
            action = np.argmax(reward_s)
            done = env.step(action)
         
if __name__ == '__main__':
    maze = 'maze1'
    create_D(maze)
    # with open(f'D_{maze}.pkl', 'rb') as fp:
    #     D = pickle.load(fp)
    # train(D, f'result_{maze}')

    # s = D[0][0]
    # s = torch.from_numpy(s)
    # action_tensor = torch.zeros(4)
    # a = D[0][1][0]
    # action_tensor[a] = 1

    # print(s, action_tensor)
    # phi(s, action_tensor)


    # testing
    # with open(result_filename, 'rb') as fp:
    #     result = pickle.load(fp)
    # test(result)


    # synthetic_reward = set_synthetic_reward()
    # print(synthetic_reward)
    # prob_matrix = set_prob_matrix(synthetic_reward)
    # max_prob_list = find_max_prob_list(prob_matrix[23])
    # p = [permutation_lookup(l) for l in max_prob_list]
    # print((p))
    # print(permutation_lookup())
