import random
from environment import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import itertools

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

def det_coord(s):
    row = (s // 5)
    col = (s % 5)

    return row, col

def det_s(row, col):
    return row * 5 + col

def create_D(save):
    env = Grid()
    D = []
    sq = env.states_to_be_queried()
    sq = [det_s(coord[0], coord[1]) for coord in sq]

    for s in sq:
        # determine coordiate of s
        curr_row, curr_col = det_coord(s)
        # query human for ranking
        rank = env.query(curr_row, curr_col)
        D.append([s,0,1,2,3,rank])

    if save:
        with open('D.pkl', 'wb') as fp:
            pickle.dump(D, fp)
            print('D saved successfully to file')

    return D

def phi(s, a):
    feature = [0] * (25 * 4)
    feature[s*4 + a] = 1
    return feature

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(parameter, D):
    loss = 0

    for i in range(len(D)):
        for j in range(4):
            t1 = math.exp(reward(parameter, D[i][0], D[i][5][j]))
            
            t2 = 0
            for k in range(j, 4):
                t2 += math.exp(reward(parameter, D[i][0], D[i][5][k]))

            curr_loss = math.log(t1/t2)
            loss += curr_loss
    
    return -1 / len(D) * loss

def solve(D, init_parameter, save):
    result = optimize.minimize(mle_loss, init_parameter, args=D)

    if save:
        with open('result.pkl', 'wb') as fp:
            pickle.dump(result, fp)
            print('result saved successfully to file')
                
if __name__ == '__main__':
    # training
    print("querying....")
    D = create_D(save=False)
    # print("optimizing....")
    # init_parameter = [1] * 100
    # solve(D, init_parameter, save=False)

    # testing
    # with open('result.pkl', 'rb') as fp:
    #     result = pickle.load(fp)

    # env = Grid()
    # while True:
    #     done = False
    #     env.reset()
    #     while not done:
    #         x, y = env.get_state()
    #         s = det_s(x, y)
    #         reward_s = result.x[s*4:s*4+4]
    #         action = np.argmax(reward_s)
    #         done = env.step(action, 1)
    # synthetic_reward = set_synthetic_reward()
    # print(synthetic_reward)
    # prob_matrix = set_prob_matrix(synthetic_reward)
    # max_prob_list = find_max_prob_list(prob_matrix[23])
    # p = [permutation_lookup(l) for l in max_prob_list]
    # print((p))
    # print(permutation_lookup())
