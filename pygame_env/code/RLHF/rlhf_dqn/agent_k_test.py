import random
from environment_test import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import itertools

def det_coord(s):
    row = (s // 6)
    col = (s % 6)

    return row, col

def det_s(row, col):
    return row * 6 + col

def create_D(env):
    env.reset()
    D = []
    
    sq = [det_s(row, col) for (row, col) in env.get_valid_pos()]

    for state in sq:
        # determine coordiate of s
        row, col = det_coord(state)
        # query human for ranking
        print(row, col)
        rank, _ = env.query(row, col)
        D.append((state, rank))

    with open(f'D_{maze}.pkl', 'wb') as fp:
        pickle.dump(D, fp)
        print(f'saved D_{maze}.pkl')

    return D

def phi(s, a):
    feature = [0] * (36 * 4)
    feature[s*4 + a] = 1
    return feature

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(parameter, D):
    loss = 0
    for i in range(len(D)):
        for j in range(4):
            state = D[i][0]
            action = D[i][1][j]
            t1 = math.exp(reward(parameter, state, action))
            
            t2 = 0
            for k in range(j, 4):
                action = D[i][1][k]
                t2 += math.exp(reward(parameter, state, action))

            curr_loss = math.log(t1/t2)
            loss += curr_loss
    
    return -1 / len(D) * loss 

def solve(D, init_parameter, save, result_filename):
    result = optimize.minimize(mle_loss, init_parameter, args=D)

    if save:
        with open(f'{result_filename}.pkl', 'wb') as fp:
            pickle.dump(result, fp)
            print(f'saved {result_filename}.pkl')

    print(result.x) 

def train(D, result_filename):
    init_parameter = [0] * (36 * 4)
    solve(D, init_parameter, save=True, result_filename=result_filename)

def test(env, parameter):
    env.show_render = True
    valid_pos = env.get_valid_pos()

    for (row, col) in valid_pos:
        _ = env.reset(row, col)

        terminated = False
        truncated = False 

        while(not terminated and not truncated):
            state = det_s(row, col)
            reward_s = parameter[state*4:state*4+4]
            action = np.argmax(reward_s)
            _, _, terminated, truncated = env.step(action)
            (row, col) = env.get_player_pos()
         
if __name__ == '__main__':
    maze = 'maze3'
    env = Grid(maze=maze, show_render=False)
    
    # create_D(env)

    # # train
    # with open(f'D_{maze}.pkl', 'rb') as fp:
    #     D = pickle.load(fp)
    # train(D, f'result_{maze}')

    # testing
    result_filename = f'result_{maze}.pkl'
    with open(result_filename, 'rb') as fp:
        result = pickle.load(fp)
    test(env, result.x)

