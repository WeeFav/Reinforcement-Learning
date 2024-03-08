import random
from environment_k import Grid
import math
import numpy as np
from scipy import optimize
import pickle

def det_coord(s):
    col = (s % 5) * 50
    row = (s // 5) * 50

    return col, row

def det_s(x, y):
    col = x // 50
    row = y // 50

    return col * 5 + row

def create_D(save):
    env = Grid()
    D = []

    for i in range(10):
        if (i == 4):
            continue
        # sample a state
        s = random.randrange(25)
        # s = i
        # determine coordiate of s
        curr_x, curr_y = det_coord(s)
        # query human for ranking
        rank = env.query(curr_x, curr_y)
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
  
    print(result.x)   
                
if __name__ == '__main__':
    # training
    # print("querying....")
    # D = create_D(save=False)
    # print("optimizing....")
    # init_parameter = [1] * 100
    # solve(D, init_parameter, save=False)

    # testing
    with open('result.pkl', 'rb') as fp:
        result = pickle.load(fp)

    print(result)

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



