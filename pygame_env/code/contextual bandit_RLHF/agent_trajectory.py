import random
from environment_trajectory import Grid
import math
import numpy as np
from scipy import optimize
import pickle

H = 5

def det_coord(s):
    row = (s // 5)
    col = (s % 5)

    return row, col

def det_s(row, col):
    return row * 5 + col

def create_D():
    env = Grid()
    D = []
    sq = env.states_to_be_queried()
    sq = [det_s(coord[0], coord[1]) for coord in sq]

    for i in range(3):
        d = []
        # sample a state
        init_s = random.choice(sq)
        d.append(init_s)

        # determine coordiate of s
        row, col = det_coord(init_s)

        for j in range(2):
            trajectory = []

            # generate trajectory of length H
            # env.show_inital(row, col)
            done = False
            for h in range(H):
                if done:
                    env.reset()

                # determine valid actions
                actions = env.allowed_actions(row, col)

                # sample an action
                a = random.choice(actions)
                trajectory.append(a)

                # step 
                done = env.step(a, render=False)
                row, col = env.get_state()
                trajectory.append(det_s(row, col))
            
            d.append(trajectory)            

        D.append(d)

        # query human
        init_row, init_col = det_coord(d[0])
        pref = env.query(init_row, init_col, d[1], d[2])
        
    print(D)


def phi(s, a):
    feature = [0] * (25 * 4)
    feature[s*4 + a] = 1
    return feature

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(parameter, D):
    loss = 0

    for i in range(len(D)):
        s = D[i][0]
        a0 = D[i][1]
        a1 = D[i][2]
        pref = D[i][3]
        t0 = math.exp(reward(parameter, s, a0))
        t1 = math.exp(reward(parameter, s, a1))

        if pref == 'left':
            p0 =  t0 / (t0 + t1)
            curr_loss = math.log(p0)
        elif pref == 'right':
            p1 =  t1 / (t0 + t1)
            curr_loss = math.log(p1)
        elif pref == 'equal':
            p0 =  t0 / (t0 + t1)
            p1 =  t1 / (t0 + t1)
            curr_loss = math.log(p0 + p1)

        loss += curr_loss
    
    return -loss  

def solve(D, init_parameter, save):
    result = optimize.minimize(mle_loss, init_parameter, args=D)

    if save:
        with open('result_pair.pkl', 'wb') as fp:
            pickle.dump(result, fp)
            print('result saved successfully to file')
  
    print(result.x)   

                
if __name__ == '__main__':
    # create_D()
    env = Grid()
    env.show_inital_2(0,0)


