import random
from environment_trajectory import Grid
import math
import numpy as np
from scipy import optimize
import pickle

S = list(range(0,4)) + list(range(5,25))
H = 5

def det_coord(s):
    col = (s % 5) * 50
    row = (s // 5) * 50

    return col, row

def det_s(x, y):
    col = x // 50
    row = y // 50

    return col * 5 + row

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

def create_D():
    env = Grid()
    D = []

    for i in range(3):
        d = []

        # sample a state
        init_s = random.choice(S)
        d.append(init_s)

        # determine coordiate of s
        curr_x, curr_y = det_coord(init_s)

        for j in range(2):
            trajectory = []

            # generate trajectory of length H
            env.show_inital(curr_x, curr_y)
            done = False
            for h in range(H):
                if done:
                    env.reset()

                # determine valid actions
                actions = allowed_actions(curr_x, curr_y)

                # sample an action
                a = random.choice(actions)
                trajectory.append(a)

                # step 
                done = env.step(a, 1)
                curr_x, curr_y = env.get_state()
                trajectory.append(det_s(curr_x, curr_y))
            
            d.append(trajectory)

        D.append(d)
        
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
    create_D()


