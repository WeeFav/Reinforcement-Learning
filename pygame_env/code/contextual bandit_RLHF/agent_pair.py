import random
from environment_pair import Grid
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

def allowed_actions(coord):
    x = coord[0]
    y = coord[1]
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

def create_D(value, save):
    env = Grid()
    D = []

    i = 0
    while i < 1000:
        # sample a state
        s = random.randrange(25)
        # determine coordiate of s
        curr_x, curr_y = det_coord(s)
        # determine valid actions
        actions = allowed_actions((curr_x, curr_y))
        # sample 2 actions
        a0, a1 = random.sample(actions, 2)
        # query human
        pref = env.query(curr_x, curr_y, a0, a1, value)
        if pref == 'incomparable':
            continue
        # append into dataset
        D.append([s, a0, a1, pref])
        i += 1

    if save:
        with open('D_pair.pkl', 'wb') as fp:
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
    with open('value.pkl', 'rb') as fp:
        value = pickle.load(fp)

    # training
    print("querying....")
    D = create_D(value, save=True)
    print("optimizing....")
    init_parameter = [1] * 100
    solve(D, init_parameter, save=True)


