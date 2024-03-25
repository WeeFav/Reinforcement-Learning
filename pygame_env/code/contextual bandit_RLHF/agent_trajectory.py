import random
from environment_trajectory import Grid
import math
import numpy as np
from scipy import optimize
import pickle

H = 5

def create_D():
    env = Grid()
    D = []
    sq = env.states_to_be_queried()
    sq = [env.det_s(coord[0], coord[1]) for coord in sq]

    for i in range(3):
        d = []
        # sample a state
        init_s = random.choice(sq)
        d.append(init_s)

        for j in range(2):
            trajectory = []
            # determine coordiate of s
            row, col = env.det_coord(init_s)
            env.move_player(row, col)
            done = False

            # pick action and observe state H times
            for h in range(H):
                if done:
                    done = env.reset()
                    trajectory.append(0) # doesn't matter what action is taken in done state
                    trajectory.append(env.det_s(4, 0)) # next state must be reset state
                    continue
                
                # determine valid actions
                row, col = env.get_state()
                actions = env.allowed_actions(row, col)

                # sample an action
                a = random.choice(actions)
                trajectory.append(a)

                # step 
                done = env.step(a, render=False)
                row, col = env.get_state()
                trajectory.append(env.det_s(row, col))
            
            d.append(trajectory)            

        D.append(d)

        # query human after gathering a pair of trajectory
        init_row, init_col = env.det_coord(d[0])
        pref = env.query(init_row, init_col, d[1], d[2])
        print(pref)
        D.append(pref)

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
    # env = Grid()


