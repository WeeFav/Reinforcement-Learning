import random
from environment_trajectory import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import copy

H = 5

def create_D():
    env = Grid()
    D = []
    sq = env.states_to_be_queried()
    sq = [env.det_s(coord[0], coord[1]) for coord in sq]

    i = 0
    while i < 15:
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
                    done = env.reset(render=False)
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


        # query human after gathering a pair of trajectory
        init_row, init_col = env.det_coord(d[0])
        pref = env.query(init_row, init_col, d[1], d[2])

        if (pref != 'incomparable'):
            d.append(pref)
            D.append(d)
            print(f"sample {i}")
            i += 1

    return D


def phi(s, a):
    feature = [0] * (25 * 4)
    feature[s*4 + a] = 1
    return feature

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(parameter, D):
    loss = 0

    for d in D:
        init_s = d[0]
        traj0 = copy.deepcopy(d[1])
        traj1 = copy.deepcopy(d[2])
        traj0.insert(0, init_s)
        traj1.insert(0, init_s)
        pref = d[3]


        r_traj0 = 0
        for h in range(H):
            s_idx = h * 2
            a_idx = h * 2 + 1
            r_traj0 += reward(parameter, traj0[s_idx], traj0[a_idx])

        r_traj1 = 0
        for h in range(H):
            s_idx = h * 2
            a_idx = h * 2 + 1
            r_traj1 += reward(parameter, traj1[s_idx], traj1[a_idx])

        exp_r_traj0 = math.exp(r_traj0)
        exp_r_traj1 = math.exp(r_traj1)


        if pref == 'left':
            p0 = exp_r_traj0 / (exp_r_traj0 + exp_r_traj1)
            curr_loss = math.log(p0)
        elif pref == 'right':
            p1 =  exp_r_traj1 / (exp_r_traj0 + exp_r_traj1)
            curr_loss = math.log(p1)
        elif pref == 'equal':
            p0 =  exp_r_traj0 / (exp_r_traj0 + exp_r_traj1)
            p1 =  exp_r_traj1 / (exp_r_traj0 + exp_r_traj1)
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

def train():
    print("querying....")
    D = create_D(save=False)
    print("optimizing....")
    init_parameter = [0] * 100
    solve(D, init_parameter, save=True)


def test(result):
    env = Grid()
    env.reset(render=True)
    while True:
        done = False
        env.reset(render=True)
        while not done:
            row, col = env.get_state()
            s = env.det_s(row, col)
            reward_s = result.x[s*4:s*4+4]
            action = np.argmax(reward_s)
            done = env.step(action, render=True)  

                
if __name__ == '__main__':
    # train()

    # testing
    with open('result.pkl', 'rb') as fp:
        result = pickle.load(fp)

    test(result)
