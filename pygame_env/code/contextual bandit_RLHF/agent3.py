import random
from environment import Grid
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pickle

class Agent:
    def __init__(self):
        self.parameter = [5] * 100 # need to change!  

def det_coord(s):
    col = (s % 5) * 50
    row = (s // 5) * 50

    return col, row

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

def create_D():
    env = Grid()
    D = []

    for _ in range(10):
        # sample a state
        s = random.randrange(25)
        # determine coordiate of s
        x, y = det_coord(s)
        # determine valid actions
        actions = allowed_actions((x, y))
        # sample 2 actions
        a0, a1 = random.sample(actions, 2)
        # append into dataset
        D.append([s, a0, a1])
    
    for d in D:
        curr_x, curr_y = det_coord(d[0])
        choice1 = [(curr_x, curr_y), d[1]]
        choice2 = [(curr_x, curr_y), d[2]]
        pref = env.query(choice1, choice2)
        d.append(pref)

    return D

def phi(s, a):
    feature = [0] * (25 * 4)
    feature[s*4 + a] = 1
    return feature

def reward(parameter, s, a):
    return np.dot(parameter, phi(s, a))

def mle_loss(D, parameter):
    loss = 0

    for d in D:
        s = d[0]
        a0 = d[1]
        a1 = d[2]
        y = d[3]
        p0 = math.exp(reward(parameter, s, a0)) / (math.exp(reward(parameter, s, a0)) + math.exp(reward(parameter, s, a1)))
        p1 = math.exp(reward(parameter, s, a1)) / (math.exp(reward(parameter, s, a0)) + math.exp(reward(parameter, s, a1)))
        if (y == 0):
            curr_loss = math.log(p0)
        else:
            curr_loss = math.log(p1)
        loss += curr_loss
    
    return -loss     
    
def pmle(parameter):
    pass
            
if __name__ == '__main__':
    pass