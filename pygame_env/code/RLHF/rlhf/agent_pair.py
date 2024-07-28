import random
from environment_pair import Grid
import math
import numpy as np
from scipy import optimize
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

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
    i = 0
    for init_s in sq:
    # while i < 10:
        # sample a state
        # init_s = random.choice(sq)
        # determine coordiate of s
        row, col = det_coord(init_s)
        if (row == 4 and col == 0):
            continue
        # determine valid actions
        actions = env.allowed_actions(row, col)
        # sample 2 actions
        a0, a1 = random.sample(actions, 2)
        # query human
        pref = env.query(row, col, a0, a1)
        if pref == 'incomparable':
            continue
        # append into dataset
        D.append([init_s, a0, a1, pref])
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

# NN for learning reward function:
class Reward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.sigmoid(self.layer2(x))
        

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

class CustomImageDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    model = Reward(2, 128, 1)
    D = create_D(False)
    x = []
    y = []
    for d in D:
        if (d[3] == 'left'):
            x.append([d[0], d[1]])
            y.append(1)
            x.append([d[0], d[2]])
            y.append(0)
        elif (d[3] == 'right'):
            x.append([d[0], d[1]])
            y.append(0)
            x.append([d[0], d[2]])
            y.append(1)
        else:
            assert(0)

    training_set = CustomImageDataset(x, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)

    for epochs in range(10):
        model.train()

        for i, data in enumerate(training_loader):
            inputs, labels = data


            optimizer.zero_grad()
            outputs = model(inputs).squeeze(dim=1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            print(loss)
