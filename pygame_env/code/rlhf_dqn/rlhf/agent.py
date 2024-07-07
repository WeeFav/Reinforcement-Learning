import random
from environment import Grid
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy
import pickle

class Agent:
    def __init__(self):
        self.action = None 
        self.replay_buffer = []
        self.model = Reward(3, 256, 1) 
        # initalize value of each state 
        self.V = {}
        for x in range(0, 250, 50):
            for y in range(0, 250, 50):
                if (x == 0 and y == 0):
                    self.V[(x,y)] = 0
                elif (x == 50 and y == 0):
                    self.V[(x,y)] = 0
                elif (x == 0 and y == 50):
                    self.V[(x,y)] = 0
                elif (x == 50 and y == 50):
                    self.V[(x,y)] = 0
                elif (x == 200 and y == 0):
                    self.V[(x,y)] = 1 
                else:
                    self.V[(x,y)] = 0


    # e-greedy policy for choosing action:
    def select_action(self, state, eps):
        p = random.uniform(0, 1)
        if p <= eps:
            self.action = torch.tensor(random.randrange(4))
        else:
            state_a1_t = torch.from_numpy(numpy.array([state[0], state[1], 0]))
            state_a2_t = torch.from_numpy(numpy.array([state[0], state[1], 1]))
            state_a3_t = torch.from_numpy(numpy.array([state[0], state[1], 2]))
            state_a4_t = torch.from_numpy(numpy.array([state[0], state[1], 3]))            
            preds1 = self.model(state_a1_t.float())
            preds2 = self.model(state_a2_t.float())
            preds3 = self.model(state_a3_t.float())
            preds4 = self.model(state_a4_t.float())
            print("up: {}, down: {}, left: {}, right: {}".format(preds1.detach().numpy(), preds2.detach().numpy(), preds3.detach().numpy(), preds4.detach().numpy()))
            self.action = torch.argmax(torch.tensor([preds1, preds2, preds3, preds4]))

        return self.action

    def select_action_value_iteration(self, eps, env):
        # choose action with most expected value
        max_nxt_reward = -1000000
        nxt_x = env.x
        nxt_y = env.y

        if random.uniform(0, 1) <= eps:
            chosen_action = random.randrange(4)
        else:
            # greedy action
            for action in range(4):
                if ((action == 0) and (env.y - 50 >= 0)): # up
                    nxt_y -= 50
                elif ((action == 1) and (env.y + 50 <= (env.COLUMNS-1)*env.TIESIZE)): # down
                    nxt_y += 50
                elif ((action == 2) and (env.x - 50 >= 0)): # left
                    nxt_x -= 50
                elif ((action == 3) and (env.x + 50 <= (env.ROWS-1)*env.TIESIZE)): # right
                    nxt_x += 50

                nxt_reward = self.V[(nxt_x, nxt_y)]

                if nxt_reward >= max_nxt_reward:
                    chosen_action = action
                    max_nxt_reward = nxt_reward

        return chosen_action
    

# NN for learning reward function:
class Reward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

# loss for reward function:
def reward_loss(model, choice1, choice2, pref):
    state1_t = torch.from_numpy(numpy.array([choice1[0][0], choice1[0][1], choice1[1]]))
    state2_t = torch.from_numpy(numpy.array([choice2[0][0], choice2[0][1], choice1[1]]))
    reward1 = model(state1_t.float())
    reward2 = model(state2_t.float())
    p1 = torch.exp(reward1) / (torch.exp(reward1) + torch.exp(reward2))
    p2 = torch.exp(reward2) / (torch.exp(reward1) + torch.exp(reward2))
    loss = - (pref[0] * torch.log(p1) + pref[1] * torch.log(p2))

    return loss

# optimizer for loss function



# agent choose an action according to e-greedy policy (involves estimates from reward function) -> 
# interact with environment for a few steps to explore -> generate query -> 
# compute loss (involves estimates from reward function and human preference) ->
# update reward function -> repeat

# when to query human? how to speed up learning process?

# find value of each state for synthetic reward 
def value_iteration(lr, rounds = 100):
    env = Grid()
    agent = Agent()
    
    for i in range(rounds):
        env.reset()
        agent.replay_buffer.clear()

        while env.game_over == False:
            action = agent.select_action_value_iteration(0.5, env)
            env.step(action, 50)
            agent.replay_buffer.append(env.get_state())

            if (env.game_over):
                value = 1
                for s in reversed(agent.replay_buffer):
                    if (s == (0,0)):
                        value = 0
                    elif (s == (50,0)):
                        value = 0
                    elif (s == (0,50)):
                        value = 0
                    elif (s == (50,50)):
                        value = 0
                    else:
                        value = agent.V[s] + lr * (value - agent.V[s])
                        agent.V[s] = value
    
    with open('value.pkl', 'wb') as fp:
        pickle.dump(agent.V, fp)
        print('dictionary saved successfully to file')

    return agent.V


def distance():
    agent = Agent()
    target_x = 200
    target_y = 0

    for x in range(0, 250, 50):
        for y in range(0, 250, 50):
            if ((x,y) == (0,0)):
                value = -1
            elif ((x,y) == (50,0)):
                value = -1
            elif ((x,y) == (0,50)):
                value = -1
            elif ((x,y) == (50,50)):
                value = -1
            else:
                value = 1 - math.sqrt(pow((target_x - x), 2) + pow((target_y - y), 2))
                value = numpy.interp(value, [-282,0], [0,1])

            agent.V[(x,y)] = value

    with open('value.pkl', 'wb') as fp:
        pickle.dump(agent.V, fp)
        print('dictionary saved successfully to file')

    return agent.V
    
    
def showValues(V):
    for y in range(0, 250, 50):
        print('----------------------------------------------')
        out = '| '
        for x in range(0, 250, 50):
            out += str(round(V[(x, y)], 3)).ljust(6) + ' | '
        print(out)
    print('----------------------------------------------')


def train(V):
    agent = Agent()
    agent.V = V
    env = Grid()
    # eps = 0.5
    optimizer = optim.Adam(agent.model.parameters(), lr=0.001)
    pref_l = [0]*2

    for i in range(50):
        done = False
        steps = 0
        env.reset()
        print("round", i)

        while not done:
            steps += 1
            action = agent.select_action(env.get_state(), 0.5-(i/100))
            agent.replay_buffer.append((env.get_state(), action))
            done = env.step(action, 1)

            if done:
                break

            if (steps % 5 == 0):
                skip = False
                choice = random.sample(agent.replay_buffer, 2)
                pref = env.query(choice[0], choice[1], agent.V)
                if pref == "left":
                    pref_l[0] = 1
                    pref_l[1] = 0
                elif pref == "right":
                    pref_l[0] = 0
                    pref_l[1] = 1
                elif pref == "equal":
                    pref_l[0] = 0.5
                    pref_l[1] = 0.5
                else:
                    skip = True

                agent.replay_buffer.clear()

                if not skip:
                    optimizer.zero_grad()
                    loss = reward_loss(agent.model, choice[0], choice[1], pref_l)
                    loss.backward()
                    optimizer.step()

def play_games():
    agent = Agent()
    env = Grid()
    steps = 0

    while True:
        steps += 1
        action = agent.select_action()
        agent.replay_buffer.append((env.get_state(), action))
        # print("state: ", agent.replay_buffer[-1])
        done = env.step(action)


        if done:
            break

        if (steps % 5 == 0):
            choice = random.sample(agent.replay_buffer, 2)
            # print(choice[0])
            # print(choice[1])
            pref = env.query(choice[0], choice[1])
            print(pref)
            agent.replay_buffer.clear()
            
if __name__ == '__main__':
    # V = value_iteration(lr=0.2)
    # V = distance()
    # showValues(V)

    with open('value.pkl', 'rb') as fp:
        value = pickle.load(fp)

    train(value)