import pygame
import sys
from pygame.locals import *
import random
import numpy as np
pygame.init()

BLOCKSIZE = 50

maze1=\
"""
wwwwwwwww
w   w   w
w     g w
w   w   w
ww www ww
w   w   w
w p     w
w   w   w
wwwwwwwww
"""

maze2=\
"""
wwwwwwwwwwwww
w   w   w   w
w         g w
w   w   w   w
ww www wwwwww
w   w   w   w
w       w   w
w   w   w   w
wwwwww wwwwww
w   w   w   w
w p         w
w   w   w   w
wwwwwwwwwwwww
"""

maze3=\
"""
wwwwww
w   gw
w ww w 
w ww w
wp   w
wwwwww 
"""

action_dict = {0:'up',
               1:'down',
               2:'left',
               3:'right'}

class Grid():
    def __init__(self, maze, show_render):
        if maze == 'maze1':
            self.world = maze1.split('\n')[1:-1]
            self.ROWS = 9
            self.COLUMNS = 9
            self.reset_pos = (6, 2)
        elif maze == 'maze2':
            self.world = maze2.split('\n')[1:-1]
        elif maze == 'maze3':
            self.world = maze3.split('\n')[1:-1]
            self.ROWS = 6
            self.COLUMNS = 6
            self.reset_pos = (4, 1)
        self.screen = pygame.display.set_mode((self.COLUMNS*BLOCKSIZE, self.ROWS*BLOCKSIZE))
        self.clock = pygame.time.Clock()
        self.FPS = 5
        self.clock.tick(self.FPS)
        self.show_render = show_render

        self.action_space = [0,1,2,3]
        self.obs_space = self.ROWS * self.COLUMNS * 3

        self.walls = pygame.sprite.Group()
        self.goals = pygame.sprite.Group()
        self.players = pygame.sprite.Group()

        self.done = False
        self.steps_taken = 0

        # building the maze
        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == 'w'):
                    self.walls.add(Wall(row_idx, col_idx))
                elif (block_type == 'g'):
                    self.goals.add(Goal(row_idx, col_idx))
                elif(block_type == 'p'):
                    self.players.add(Player(row_idx, col_idx))
    
        # since goal and wall won't change, we store their position as a variable for easier access
        self.goal_row, self.goal_col = self.get_goal_pos()
        self.wall_list = self.get_wall_pos()

    def reset(self, row=None, col=None):
        self.done = False
        self.steps_taken = 0
        if row is not None or col is not None:
            self.move_player(row, col)
        else:
            self.move_player(*self.reset_pos)

        if self.show_render:
            self.render()

        return self.get_env_state()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # update game window
        # background
        self.clock.tick(self.FPS)
        self.screen.fill("white")

        # draw lines
        for c in range (1, self.COLUMNS):
            pygame.draw.line(self.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.screen.get_height()))
        for r in range (1, self.ROWS):
            pygame.draw.line(self.screen, "gray", (0, r*BLOCKSIZE), (self.screen.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(self.screen)
        self.goals.draw(self.screen)
        self.players.draw(self.screen)

        pygame.display.flip()

    def step(self, action):
        """
        execute the action then render
        """

        # collect user input
        player_row, player_col = self.get_player_pos()

        if (action == 0): # up
            next_row = player_row - 1
            next_col = player_col
        elif (action == 1): # down
            next_row = player_row + 1
            next_col = player_col
        elif (action == 2): # left
            next_row = player_row 
            next_col = player_col - 1
        elif (action == 3): # right
            next_row = player_row
            next_col = player_col + 1

        # only move player if action is valid (not hitting walls)
        if ((next_row, next_col) not in self.wall_list):
            self.move_player(next_row, next_col)

        self.steps_taken += 1
        
        # check if game is over
        player_row, player_col = self.get_player_pos()            
        if (player_row == self.goal_row and player_col == self.goal_col):
            self.done = True
        else:
            self.done = False
        # check if truncated
        if self.steps_taken > 200 and self.done == False:
            truncated = True
        else:
            truncated = False

        # render
        if self.show_render:
            self.render()

        obs = self.get_env_state()
        reward = self.get_reward(next_row, next_col)
        return obs, reward, self.done, truncated
    
    def get_reward(self, next_row, next_col):
        if (next_row == self.goal_row and next_col == self.goal_col):
            reward = 1
        # elif ((next_row, next_col) in self.wall_list):
        #     reward = -1
        else:
            reward = 0
        return reward
    
    def get_env_state(self):
        # player grid
        player_state = np.zeros((self.ROWS, self.COLUMNS))
        player_row, player_col = self.get_player_pos()
        player_state[player_row][player_col] = 1

        # goal grid
        goal_state = np.zeros((self.ROWS, self.COLUMNS))
        goal_state[self.goal_row][self.goal_col] = 1

        # wall grid
        wall_state = np.zeros((self.ROWS, self.COLUMNS))
        for w in self.wall_list:
            wall_row = w[0]
            wall_col = w[1]
            wall_state[wall_row][wall_col] = 1
        
        return np.stack([player_state, goal_state, wall_state], axis=0).flatten()

    def get_player_pos(self):
        p = self.players.sprites()
        player_row = p[0].rect.y // BLOCKSIZE
        player_col = p[0].rect.x // BLOCKSIZE
        return player_row, player_col
    
    def get_goal_pos(self):
        g = self.goals.sprites()
        goal_row = g[0].rect.y // BLOCKSIZE
        goal_col = g[0].rect.x // BLOCKSIZE
        return goal_row, goal_col

    def get_wall_pos(self):
        walls_list = []
        for w in self.walls:
            wall_row = w.rect.y // BLOCKSIZE
            wall_col = w.rect.x // BLOCKSIZE
            walls_list.append((wall_row, wall_col))
        return walls_list
    
    def move_player(self, row, col):
        p = self.players.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE

    def get_valid_pos(self):
        valid_pos = []
        for row in range(1, self.ROWS):
            for col in range(1, self.COLUMNS):
                if (row, col) not in self.wall_list:
                    valid_pos.append((row, col))
        return valid_pos
    
    def get_input_from_human(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_UP:
                    return 0
                elif event.type == KEYDOWN and event.key == K_DOWN:
                    return 1
                elif event.type == KEYDOWN and event.key == K_LEFT:
                    return 2
                elif event.type == KEYDOWN and event.key == K_RIGHT:
                    return 3 

    def query(self, row, col):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        new_window = pygame.display.set_mode((self.COLUMNS*BLOCKSIZE, self.ROWS*BLOCKSIZE+50))
        # background
        self.clock.tick(self.FPS)
        new_window.fill("white")

        # draw lines
        for c in range (1, self.COLUMNS):
            pygame.draw.line(new_window, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, new_window.get_height()-50))
        for r in range (1, self.ROWS):
            pygame.draw.line(new_window, "gray", (0, r*BLOCKSIZE), (new_window.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(new_window)
        self.goals.draw(new_window)
        self.move_player(row, col)
        self.players.draw(new_window)

        pygame.display.flip()

        # text
        display_text = ""
        rank = []
        for i in range(4):
            pref = self.get_input_from_human()
            rank.append(pref)
            if i != 3:
                display_text += action_dict[pref] + "->"
            else:
                display_text += action_dict[pref]
            
            font = font = pygame.font.Font('freesansbold.ttf', 20)
            text1 = font.render(display_text, True, [0, 0, 0], [255, 255, 255])
            textRect1 = text1.get_rect()
            textRect1.center = (new_window.get_width()//2, new_window.get_height()-25)
            new_window.blit(text1, textRect1)
            pygame.display.update()
   
        pygame.time.wait(500)

        return rank, self.get_env_state()
   
class Wall(pygame.sprite.Sprite):
    def __init__(self, row, col):
        super().__init__()
        self.image = pygame.Surface((BLOCKSIZE,BLOCKSIZE))
        self.image.fill("black")
        self.rect = self.image.get_rect()
        self.rect.x = col*BLOCKSIZE
        self.rect.y = row*BLOCKSIZE

class Goal(pygame.sprite.Sprite):
    def __init__(self, row, col):
        super().__init__()
        self.image = pygame.Surface((BLOCKSIZE,BLOCKSIZE))
        self.image.fill("green")
        self.rect = self.image.get_rect()
        self.rect.x = col*BLOCKSIZE
        self.rect.y = row*BLOCKSIZE

class Player(pygame.sprite.Sprite):
    def __init__(self, row, col):
        super().__init__()
        self.image = pygame.Surface((BLOCKSIZE,BLOCKSIZE))
        self.image.fill("yellow")
        self.rect = self.image.get_rect()
        self.rect.x = col*BLOCKSIZE
        self.rect.y = row*BLOCKSIZE

    def move(self, row, col):
        self.rect.x = col
        self.rect.y = row


def human_action():
    action = None
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_UP:
            action = 0
        elif event.type == KEYDOWN and event.key == K_DOWN:
            action = 1
        elif event.type == KEYDOWN and event.key == K_LEFT:
            action = 2
        elif event.type == KEYDOWN and event.key == K_RIGHT:
            action = 3
        
    return action
        
# test to see if environment is working
if __name__ == '__main__':
    env = Grid('maze3', show_render=True)
    env.reset()

    done = False
    while True:
        # action = random.choice(env.action_space)
        action = None
        while action is None:
            action = human_action()
        new_obs, rew, done, truncated = env.step(action)
        print(truncated)

        if done:
            obs = env.reset()

    pygame.quit()
