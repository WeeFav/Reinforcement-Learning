import pygame
import sys
from pygame.locals import *
pygame.init()

ROWS = 5
COLUMNS = 5
BLOCKSIZE = 50
SPACE = 2 * BLOCKSIZE

world=\
"""
wwwwg
wwww 
w    
w www
p www
"""

action_dict = {0:'up',
               1:'down',
               2:'left',
               3:'right'}

class Grid():
    def __init__(self):
        self.screen = pygame.display.set_mode((2*COLUMNS*BLOCKSIZE+SPACE, ROWS*BLOCKSIZE))
        self.left_screen = pygame.Surface((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
        self.right_screen = pygame.Surface((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
        self.clock = pygame.time.Clock()
        self.FPS = 3
        self.world = world.split('\n')[1:-1]
        self.walls = pygame.sprite.Group()
        self.goals = pygame.sprite.Group()
        self.players = pygame.sprite.Group()

        self.left_player = pygame.sprite.Group()
        self.left_player.add(Player(0,0))
        self.right_player = pygame.sprite.Group()
        self.right_player.add(Player(0,0))

        self.done = False
        self.clock.tick(self.FPS)

        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == 'w'):
                    self.walls.add(Wall(row_idx, col_idx))
                elif (block_type == 'g'):
                    self.goals.add(Goal(row_idx, col_idx))
                elif(block_type == 'p'):
                    self.players.add(Player(row_idx, col_idx))
    
        g = self.goals.sprites()
        self.goal_row = g[0].rect.y // BLOCKSIZE
        self.goal_col = g[0].rect.x // BLOCKSIZE

    def reset(self):
        self.done = False
        self.show_inital(4, 0)


    def show_inital(self, row, col):
        # inital game window
        # background
        self.clock.tick(self.FPS)
        self.screen.fill("white")
        for c in range (1, COLUMNS):
            pygame.draw.line(self.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(self.screen, "gray", (0, r*BLOCKSIZE), (self.screen.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(self.screen)
        self.goals.draw(self.screen)
        self.move_player(row, col)
        self.players.draw(self.screen)

        pygame.display.flip()

    def show_inital_2(self, row, col):
        # inital game window
        # background
        self.clock.tick(self.FPS)
        self.screen.fill("white")
        self.left_screen.fill("white")
        self.right_screen.fill("white")

        for c in range (1, COLUMNS):
            pygame.draw.line(self.left_screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.left_screen.get_height()))
            pygame.draw.line(self.right_screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.right_screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(self.left_screen, "gray", (0, r*BLOCKSIZE), (self.left_screen.get_width(), r*BLOCKSIZE))
            pygame.draw.line(self.right_screen, "gray", (0, r*BLOCKSIZE), (self.right_screen.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(self.left_screen)
        self.walls.draw(self.right_screen)
        self.goals.draw(self.left_screen)
        self.goals.draw(self.right_screen)
        self.move_left_player(row, col)
        self.left_player.draw(self.left_screen)
        self.move_right_player(row, col)
        self.right_player.draw(self.right_screen)

        self.screen.blit(self.left_screen, (0,0))
        self.screen.blit(self.right_screen, (COLUMNS*BLOCKSIZE+SPACE,0))

        pygame.display.flip()

    def step(self, action, render):
        # collect user input
        player_row, player_col = self.get_state()            
        if ((action == 0) and (player_row > 0)): # up
            self.move_player(player_row - 1, player_col)
        elif ((action == 1) and (player_row < 4)): # down
            self.move_player(player_row + 1, player_col)
        elif ((action == 2) and (player_col > 0)): # left
            self.move_player(player_row, player_col - 1)
        elif ((action == 3) and (player_col < 4)): # right
            self.move_player(player_row, player_col + 1)

        # check if game is over
        if (player_row == self.goal_row and player_col == self.goal_col):
            self.done = True

        # render
        if (render):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    
            # update game window
            # background
            self.clock.tick(self.FPS)
            self.screen.fill("white")
            for c in range (1, COLUMNS):
                pygame.draw.line(self.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.screen.get_height()))
            for r in range (1, ROWS):
                pygame.draw.line(self.screen, "gray", (0, r*BLOCKSIZE), (self.screen.get_width(), r*BLOCKSIZE))

            # environment
            self.walls.draw(self.screen)
            self.goals.draw(self.screen)
            self.players.draw(self.screen)
            pygame.display.flip()

        return self.done
    
    def step_2(self, left_action, right_action, render):
        # left action
        player_row, player_col = self.get_left_state()            
        if ((left_action == 0) and (player_row > 0)): # up
            self.move_left_player(player_row - 1, player_col)
        elif ((left_action == 1) and (player_row < 4)): # down
            self.move_left_player(player_row + 1, player_col)
        elif ((left_action == 2) and (player_col > 0)): # left
            self.move_left_player(player_row, player_col - 1)
        elif ((left_action == 3) and (player_col < 4)): # right
            self.move_left_player(player_row, player_col + 1)

        # right action
        player_row, player_col = self.get_right_state()            
        if ((right_action == 0) and (player_row > 0)): # up
            self.move_left_player(player_row - 1, player_col)
        elif ((right_action == 1) and (player_row < 4)): # down
            self.move_left_player(player_row + 1, player_col)
        elif ((right_action == 2) and (player_col > 0)): # left
            self.move_left_player(player_row, player_col - 1)
        elif ((right_action == 3) and (player_col < 4)): # right
            self.move_left_player(player_row, player_col + 1)

        # render
        if (render):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    
            # inital game window
            # background
            self.clock.tick(self.FPS)
            self.screen.fill("white")
            self.left_screen.fill("white")
            self.right_screen.fill("white")

            for c in range (1, COLUMNS):
                pygame.draw.line(self.left_screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.left_screen.get_height()))
                pygame.draw.line(self.right_screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.right_screen.get_height()))
            for r in range (1, ROWS):
                pygame.draw.line(self.left_screen, "gray", (0, r*BLOCKSIZE), (self.left_screen.get_width(), r*BLOCKSIZE))
                pygame.draw.line(self.right_screen, "gray", (0, r*BLOCKSIZE), (self.right_screen.get_width(), r*BLOCKSIZE))

            # environment
            self.walls.draw(self.left_screen)
            self.walls.draw(self.right_screen)
            self.goals.draw(self.left_screen)
            self.goals.draw(self.right_screen)
            self.left_player.draw(self.left_screen)
            self.right_player.draw(self.right_screen)

            self.screen.blit(self.left_screen, (0,0))
            self.screen.blit(self.right_screen, (COLUMNS*BLOCKSIZE+SPACE,0))

            pygame.display.flip()

    def states_to_be_queried(self):
        sq = []
        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == ' ' or block_type == 'p'):        
                    sq.append((row_idx, col_idx))
        
        return sq
    
    def allowed_actions(self, row, col):
        r = []
        for a in range(4):
            next_row = row
            next_col = col
            allowed = True
            if (a == 0 and row > 0): # up
                next_row = row - 1
            elif (a == 1 and row < 4): # down
                next_row = row + 1
            elif (a == 2 and col > 0): # left
                next_col = col - 1
            elif (a == 3 and col < 4): # right
                next_col = col + 1
            else:
                continue
                
            for w in self.walls:
                w_row = w.rect.y // BLOCKSIZE
                w_col = w.rect.x // BLOCKSIZE
                if (next_row == w_row and next_col == w_col):
                    allowed = False
                    break
            
            if (allowed):
                r.append(a)
        
        return r

    def get_state(self):
        p = self.players.sprites()
        row = p[0].rect.y // BLOCKSIZE
        col = p[0].rect.x // BLOCKSIZE
        return row, col
    
    def get_left_state(self):
        p = self.left_player.sprites()
        row = p[0].rect.y // BLOCKSIZE
        col = p[0].rect.x // BLOCKSIZE
        return row, col
    
    def get_right_state(self):
        p = self.right_player.sprites()
        row = p[0].rect.y // BLOCKSIZE
        col = p[0].rect.x // BLOCKSIZE
        return row, col

    def move_player(self, row, col):
        p = self.players.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE

    def move_left_player(self, row, col):
        p = self.left_player.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE

    def move_right_player(self, row, col):
        p = self.right_player.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE
    
    def query(self, init_row, init_col, traj1, traj2):
        self.show_inital_2(init_row, init_col)
        for i in range(len(traj1)):
            if (i % 2 == 0):
                self.step_2(traj1[i], traj2[i], render=True)
             
   
        pygame.time.wait(2000)

        # return pref            

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


if __name__ == '__main__':
    env = Grid()

    run = True
    while run:
        # background
        env.clock.tick(env.FPS)
        env.screen.fill("white")
        for c in range (1, COLUMNS):
            pygame.draw.line(env.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, env.screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(env.screen, "gray", (0, r*BLOCKSIZE), (env.screen.get_width(), r*BLOCKSIZE))

        # environment
        env.walls.draw(env.screen)
        env.goals.draw(env.screen)
        env.players.draw(env.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        pygame.display.flip()

    pygame.quit()
