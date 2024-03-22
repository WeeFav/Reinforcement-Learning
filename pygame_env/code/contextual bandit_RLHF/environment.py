import pygame
import sys
from pygame.locals import *
pygame.init()

ROWS = 5
COLUMNS = 5
BLOCKSIZE = 50

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
        self.screen = pygame.display.set_mode((ROWS*BLOCKSIZE, COLUMNS*BLOCKSIZE))
        self.clock = pygame.time.Clock()
        self.FPS = 3
        self.world = world.split('\n')[1:-1]
        self.walls = pygame.sprite.Group()
        self.goals = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
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
        self.move_player(4,0)
        self.players.draw(self.screen)

        pygame.display.flip()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

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
    
    def states_to_be_queried(self):
        sq = []
        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == ' ' or block_type == 'p'):        
                    sq.append((row_idx, col_idx))
        
        return sq
    
    def get_state(self):
        p = self.players.sprites()
        row = p[0].rect.y // BLOCKSIZE
        col = p[0].rect.x // BLOCKSIZE
        return row, col

    def move_player(self, row, col):
        p = self.players.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE
    
    def query(self, curr_row, curr_col):
        # background
        new_window = pygame.display.set_mode((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE+50))
        new_window.fill("white")
        for c in range (1, COLUMNS):
            pygame.draw.line(new_window, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, new_window.get_height()-50))
        for r in range (1, ROWS):
            pygame.draw.line(new_window, "gray", (0, r*BLOCKSIZE), (new_window.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(new_window)
        self.goals.draw(new_window)
        self.move_player(curr_row, curr_col)
        self.players.draw(new_window)

        pygame.display.flip()
        
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
            textRect1.center = (125, 270)
            new_window.blit(text1, textRect1)
            pygame.display.update()
   
        pygame.time.wait(2000)

        return rank            

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
