import pygame
import sys
from pygame.locals import *
pygame.init()

ROWS = 5
COLUMNS = 5
BLOCKSIZE = 50
SPACE = 2 * BLOCKSIZE

# maze1=\
# """
# wwwwg
# wwww 
# w    
# w www
# p www
# """

maze2=\
"""
g   w
www w
w   w
w www
p www
"""

class Grid():
    def __init__(self):
        self.screen = pygame.display.set_mode((2*COLUMNS*BLOCKSIZE+SPACE, ROWS*BLOCKSIZE))
        self.left_screen = pygame.Surface((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
        self.right_screen = pygame.Surface((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
        self.mid_screen = pygame.Surface((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
        self.clock = pygame.time.Clock()
        self.FPS = 4
        self.world = maze2.split('\n')[1:-1]
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

    def reset(self, render):
        self.done = False
        self.move_player(4, 0)
        if (render):
            self.render(inital=False)
        return self.done


    def render(self, inital, row=None, col=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # update game window
        # background
        self.clock.tick(self.FPS)
        self.screen.fill("white")
        self.mid_screen.fill("white")

        for c in range (1, COLUMNS):
            pygame.draw.line(self.mid_screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.mid_screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(self.mid_screen, "gray", (0, r*BLOCKSIZE), (self.mid_screen.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(self.mid_screen)
        self.goals.draw(self.mid_screen)

        if (inital == True):
            self.move_player(row, col)

        self.players.draw(self.mid_screen)

        self.screen.blit(self.mid_screen, (175, 0))
        pygame.display.flip()

    def render_2(self, left_row, left_col, right_row, right_col, color):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # update game window
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

        self.move_left_player(left_row, left_col)
        self.left_player_color(color)
        self.left_player.draw(self.left_screen)

        self.move_right_player(right_row, right_col)
        self.right_player_color(color)
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
        player_row, player_col = self.get_state()            
        if (player_row == self.goal_row and player_col == self.goal_col):
            self.done = True
        else:
            self.done = False

        # render
        if (render):
            self.render(inital=False)

        return self.done
    
    def det_coord(self, s):
        row = (s // 5)
        col = (s % 5)
        return row, col

    def det_s(self, row, col):
        return row * 5 + col
    
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

    def left_player_color(self, color):
        p = self.left_player.sprites()
        p[0].image.fill(color)

    def right_player_color(self, color):
        p = self.right_player.sprites()
        p[0].image.fill(color)
   
    def query(self, init_row, init_col, left_traj, right_traj):
        pref = None
        # loop until get human input
        while True: 
            self.render_2(init_row, init_col, init_row, init_col, "red")
            for i in range(len(left_traj)):
                if (i % 2 == 1):
                    left_row, left_col = self.det_coord(left_traj[i]) 
                    right_row, right_col = self.det_coord(right_traj[i]) 
                    self.render_2(left_row, left_col, right_row, right_col, "yellow")

            pygame.time.wait(1000)

            # human input
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_a:
                    pref = 'left'
                elif event.type == KEYDOWN and event.key == K_d:
                    pref = 'right'
                elif event.type == KEYDOWN and event.key == K_s:
                    pref = 'equal'
                elif event.type == KEYDOWN and event.key == K_x:
                    pref = 'incomparable'
                
            if (pref is not None):
                green_box = pygame.Surface([250, 250])
                green_box.set_alpha(125)
                green_box.fill([0, 255, 0])
                red_box = pygame.Surface([250, 250])
                red_box.set_alpha(125)
                red_box.fill([255, 0, 0])
                if pref == 'left':
                    self.screen.blit(green_box, [0,0])
                elif pref == 'right':
                    self.screen.blit(green_box, [350,0])
                elif pref == 'equal':
                    self.screen.blit(green_box, [0,0])
                    self.screen.blit(green_box, [350,0])
                else:
                    self.screen.blit(red_box, [0,0])
                    self.screen.blit(red_box, [350,0])

                pygame.display.update()
                pygame.time.wait(100)

                return pref                

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
