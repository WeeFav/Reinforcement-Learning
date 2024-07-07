import pygame
import sys
from pygame.locals import *

pygame.init()

ROWS = 5
COLUMNS = 5
BLOCKSIZE = 50
SPACE = 2 * BLOCKSIZE


action_dict = {0:'up',
               1:'down',
               2:'left',
               3:'right'}

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

class Grid:
    def __init__(self):
        self.screen = pygame.display.set_mode((COLUMNS*BLOCKSIZE, ROWS*BLOCKSIZE))
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

    def move_player(self, row, col):
        p = self.players.sprites()
        p[0].rect.y = row * BLOCKSIZE
        p[0].rect.x = col * BLOCKSIZE

    def query(self, row, col, a0, a1):
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
        self.move_player(row, col)
        self.players.draw(new_window)
        
        display_text = f"{action_dict[a0]}  |  {action_dict[a1]}"
        font = font = pygame.font.Font('freesansbold.ttf', 20)
        text1 = font.render(display_text, True, [0, 0, 0], [255, 255, 255])
        textRect1 = text1.get_rect()
        textRect1.center = (125, 270)
        new_window.blit(text1, textRect1)
        
        pygame.display.flip()
        pref = self.get_input_from_human()

        pygame.display.update()
        pygame.time.wait(100)

        return pref            
    
    def get_input_from_human(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_LEFT:
                    return 'left'
                elif event.type == KEYDOWN and event.key == K_RIGHT:
                    return 'right'
                elif event.type == KEYDOWN and event.key == K_s:
                    return 'equal'
                elif event.type == KEYDOWN and event.key == K_x:
                    return 'incomparable' 

    def get_input_from_robot(self, choice1, choice2, V):
        curr_V = []
        next_V = []
        for choice in [choice1, choice2]:
            curr_x = choice[0][0]
            curr_y = choice[0][1]
            curr_V.append(V[(curr_x, curr_y)])
            next_x = curr_x
            next_y = curr_y
            action = choice[1]

            if ((action == 0) and (curr_y - 50 >= 0)): # up
                next_y -= 50
            elif ((action == 1) and (curr_y + 50 <= (self.COLUMNS-1)*self.TIESIZE)): # down
                next_y += 50
            elif ((action == 2) and (curr_x - 50 >= 0)): # left
                next_x -= 50
            elif ((action == 3) and (curr_x + 50 <= (self.ROWS-1)*self.TIESIZE)): # right
                next_x += 50

            next_V.append(V[(next_x, next_y)])

        
        if ((next_V[0] - curr_V[0] > 0) and (next_V[1] - curr_V[1] <= 0)):
            pref = 'left'
        elif ((next_V[0] - curr_V[0] <= 0) and (next_V[1] - curr_V[1] > 0)):
            pref = 'right' 
        elif ((next_V[0] - curr_V[0] > 0) and (next_V[1] - curr_V[1] > 0)):
            pref = 'equal'
        else:
            pref = 'incomparable'

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
        action = None
        env.clock.tick(env.FPS)
        env.screen.fill("white")
        for c in range (1, COLUMNS):
            pygame.draw.line(env.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, env.screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(env.screen, "gray", (0, r*BLOCKSIZE), (env.screen.get_width(), r*BLOCKSIZE))

        while (action is None):
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

        if (action in env.allowed_actions(*(env.get_state()))):
            env.step(action)      

        # environment
        env.walls.draw(env.screen)
        env.goals.draw(env.screen)
        env.players.draw(env.screen)
        
        pygame.display.flip()

    pygame.quit()

                             
