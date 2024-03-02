import pygame
import sys
from pygame.locals import *
import random

pygame.init()


action_dict = {0:'up',
               1:'down',
               2:'left',
               3:'right'}

class Grid:
    def __init__(self):
        self.COLUMNS = 5
        self.ROWS = 5
        self.TIESIZE = 50
        self.x = 0
        self.y = 0
        self.width = 50
        self.height = 50
        self.rx = (4*self.TIESIZE)
        self.ry = (0*self.TIESIZE)
        self.window = pygame.display.set_mode((self.COLUMNS*self.TIESIZE, self.ROWS*self.TIESIZE))
        self.clock = pygame.time.Clock()
        self.reset()
        self.game_over = False

    def reset(self):
        # self.x = 0
        # self.y = 200
        self.x = random.randrange(0, 250, 50)
        self.y = random.randrange(0, 250, 50)
        self.game_over = False

        # inital game window
        self.window = pygame.display.set_mode((self.COLUMNS*self.TIESIZE, self.ROWS*self.TIESIZE))
        self.window.fill("white")
        for c in range (1, self.COLUMNS):
            pygame.draw.line(self.window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, self.window.get_height()))
        for r in range (1, self.ROWS):
            pygame.draw.line(self.window, "gray", (0, r*self.TIESIZE), (self.window.get_width(), r*self.TIESIZE))

        pygame.draw.rect(self.window, [255, 255, 55], [self.x, self.y, self.width, self.height], 0)
        pygame.draw.rect(self.window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)

        
        pygame.display.update()
        self.clock.tick(1)

    def step(self, action, tick):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # collect user input
        if ((action == 0) and (self.y - 50 >= 0)): # up
            self.y -= 50
        elif ((action == 1) and (self.y + 50 <= (self.ROWS-1)*self.TIESIZE)): # down
            self.y += 50
        elif ((action == 2) and (self.x - 50 >= 0)): # left
            self.x -= 50
        elif ((action == 3) and (self.x + 50 <= (self.COLUMNS-1)*self.TIESIZE)): # right
            self.x += 50

        # check if game is over
        if (self.x == self.rx and self.y == self.ry):
            self.game_over = True

        # update game window
        self.window = pygame.display.set_mode((self.COLUMNS*self.TIESIZE, self.ROWS*self.TIESIZE))
        self.window.fill("white")
        for c in range (1, self.COLUMNS):
            pygame.draw.line(self.window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, 250))
        for r in range (1, self.ROWS):
            pygame.draw.line(self.window, "gray", (0, r*self.TIESIZE), (250, r*self.TIESIZE))

        pygame.draw.rect(self.window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)
        pygame.draw.rect(self.window, [255, 255, 55], [self.x, self.y, self.width, self.height], 0)

        pygame.display.update()
        self.clock.tick(tick)

        return self.game_over

    def get_state(self):
        return (self.x, self.y)

    def query(self, curr_x, curr_y):
        new_window = pygame.display.set_mode((self.COLUMNS*self.TIESIZE, self.ROWS*self.TIESIZE+50))
        new_window.fill("white")
        for c in range (1, self.COLUMNS):
            pygame.draw.line(new_window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, 250))
        for r in range (1, self.ROWS+1):
            pygame.draw.line(new_window, "gray", (0, r*self.TIESIZE), (250, r*self.TIESIZE))

        pygame.draw.rect(new_window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)
        pygame.draw.rect(new_window, [255, 255, 55], [curr_x, curr_y, self.width, self.height], 0)

        pygame.display.update()
        
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
        self.clock.tick(3)


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
    
    def play_games(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == KEYDOWN:
                if ((event.key == K_DOWN) and (self.y + 50 <= (self.COLUMNS-1)*self.TIESIZE)):
                    self.y += 50
                if ((event.key == K_UP) and (self.y - 50 >= 0)):
                    self.y -= 50
                if ((event.key == K_RIGHT) and (self.x + 50 <= (self.ROWS-1)*self.TIESIZE)):
                    self.x += 50
                if ((event.key == K_LEFT) and ((self.x - 50 >= 0))):
                    self.x -= 50
                

        self.window.fill("white")
        for c in range (1, self.COLUMNS):
            pygame.draw.line(self.window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, self.window.get_height()))
        for r in range (1, self.ROWS):
            pygame.draw.line(self.window, "gray", (0, r*self.TIESIZE), (self.window.get_width(), r*self.TIESIZE))

        pygame.draw.rect(self.window, [255, 255, 55], [self.x, self.y, self.width, self.height], 0)
        pygame.draw.rect(self.window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)

        pygame.display.update()
        

                             
