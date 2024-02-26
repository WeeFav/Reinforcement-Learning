import pygame
import sys
from pygame.locals import *

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
        self.x = 0
        self.y = (self.ROWS-1)*self.TIESIZE
        self.frame_iteration = 0
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
        pygame.draw.rect(self.window, [0, 0, 0], [0, 0, 100, 100], 0)

        
        pygame.display.update()
        self.clock.tick(5)

    def step(self, action, tick):
        self.frame_iteration += 1
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
            pygame.draw.line(self.window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, self.window.get_height()))
        for r in range (1, self.ROWS):
            pygame.draw.line(self.window, "gray", (0, r*self.TIESIZE), (self.window.get_width(), r*self.TIESIZE))

        pygame.draw.rect(self.window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)
        pygame.draw.rect(self.window, [0, 0, 0], [0, 0, 100, 100], 0)
        pygame.draw.rect(self.window, [255, 255, 55], [self.x, self.y, self.width, self.height], 0)
        
        pygame.display.update()
        self.clock.tick(tick)

        return self.game_over

    def get_state(self):
        return (self.x, self.y)

    def query(self, choice1, choice2, V=None):
        new_window = pygame.display.set_mode((self.COLUMNS*self.TIESIZE*2+(2*self.TIESIZE), self.ROWS*self.TIESIZE+50))
        new_window.fill("white")
        for c in range (1, self.COLUMNS*2+2):
            if (c <= self.COLUMNS or c >= self.COLUMNS+2):
                pygame.draw.line(new_window, "gray", (c*self.TIESIZE, 0), (c*self.TIESIZE, 250))
        for r in range (1, self.ROWS+1):
            pygame.draw.line(new_window, "gray", (0, r*self.TIESIZE), (250, r*self.TIESIZE))
            pygame.draw.line(new_window, "gray", (350, r*self.TIESIZE), (new_window.get_width(), r*self.TIESIZE))

        font = font = pygame.font.Font('freesansbold.ttf', 32)
        text1 = font.render(action_dict[choice1[1]], True, [0, 0, 0], [255, 255, 255])
        text2 = font.render(action_dict[choice2[1]], True, [0, 0, 0], [255, 255, 255])

        textRect1 = text1.get_rect()
        textRect2 = text2.get_rect()

        textRect1.center = (125, 270)
        textRect2.center = (475, 270)

        new_window.blit(text1, textRect1)
        new_window.blit(text2, textRect2)
        
        pygame.draw.rect(new_window, [0, 0, 0], [0, 0, 100, 100], 0)
        pygame.draw.rect(new_window, [0, 0, 0], [0+350, 0, 100, 100], 0)
        pygame.draw.rect(new_window, [255, 0, 0], [self.rx, self.ry, self.width, self.height], 0)
        pygame.draw.rect(new_window, [255, 0, 0], [self.rx+350, self.ry, self.width, self.height], 0)
        pygame.draw.rect(new_window, [255, 255, 55], [choice1[0][0], choice1[0][1], self.width, self.height], 0)
        pygame.draw.rect(new_window, [255, 255, 55], [choice2[0][0]+350, choice2[0][1], self.width, self.height], 0)

        pygame.display.update()

        pygame.time.wait(200)
        pref = self.get_input_from_human()

        green_box = pygame.Surface([250, 250])
        green_box.set_alpha(125)
        green_box.fill([0, 255, 0])
        red_box = pygame.Surface([250, 250])
        red_box.set_alpha(125)
        red_box.fill([255, 0, 0])
        if pref == 'left':
            new_window.blit(green_box, [0,0])
        elif pref == 'right':
            new_window.blit(green_box, [350,0])
        elif pref == 'equal':
            new_window.blit(green_box, [0,0])
            new_window.blit(green_box, [350,0])
        else:
            new_window.blit(red_box, [0,0])
            new_window.blit(red_box, [350,0])

        pygame.display.update()
        pygame.time.wait(200)

        return pref            
    
    def get_input_from_human(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_a:
                    return 'left'
                elif event.type == KEYDOWN and event.key == K_d:
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
        

                             
