import pygame
pygame.init()

ROWS = 5
COLUMNS = 5
BLOCKSIZE = 50

world=\
"""
ww  g
ww ww
w  ww
w www
p www
"""

class Grid():
    def __init__(self):
        self.screen = pygame.display.set_mode((ROWS*BLOCKSIZE, COLUMNS*BLOCKSIZE))
        self.clock = pygame.time.Clock()
        self.FPS = 3
        self.world = world.split('\n')[1:-1]
        self.walls = pygame.sprite.Group()
        self.goals = pygame.sprite.Group()
        self.players = pygame.sprite.Group()

        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == 'w'):
                    self.walls.add(Wall(row_idx, col_idx))
                elif (block_type == 'g'):
                    self.goals.add(Goal(row_idx, col_idx))
                elif(block_type == 'p'):
                    self.players.add(Player(row_idx, col_idx))
    
    def states_to_be_queried(self):
        sq = []
        for row_idx, row in enumerate(self.world):
            for col_idx, block_type in enumerate(row):
                if (block_type == ' ' or block_type == 'p'):        
                    sq.append((row_idx, col_idx))
        
        return sq
    
    def query(self, curr_row, curr_col):
        # background
        self.screen.fill("white")
        for c in range (1, COLUMNS):
            pygame.draw.line(self.screen, "gray", (c*BLOCKSIZE, 0), (c*BLOCKSIZE, self.screen.get_height()))
        for r in range (1, ROWS):
            pygame.draw.line(self.screen, "gray", (0, r*BLOCKSIZE), (self.screen.get_width(), r*BLOCKSIZE))

        # environment
        self.walls.draw(self.screen)
        self.goals.draw(self.screen)

        p = self.players.sprites()
        p[0].rect.x = curr_col*BLOCKSIZE
        p[0].rect.y = curr_row*BLOCKSIZE
        self.players.draw(self.screen)

        pygame.display.flip()
        
        # display_text = ""
        # rank = []
        # for i in range(4):
        #     pref = self.get_input_from_human()
        #     rank.append(pref)
        #     if i != 3:
        #         display_text += action_dict[pref] + "->"
        #     else:
        #         display_text += action_dict[pref]
            
        #     font = font = pygame.font.Font('freesansbold.ttf', 20)
        #     text1 = font.render(display_text, True, [0, 0, 0], [255, 255, 255])
        #     textRect1 = text1.get_rect()
        #     textRect1.center = (125, 270)
        #     new_window.blit(text1, textRect1)
        #     pygame.display.update()
   
        pygame.time.wait(2000)

        return 0            


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
