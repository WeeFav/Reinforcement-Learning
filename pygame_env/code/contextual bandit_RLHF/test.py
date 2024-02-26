import pygame
from pygame.locals import *

pygame.init()
COLUMNS, ROWS, TIESIZE = 5, 5, 50

window = pygame.display.set_mode((COLUMNS*TIESIZE, ROWS*TIESIZE))

x = 0
y = 4*50
width = 50
height = 50

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == KEYDOWN:
            if ((event.key == K_DOWN) and (y + 50 <= (ROWS-1)*TIESIZE)):
                y += 50
            if ((event.key == K_UP) and (y - 50 >= 0)):
                y -= 50
            if ((event.key == K_RIGHT) and (x + 50 <= (COLUMNS-1)*TIESIZE)):
                x += 50
            if ((event.key == K_LEFT) and ((x - 50 >= 0))):
                x -= 50

    window.fill("white")
    for c in range (1, COLUMNS):
        pygame.draw.line(window, "gray", (c*TIESIZE, 0), (c*TIESIZE, window.get_height()))
    for r in range (1, ROWS):
        pygame.draw.line(window, "gray", (0, r*TIESIZE), (window.get_width(), r*TIESIZE))

    pygame.draw.rect(window, [255, 255, 55], [x, y, width, height], 0)

    pygame.display.update()
    
pygame.quit()