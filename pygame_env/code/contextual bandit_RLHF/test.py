import pygame

# Initialize Pygame
pygame.init()

# Set the dimensions of the game window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Set the dimensions of each player's screen
SCREEN_WIDTH = WINDOW_WIDTH // 2  # Half of the window width
SCREEN_HEIGHT = WINDOW_HEIGHT

# Define colors
WHITE = (255, 255, 255)

# Create the game window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# Create surfaces for each player's screen
player1_screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
player2_screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill player screens with white color
    player1_screen.fill(WHITE)
    player2_screen.fill(WHITE)

    # Draw something on each player's screen (just as a demonstration)
    pygame.draw.rect(player1_screen, (255, 0, 0), (50, 50, 100, 100))  # Red rectangle on player 1's screen
    pygame.draw.rect(player2_screen, (0, 0, 255), (50, 50, 100, 100))  # Blue rectangle on player 2's screen

    # Blit (draw) each player's screen onto the main window
    window.blit(player1_screen, (0, 0))  # Player 1's screen is on the left
    window.blit(player2_screen, (SCREEN_WIDTH, 0))  # Player 2's screen is on the right

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()