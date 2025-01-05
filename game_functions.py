import pygame
import sys
import tensorflow as tf
from RL_agent import get_rl_action, DQN
from game_functions import *
from config import config


cf = config() # config values


# Function to draw the Connect 4 board
def draw_board(screen, board):

    # Draw the Connect 4 board inside the background frame
    board_rect = pygame.Rect(
        (cf.WINDOW_WIDTH - cf.WIDTH * cf.CELL_SIZE) // 2,
        (cf.WINDOW_HEIGHT - (cf.HEIGHT + 2.5) * cf.CELL_SIZE) // 2,
        cf.WIDTH * cf.CELL_SIZE,
        cf.HEIGHT * cf.CELL_SIZE,
    )

    pygame.draw.rect(screen, cf.BACKGROUND_COLOR, board_rect)

    for col in range(cf.WIDTH):
        for row in range(cf.HEIGHT):
            pygame.draw.rect(
                screen,
                cf.BACKGROUND_COLOR,
                (
                    board_rect.left + col * cf.CELL_SIZE,
                    board_rect.top + (row + 1.5) * cf.CELL_SIZE,
                    cf.CELL_SIZE,
                    cf.CELL_SIZE,
                ),
            )
            pygame.draw.circle(
                screen,
                cf.GRID_COLOR,
                (
                    board_rect.left + col * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                    board_rect.top + (row + 1.5) * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                ),
                cf.CELL_SIZE // 2,
                5,
            )
            if board[row][col] == 1:
                pygame.draw.circle(
                    screen,
                    cf.RED,
                    (
                        board_rect.left + col * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                        board_rect.top + (row + 1.5) * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                    ),
                    cf.CELL_SIZE // 2 - 5,
                )
            elif board[row][col] == 2:
                pygame.draw.circle(
                    screen,
                    cf.BLUE,
                    (
                        board_rect.left + col * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                        board_rect.top + (row + 1.5) * cf.CELL_SIZE + cf.CELL_SIZE // 2,
                    ),
                    cf.CELL_SIZE // 2 - 5,
                )


# Function to drop a disc in a column
def drop_disc(board, col, player):
    for row in range(cf.HEIGHT - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return True
    return False


# Function to check for a win
def check_win(board, player):
    # Check horizontally, vertically, and diagonally
    for row in range(cf.HEIGHT):
        for col in range(cf.WIDTH - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True

    for col in range(cf.WIDTH):
        for row in range(cf.HEIGHT - 3):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    for row in range(3, cf.HEIGHT):
        for col in range(cf.WIDTH - 3):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    for row in range(cf.HEIGHT - 3):
        for col in range(cf.WIDTH - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    return False


# Function to check for a draw
def check_draw(board):
    return all(board[0][col] != 0 for col in range(cf.WIDTH))


# Function to reset the game
def reset_game():
    return [[0] * cf.WIDTH for _ in range(cf.HEIGHT)]


# Function to display the end screen
def end_screen(message, screen, player1_wins, player2_wins, draws):
    font_large = pygame.font.Font(None, 74)

    # Render white text
    text_large_white = font_large.render(message, True, (255, 0, 0))

    # Create a surface with an alpha channel
    text_large_surface = pygame.Surface(text_large_white.get_size(), pygame.SRCALPHA)

    # Render black-bordered text on the alpha surface
    text_large_black = font_large.render(message, True, (0, 0, 0))
    text_large_surface.blit(
        text_large_black, (2, 2)
    )  # Offset by (2, 2) to create a border

    # Blit the white text on top of the black-bordered text (shadow)
    text_large_surface.blit(text_large_white, (0, 0))

    rect_large = text_large_surface.get_rect(
        center=(cf.WIDTH * cf.CELL_SIZE // 2, cf.HEIGHT * cf.CELL_SIZE // 2)
    )

    # Display game statistics with a smaller font
    font_small = pygame.font.Font(None, 36)

    # Render white text for stats_text
    stats_text_white = font_small.render(
        f"Your Wins: {player1_wins} | Agent Wins: {player2_wins} | Draws: {draws}",
        True,
        (255, 255, 255),
    )

    # Create a surface with an alpha channel for stats_text
    stats_text_surface = pygame.Surface(stats_text_white.get_size(), pygame.SRCALPHA)

    # Render black-bordered text on the alpha surface for stats_text
    stats_text_black = font_small.render(
        f"Your Wins: {player1_wins} | Agent Wins: {player2_wins} | Draws: {draws}",
        True,
        (0, 0, 0),
    )
    stats_text_surface.blit(
        stats_text_black, (2, 2)
    )  # Offset by (2, 2) to create a border

    # Blit the white text on top of the black-bordered text for stats_text (shadow)
    stats_text_surface.blit(stats_text_white, (0, 0))

    # Display instructions with a smaller font
    instructions_white = font_small.render(
        "Press any key to continue", True, (255, 255, 255)
    )

    # Create a surface with an alpha channel for instructions
    instructions_surface = pygame.Surface(
        instructions_white.get_size(), pygame.SRCALPHA
    )

    # Render black-bordered text on the alpha surface for instructions
    instructions_black = font_small.render("Press any key to continue", True, (0, 0, 0))
    instructions_surface.blit(
        instructions_black, (2, 2)
    )  # Offset by (2, 2) to create a border

    # Blit the white text on top of the black-bordered text for instructions
    instructions_surface.blit(instructions_white, (0, 0))

    # Calculate the size of the background surface
    padding = 20
    background_WIDTH = (
        max(
            rect_large.height,
            stats_text_surface.get_width(),
            instructions_surface.get_width(),
        )
        + 2 * padding
    )
    background_HEIGHT = (
        rect_large.height
        + stats_text_surface.get_height()
        + instructions_surface.get_height()
        + 4 * padding
    )

    background_surface = pygame.Surface(
        (background_WIDTH, background_HEIGHT), pygame.SRCALPHA
    )
    pygame.draw.rect(
        background_surface,
        (255, 255, 255, 128),
        (0, 0, background_WIDTH, background_HEIGHT),
        border_radius=20,
    )
    pygame.draw.rect(
        background_surface,
        (255, 255, 255),
        (0, 0, background_WIDTH, background_HEIGHT),
        5,
        border_radius=20,
    )

    # Blit the text surface on the background surface
    background_surface.blit(
        text_large_surface, (background_WIDTH // 2 - rect_large.width // 2, padding)
    )

    # Blit the game statistics on the background surface
    background_surface.blit(
        stats_text_surface, (padding, rect_large.height + 2 * padding)
    )

    # Blit the instructions on the background surface, centered horizontally
    instructions_x = (background_WIDTH - instructions_surface.get_width()) // 2
    background_surface.blit(
        instructions_surface,
        (
            instructions_x,
            rect_large.height + stats_text_surface.get_height() + 3 * padding,
        ),
    )

    # Adjust the position of the background_rect to center it on the screen
    background_rect = background_surface.get_rect(
        center=(cf.WINDOW_WIDTH // 2, cf.WINDOW_HEIGHT // 2)
    )
    screen.blit(background_surface, background_rect)

    pygame.display.flip()

    pygame.time.wait(2000)  # Pause for 2 seconds before starting a new game

    waiting_for_key = True
    while waiting_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting_for_key = False