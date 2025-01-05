import pygame
import sys
import tensorflow as tf
import random
from RL_agent import get_rl_action, DQN
from game_functions import *


# Initialize pygame
pygame.init()


# Main game loop
def main():
    model = DQN()
    model = DQN(num_actions=cf.WIDTH)
    model.build([(None, 2, cf.HEIGHT, cf.WIDTH), (None, 5)])
    model.compile(optimizer="adam", loss="mse")

    model.load_weights("./checkpoints/my_checkpoint")
    screen = pygame.display.set_mode((cf.WINDOW_WIDTH, cf.WINDOW_HEIGHT))
    pygame.display.set_caption("Connect 4")

    clock = pygame.time.Clock()

    board = reset_game()
    current_player = 2

    # Initialize game statistics
    your_wins = 0
    rl_agent_wins = 0
    draws = 0

    # Height of the space below the game board for displaying stats
    stats_height = 80

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and current_player == cf.HUMAN_PLAYER
            ):
                column = (
                    event.pos[0] // cf.CELL_SIZE - 1
                )  # Column where to set the coin
                if 0 <= column < cf.WIDTH and board[0][column] == 0:
                    if drop_disc(board, column, current_player):
                        if check_win(board, current_player):
                            your_wins += 1
                            draw_board(screen, board)
                            end_screen(
                                "You Win!", screen, your_wins, rl_agent_wins, draws
                            )
                            board = reset_game()
                        elif check_draw(board):
                            draws += 1
                            draw_board(screen, board)
                            end_screen(
                                "It's a Draw!", screen, your_wins, rl_agent_wins, draws
                            )
                            board = reset_game()
                        else:
                            current_player = 3 - current_player  # Switch players
            # Add logic for RL agent's move
            if current_player == cf.RL_PLAYER:
                # Get the RL agent's action
                rl_action = get_rl_action(board, model)
                print(rl_action)
                # Update the board based on the RL agent's move
                if drop_disc(board, rl_action, current_player):
                    if check_win(board, current_player):
                        rl_agent_wins += 1
                        draw_board(screen, board)
                        end_screen(
                            "RL-Agent Wins!", screen, your_wins, rl_agent_wins, draws
                        )
                        board = reset_game()
                    elif check_draw(board):
                        draws += 1
                        draw_board(screen, board)
                        end_screen(
                            "It's a Draw!", screen, your_wins, rl_agent_wins, draws
                        )
                        board = reset_game()
                    else:
                        current_player = 3 - current_player  # Switch players
                elif drop_disc(
                    board, random.randint(0, cf.WIDTH - 1), current_player
                ):  # Make random action if agent finds no action
                    if check_win(board, current_player):
                        rl_agent_wins += 1
                        draw_board(screen, board)
                        end_screen(
                            "RL-Agent Wins!", screen, your_wins, rl_agent_wins, draws
                        )
                        board = reset_game()
                    elif check_draw(board):
                        draws += 1
                        draw_board(screen, board)
                        end_screen(
                            "It's a Draw!", screen, your_wins, rl_agent_wins, draws
                        )
                        board = reset_game()
                    else:
                        current_player = 3 - current_player  # Switch players

        screen.fill(cf.BACKGROUND_COLOR)
        # frame_rect = pygame.Rect(0, 0, cf.WINDOW_WIDTH, cf.WINDOW_HEIGHT)
        # pygame.draw.rect(screen, cf.GRID_COLOR, frame_rect, border_radius=10)
        draw_board(screen, board)

        # Display current player
        pygame.draw.circle(
            screen,
            cf.RED if current_player == 1 else cf.BLUE,
            (
                (cf.WINDOW_WIDTH - cf.WIDTH * cf.CELL_SIZE) // 2
                + cf.WIDTH * cf.CELL_SIZE // 2,
                cf.CELL_SIZE // 2 + 25,
            ),
            cf.CELL_SIZE // 2 - 5,
        )

        # Draw stats background
        stats_background_rect = pygame.Rect(
            0, cf.WINDOW_HEIGHT - stats_height, cf.WINDOW_WIDTH, stats_height
        )
        pygame.draw.rect(screen, cf.GRID_COLOR, stats_background_rect)

        # Display game statistics
        font_stats = pygame.font.Font(None, 36)
        your_wins_text = font_stats.render(
            f"Your Wins: {your_wins}", True, (255, 255, 255)
        )  # White text
        rl_agent_wins_text = font_stats.render(
            f"Agent Wins: {rl_agent_wins}", True, (255, 255, 255)
        )  # White text
        draws_text = font_stats.render(
            f"Draws: {draws}", True, (255, 255, 255)
        )  # White text

        rect_your_wins = your_wins_text.get_rect(
            center=(cf.WINDOW_WIDTH // 4, cf.WINDOW_HEIGHT - stats_height // 2)
        )
        rect_rl_agent_wins = rl_agent_wins_text.get_rect(
            center=(3 * cf.WINDOW_WIDTH // 4, cf.WINDOW_HEIGHT - stats_height // 2)
        )
        rect_draws = draws_text.get_rect(
            center=(cf.WINDOW_WIDTH // 2, cf.WINDOW_HEIGHT - stats_height // 2)
        )

        screen.blit(your_wins_text, rect_your_wins)
        screen.blit(rl_agent_wins_text, rect_rl_agent_wins)
        screen.blit(draws_text, rect_draws)

        pygame.display.flip()
        clock.tick(cf.FPS)


if __name__ == "__main__":
    main()
