import tensorflow as tf
import numpy as np
import random
import pathlib
import pygame
import pandas as pd

import os
from datetime import datetime
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate
from config import config

config_values = config()


# Deep Q Network (DQN) Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions=config_values.WIDTH):
        super(DQN, self).__init__()
        # Layers for processing the game field
        self.conv1 = Conv2D(
            32, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.flatten = Flatten()

        # Layers for processing flags
        self.flag_fc = Dense(64, activation="relu")  # Adjust the size as needed
        self.flag_output = Dense(1, activation="sigmoid")

        # Common dense layers
        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(num_actions)

    def call(self, inputs):
        # Separate game field and flags
        game_field = inputs[0]
        flags = inputs[1]

        # Process game field
        x = self.conv1(game_field)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        # Process flags separately
        flags_output = self.flag_fc(flags)
        flags_output = self.flag_output(flags_output)

        # Concatenate the processed game field and flags
        x = tf.concat([x, flags_output], axis=-1)

        # Common dense layers
        output = self.fc2(x)
        output.set_shape((None, NUM_ACTIONS))  # Adjust NUM_ACTIONS as needed
        return output

    def set_custom_weights(self, weights):
        # Set custom weights for each layer
        self.conv1.set_weights(weights[0:2])
        self.conv2.set_weights(weights[2:4])
        self.conv3.set_weights(weights[4:6])
        self.fc1.set_weights(weights[6:8])
        self.fc2.set_weights(weights[8:10])


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = pd.DataFrame(
            columns=[
                "index",  # Add an 'index' column to store unique identifiers for each sample
                "state",
                "action",
                "next_state",
                "reward",
                "game_terminated_flag",
                "opponent_won_flag",
                "agent_won_flag",
                "illegal_agent_move_flag",
                "board_full_flag",
                "loss",  # Add a 'loss' column to track the loss for each sample
            ]
        )
        self.position = 0
        self.next_index = 0  # To assign unique indices to new samples

    def push(
        self,
        state,
        action,
        next_state,
        reward,
        game_terminated_flag,
        opponent_won_flag,
        agent_won_flag,
        illegal_agent_move_flag,
        board_full_flag,
        loss,
    ):
        row = pd.DataFrame(
            [
                (
                    self.next_index,
                    state,
                    action,
                    next_state,
                    reward,
                    game_terminated_flag,
                    opponent_won_flag,
                    agent_won_flag,
                    illegal_agent_move_flag,
                    board_full_flag,
                    loss,  # Include the loss value
                )
            ],
            columns=self.memory.columns,
        )

        if len(self.memory) < self.capacity:
            self.memory = pd.concat([self.memory, row], ignore_index=True)
        else:
            self.memory.loc[self.position] = row.iloc[0]

        self.position = (self.position + 1) % self.capacity
        self.next_index += 1

    def update_loss(self, indices, new_loss):
        # Update the loss for the sample with the specified index
        for i, index in enumerate(indices):
            self.memory.loc[self.memory["index"] == index, "loss"] = new_loss[i]

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in the replay buffer")

        # Assuming you have a 'loss' column in your DataFrame indicating the current loss
        sorted_memory = self.memory.sort_values(by="loss", ascending=False)

        # Take the top batch_size samples with the highest loss
        selected_samples = sorted_memory.head(batch_size)

        # Convert the DataFrame back to a list of tuples for compatibility with your original code
        selected_samples_list = [tuple(row) for row in selected_samples.values]

        return selected_samples_list


def draw_board(screen, board):
    for col in range(config_values.WIDTH):
        for row in range(config_values.HEIGHT):
            pygame.draw.rect(
                screen,
                config_values.BACKGROUND_COLOR,
                (
                    col * config_values.CELL_SIZE,
                    (row + 1.5) * config_values.CELL_SIZE,
                    config_values.CELL_SIZE,
                    config_values.CELL_SIZE,
                ),  # Angepasste Y-Koordinate
            )
            pygame.draw.circle(
                screen,
                config_values.GRID_COLOR,
                (
                    col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                    (row + 1.5) * config_values.CELL_SIZE
                    + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                ),
                config_values.CELL_SIZE // 2,
                5,
            )  # Draw grid circles
            if board[0][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    config_values.RED,
                    (
                        col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                        (row + 1.5) * config_values.CELL_SIZE
                        + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    config_values.CELL_SIZE // 2 - 5,
                )
            elif board[1][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    config_values.BLUE,
                    (
                        col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                        (row + 1.5) * config_values.CELL_SIZE
                        + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    config_values.CELL_SIZE // 2 - 5,
                )


def visualize_training(screen, q_values, random_action, action, reward, opponent):
    q_values = q_values[0]
    for col, value in enumerate(q_values):
        # Draw the bar for each column
        if value.numpy() == np.max(q_values.numpy()):
            color = (255, 0, 0)
        else:
            color = (10, 10, 255)
        pygame.draw.rect(
            screen,
            color,
            (
                col * config_values.CELL_SIZE + 0.25 * config_values.CELL_SIZE,
                config_values.WINDOW_HEIGHT - int(value.numpy() * 200),
                config_values.CELL_SIZE * 0.5,
                int(value.numpy() * 200),
            ),
        )

        # Display Q-values on the bars with 1 digit after the comma
        font_q_values = pygame.font.Font(None, 20)
        q_value_text = font_q_values.render(f"{value:.3f}", True, (255, 255, 255))
        screen.blit(
            q_value_text,
            (col * config_values.CELL_SIZE, config_values.WINDOW_HEIGHT - 20),
        )
    # display reward of state
    font_reward = pygame.font.Font(None, 36)
    reward_text = font_reward.render(f"Reward:{reward:.3f}", True, (255, 255, 255))
    screen.blit(reward_text, (screen.get_width() - reward_text.get_width(), 40))

    # Display information about the chosen action in the top right corner
    font_action = pygame.font.Font(None, 36)
    action_text = font_action.render(
        f"Chosen Action: {action}{' (Random)' if random_action else ''}",
        True,
        (255, 255, 255),
    )
    screen.blit(action_text, (screen.get_width() - action_text.get_width(), 0))

    # Display which opponent
    font_action = pygame.font.Font(None, 36)
    opponent_text = font_action.render(
        f"Opponent: " + opponent,
        True,
        (255, 255, 255),
    )
    screen.blit(opponent_text, (screen.get_width() - opponent_text.get_width(), 80))


# Constants
NUM_ACTIONS = config_values.WIDTH
STATE_SHAPE = (
    2,
    config_values.HEIGHT,
    config_values.WIDTH,
)  # 2 channels for current player, opponent


# Epsilon-Greedy Exploration
def epsilon_greedy_action(state, epsilon, model):
    q_values = model([state, np.expand_dims(np.zeros(5), axis=0)])
    if np.random.rand() < epsilon:
        random_move = True
        return np.random.randint(NUM_ACTIONS), q_values, random_move  # Explore
    else:
        random_move = False
        return np.argmax(q_values), q_values, random_move  # Exploit


# Function to check for a winning move


def check_win(board):
    players, rows, cols = board.shape

    # Check for a win in rows
    for player in range(players):
        for row in range(rows):
            for col in range(cols - 3):
                if np.all(board[player, row, col : col + 4] == 1):
                    return True

    # Check for a win in columns
    for player in range(players):
        for col in range(cols):
            for row in range(rows - 3):
                if np.all(board[player, row : row + 4, col] == 1):
                    return True

    # Check for a win in diagonals (from bottom-left to top-right)
    for player in range(players):
        for row in range(3, rows):
            for col in range(cols - 3):
                if np.all(board[player, row - np.arange(4), col + np.arange(4)] == 1):
                    return True

    # Check for a win in diagonals (from top-left to bottom-right)
    for player in range(players):
        for row in range(rows - 3):
            for col in range(cols - 3):
                if np.all(board[player, row + np.arange(4), col + np.arange(4)] == 1):
                    return True

    return False


def is_blocking_opponent(board, action_column):
    # Copy the board to simulate the effect of placing a disc in the specified column
    temp_board = np.copy(board)

    # Find the empty row in the specified column
    empty_row = next_empty_row(temp_board, action_column)

    if empty_row is None:
        # The column is full, and placing a disc is not possible
        return False

    # Place a disc in the specified column
    temp_board[1, empty_row, action_column] = 1

    # Check if this move blocks the opponent from connecting four discs
    return check_win(temp_board)


def next_empty_row(board, action):
    try:
        for row in range(config_values.HEIGHT):
            if board[0, row, action] == 0 and board[1, row, action] == 0:
                next = row
                break
            else:
                continue
        return next
    except:
        pass


# Function to calculate the reward
def calculate_reward(board, action, current_player):
    # Default reward
    reward = 0

    # check if board has free spaces (not necessary but doesn't hurt)
    if np.sum(board) < config_values.HEIGHT * config_values.WIDTH:
        # Check if the column is full
        adjacent_count = count_adjacent_discs(board, action)
        reward += 0.1 * adjacent_count  # Increase reward based on the count

        # Reward for a valid move
        reward += 1  # Give a small reward for a valid move

    else:
        print("BOARD IS FULL!!")

    return reward


# def is_blocking_opponent(board, action_column):
#     return False


def count_adjacent_discs(board, action_column):
    # check surrounings and count discs
    count = 0
    action_row = 0
    for row in range(config_values.HEIGHT):  # find row in which ction was taken
        if board[0, row, action_column] == 1:
            action_row = row

    # go around disc with catching errors
    for row_offset in [-1, 0, 1]:
        for column_offset in [-1, 0, 1]:
            try:
                if (
                    board[0, action_row + row_offset, action_column + column_offset]
                    == 1
                ):
                    count += 1
            except:
                pass
    return count


# Function to train the opponent
def train_opponent(opponent, opponent_model, epsilon, state, step):
    if opponent == "rand":
        action = np.random.randint(NUM_ACTIONS)
    elif opponent == "self":
        # flip the current state so the opponent sees his situation on the top layer!!
        # otherwise the rl opponent will predict action based on rl agents position
        state_copy = state.copy()
        state_copy = np.flip(state_copy, axis=1)
        action, _, _ = epsilon_greedy_action(state_copy, epsilon, opponent_model)
    elif opponent == "ascending_columns":
        # Opponent places discs in columns in ascending order
        action = step % NUM_ACTIONS
    # Add more opponent strategies as needed
    return action


# Function to initialize the models
def model_init(train_from_start):
    optimizer = tf.keras.optimizers.Adam(config_values.learning_rate)
    replay_buffer = ReplayBuffer(capacity=10000)

    if train_from_start:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build([(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)])
        model.compile(optimizer="adam", loss="mse")
    else:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build([(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)])
        model.compile(optimizer="adam", loss="mse")

        model.load_weights("./checkpoints/my_checkpoint")

    # Inside model_init() function
    opponent_model = DQN(num_actions=NUM_ACTIONS)
    opponent_model.build(
        [(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)]
    )

    return (
        model,
        opponent_model,
        replay_buffer,
        optimizer,
    )


# Function to get RL action when playing game ->used in other code
def get_rl_action(board, model):
    state = board_to_numpy(board, 2)
    q_values = model(
        [state, np.expand_dims(np.zeros(5), axis=0)]
    )  # all the flags are 0s
    return np.argmax(q_values)


# Convert Connect 4 board to NumPy array and make input indifferent
def board_to_numpy(board, current_player):
    array = np.zeros((config_values.HEIGHT, config_values.WIDTH, 2), dtype=np.float32)
    array[:, :, 0] = board == current_player  # Current player's discs
    array[:, :, 1] = board == 3 - current_player  # Opponent's discs
    return array.transpose((2, 0, 1))[np.newaxis, :]  # Add batch dimension


def numpy_to_board(array, current_player):
    # Transpose back to the original shape
    array = array.transpose((1, 0, 2))

    # Extract current player's discs and opponent's discs
    current_player_discs = np.where(array[:, :, 0] == 1)
    opponent_discs = np.where(array[:, :, 0] == 0)

    # Create an empty board
    board = np.zeros((config_values.HEIGHT, config_values.WIDTH))

    # Fill in the board with player and opponent discs
    board[current_player_discs[0], current_player_discs[1]] = current_player
    board[opponent_discs[0], opponent_discs[1]] = 3 - current_player

    return board


def choose_opponent(episode, opponent_switch_interval):
    if episode % opponent_switch_interval == 0:
        if np.random.rand() < 0.5:
            current_opponent = "rand"
        else:
            current_opponent = "ascending_columns"
    else:
        current_opponent = "self"

    return current_opponent