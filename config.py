class config:
    def __init__(self):
        #################################
        ##### Adjust variables here #####
        #################################

        self.num_episodes = 5000
        self.learning_rate = 0.001

        self.gamma = 0.9  # Discount factor for q-learning

        self.train_from_start = False

        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.99
        self.target_update_frequency = 200
        self.batch_size = 64

        self.visualization_frequency = 1000  # Put in a high value to train faster

        self.opponent_switch_interval = 3

        # Visualization constants
        self.WIDTH, self.HEIGHT = 7, 6
        self.CELL_SIZE = 100
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = (
            self.WIDTH * self.CELL_SIZE + 2 * self.CELL_SIZE, # Width of the window
            (self.HEIGHT + 2.5) * self.CELL_SIZE,             # Hight of the window
        )

        self.FPS = 30

        self.BACKGROUND_COLOR = (25, 25, 25)  # Dark background color
        self.GRID_COLOR = (100, 100, 100)  # Grid color
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.NUM_ACTIONS = self.WIDTH
        self.STATE_SHAPE = (2, self.HEIGHT, self.WIDTH)

        # Define constants for players
        self.HUMAN_PLAYER = 1
        self.RL_PLAYER = 2