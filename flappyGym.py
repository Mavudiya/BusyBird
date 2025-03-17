import os
import gym
import numpy as np
import pygame
import sys
import random

# Append the directory containing the PyGame-Learning-Environment to the system path
sys.path.append("env/PyGame-Learning-Environment/")  # Ensure PLE can be found
from ple import PLE
from ple.games.flappybird import FlappyBird

class FlappyBirdGym(gym.Env):
    """
    Custom Gym environment for Flappy Bird using the PyGame-Learning-Environment (PLE).
    This environment wraps the Flappy Bird game to provide a gym-compatible interface.
    """
    def __init__(self, display=False, seed=None):
        """
        Initialize the Flappy Bird Gym environment.
        
        Parameters:
            display (bool): Whether to display the game screen.
            seed (int, optional): Seed for reproducibility.
        """
        super(FlappyBirdGym, self).__init__()

        print(f"Initializing FlappyBirdGym environment. display={display}")

        self.display = display

        # If a seed is provided, set the random seeds for reproducibility.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # If display is disabled, force headless mode by setting the SDL video driver to 'dummy'
        if not self.display:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
            pygame.display.set_mode((1, 1))

        # Initialize the Flappy Bird game from PLE
        self.game = FlappyBird()
        # Create the PLE environment with a fixed frames-per-second (fps) and display settings
        self.env = PLE(self.game, fps=30, display_screen=self.display)

        # Define the action space: 0 = No Flap, 1 = Flap
        self.action_space = gym.spaces.Discrete(2)

        # Define the observation space with three features:
        # [bird_y (vertical position), bird_velocity, distance to the next pipe]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -15, 0], dtype=np.float32),
            high=np.array([512, 15, 288], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize previous score to keep track of when the bird passes a pipe
        self.prev_score = 0

        # Initialize the PLE environment
        self.env.init()

    def reset(self):
        """
        Reset the Flappy Bird game and return the initial state.
        
        This function resets the game, resets the stored score, and introduces a bit of randomness
        by performing a few random actions before returning the initial state.
        
        Returns:
            np.ndarray: The initial state of the game.
        """
        self.env.reset_game()
        self.prev_score = 0  # Reset the stored score at episode start

        # Introduce randomness by taking 0-5 random actions
        for _ in range(random.randint(0, 5)):
            # Randomly select an action from the available actions (0 or 1)
            self.env.act(self.env.getActionSet()[random.choice([0, 1])])

        return self.get_state()

    def step(self, action):
        """
        Execute the given action, apply reward shaping, and return the result.
        
        Reward scheme:
          - Base reward from the game (raw_reward from PLE).
          - +1.0 for each new pipe passed (score increase).
          - +0.1 bonus for every step survived.
          - -1.0 penalty if the bird dies.
        
        Parameters:
            action (int): The action to be taken (0 or 1).
        
        Returns:
            tuple: (state, reward, done, info)
                state (np.ndarray): The new state after the action.
                reward (float): The shaped reward.
                done (bool): True if the episode is over.
                info (dict): Additional information (empty in this case).
        """
        # Execute the action in the PLE environment and get the raw reward.
        raw_reward = self.env.act(self.env.getActionSet()[action])

        # Get the updated game state after the action.
        state = self.get_state()
        # Determine if the game is over.
        done = self.env.game_over()

        # Start with the base reward from PLE.
        reward = raw_reward

        # Check if the score has increased (i.e., the bird passed a pipe)
        current_score = self.game.getScore()  # Alternatively, can use getGameState()["score"]
        if current_score > self.prev_score:
            # Add the score difference to the reward (e.g., +1 for each pipe passed)
            reward += (current_score - self.prev_score)
        # Update previous score to current score for future comparisons.
        self.prev_score = current_score

        if not done:
            # Award a small bonus for surviving the step.
            reward += 0.1
        else:
            # Apply a penalty if the bird dies.
            reward = -1.0

        return state, reward, done, {}

    def get_state(self):
        """
        Retrieve the current state of the game as a NumPy array.
        
        The state consists of:
            - The vertical position of the bird.
            - The velocity of the bird.
            - The horizontal distance to the next pipe.
        
        Returns:
            np.ndarray: The current state.
        """
        st = self.env.getGameState()
        return np.array([
            st["player_y"],
            st["player_vel"],
            st["next_pipe_dist_to_player"]
        ], dtype=np.float32)

    def render(self, mode="human"):
        """
        Render the game screen.
        
        If display is enabled, set the environment's display flag accordingly.
        
        Parameters:
            mode (str): The mode of rendering (only "human" is supported).
        """
        if self.display:
            self.env.display_screen = True
        else:
            self.env.display_screen = False

    def close(self):
        """
        Close the Pygame environment and clean up resources.
        """
        pygame.quit()
