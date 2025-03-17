import sys
# Add the directory containing the PyGame-Learning-Environment to the Python path
sys.path.append("env/PyGame-Learning-Environment/")
# Add the current directory to the Python path for importing the flappyGym module
sys.path.append(".")

from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym  # Import the Flappy Bird environment
import time

# Initialize the Flappy Bird environment
env = FlappyBirdGym()
state = env.reset()  # Reset the environment to get the initial state

done = False  # Flag to indicate whether the game is over

while not done:
    env.render()  # Render the environment to display the game
    action = 0  # Choose action 0 (No Flap) to let the bird fall naturally
    state, reward, done, _ = env.step(action)  # Execute the action and receive feedback
    time.sleep(0.05)  # Introduce a small delay to slow down the simulation for observation
