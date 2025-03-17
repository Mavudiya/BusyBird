import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate library load for KMP on some systems
import torch
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Suppress PNG warnings and other user warnings
warnings.filterwarnings("ignore", category=UserWarning)

from agent.dqnAgent import DQNAgent
from agent.gaAgent import set_seeds
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym

# Set global seeds for reproducibility
set_seeds(42)

def evaluate_ga_model(model_file="best_ga_model.pth", 
                      n_episodes=50, 
                      max_steps=2000,
                      render_freq=0.1):
    """
    Evaluate the GA-evolved model using CPU only.
    
    Parameters:
        model_file (str): Path to the GA model file.
        n_episodes (int): Number of episodes to evaluate.
        max_steps (int): Maximum steps per episode.
        render_freq (float): Fraction of episodes to render (e.g., 0.1 = 10%).
    
    Returns:
        None. Prints performance statistics and saves a histogram of scores.
    """
    # Initialize the agent with the given state and action dimensions
    agent = DQNAgent(state_dim=3, action_dim=2)
    # Load the model weights from the specified file to CPU
    agent.online_net.load_state_dict(torch.load(model_file, map_location='cpu'))
    
    # Set the network to evaluation mode (disable dropout, etc.)
    agent.online_net.eval()
    
    scores = []  # List to store total rewards for each episode
    render_episodes = int(n_episodes * render_freq)  # Determine number of episodes to render
    
    print(f"Evaluating {n_episodes} episodes (rendering {render_episodes})...")
    
    for ep in tqdm(range(n_episodes)):
        # Determine whether to render this episode based on render frequency
        render = ep < render_episodes
        # Initialize the Flappy Bird environment with or without display
        env = FlappyBirdGym(display=render)
        state = env.reset()  # Reset the environment and get the initial state
        total_reward = 0.0  # Initialize total reward for this episode
        done = False
        
        # Set frame skipping to speed up evaluation: fewer frames processed if not rendering
        frame_skip = 2 if render else 4
        
        for step in range(max_steps):
            # Only process an action every 'frame_skip' frames
            if step % frame_skip == 0:
                with torch.no_grad():
                    # Convert state to a tensor and compute Q-values
                    state_tensor = torch.FloatTensor(state)
                    q_values = agent.online_net(state_tensor)
                    # Select the action with the highest Q-value
                    action = torch.argmax(q_values).item()
            
            # Take the selected action and observe the next state and reward
            state, reward, done, _ = env.step(action)
            total_reward += reward  # Accumulate reward
            
            # If rendering, add a small delay for smooth visualization
            if render:
                time.sleep(0.02)
                
            # Exit the loop if the episode is finished
            if done:
                break
        
        env.close()  # Close the environment
        scores.append(total_reward)  # Record the total reward for the episode
    
    # Calculate performance statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    
    # Print performance statistics
    print(f"\nGA Model Performance:")
    print(f"Average Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Best Score: {max_score:.2f}")
    
    # Visualize the score distribution using a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7)
    plt.title(f"Score Distribution ({n_episodes} Episodes)")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("ga_performance.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    evaluate_ga_model(
        model_file="best_ga_model.pth",
        n_episodes=100,
        max_steps=2000,
        render_freq=0.1
    )
