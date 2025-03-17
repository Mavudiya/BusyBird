# compareModels.py
import pygame 
import os
import time 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate KMP library loading
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress PNG warnings and other unnecessary alerts
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from agent.dqnAgent import DQNAgent
from agent.gaAgent import set_seeds
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym

# Set global seeds for reproducibility
set_seeds(42)

# Updated evaluate_model function with visualization control
def evaluate_model(model_path: str, 
                   num_episodes: int = 10,
                   max_steps: int = 2000,
                   render_freq: float = 0.1) -> dict:
    """
    Optimized model evaluation with timing improvements.
    
    Loads the model from 'model_path' and evaluates it over a given number of episodes.
    Returns a dictionary with evaluation metrics.
    
    Parameters:
        model_path (str): Path to the model file.
        num_episodes (int): Number of episodes to evaluate.
        max_steps (int): Maximum steps per episode.
        render_freq (float): Fraction of episodes to render for visualization.
    
    Returns:
        dict: Evaluation metrics including scores, mean, standard deviation, max, and min.
    """
    # Load model directly to CPU and set it to evaluation mode
    agent = DQNAgent(state_dim=3, action_dim=2)
    agent.online_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.online_net.eval()
    
    scores = []  # List to store total rewards for each episode
    render_episodes = max(1, int(num_episodes * render_freq))  # Determine number of episodes to render
    
    # Evaluate the model over a number of episodes
    for ep in tqdm(range(num_episodes), desc=f"Evaluating {os.path.basename(model_path)}"):
        render = ep < render_episodes  # Enable rendering only for the first few episodes
        env = FlappyBirdGym(display=render)  # Create environment instance with/without display
        state = env.reset()  # Reset environment to get initial state
        total_reward = 0.0  # Initialize total reward for the episode
        done = False  # Flag to track if the episode has ended
        clock = pygame.time.Clock()  # Clock for controlling FPS
        
        # Iterate over steps within an episode
        for step in range(max_steps):
            if step % 2 == 0:  # Process every 2nd frame for efficiency
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)  # Convert state to tensor
                    q_values = agent.online_net(state_tensor)  # Get Q-values from the network
                    action = torch.argmax(q_values).item()  # Select action with highest Q-value
            
            # Execute the chosen action and get new state, reward, and done flag
            state, reward, done, _ = env.step(action)
            total_reward += reward  # Accumulate reward
            
            # Handle rendering if enabled
            if render:
                clock.tick(30)  # Limit to 30 FPS
                pygame.event.pump()  # Process event queue to keep window responsive
                time.sleep(0.02)  # Delay for smooth animation
            
            # Break the loop if the episode has ended
            if done:
                if render:
                    # Keep window open for 0.5 seconds after episode ends
                    start_time = time.time()
                    while time.time() - start_time < 0.5:
                        pygame.event.pump()
                        clock.tick(30)
                break
                
        env.close()  # Close the environment
        scores.append(total_reward)  # Record the episode's total reward
    
    # Return evaluation metrics as a dictionary
    return {
        "scores": np.array(scores),
        "mean": np.mean(scores),
        "std": np.std(scores),
        "max": np.max(scores),
        "min": np.min(scores)
    }

def compare_models(dqn_path: str = "best_dqn_model.pth",
                   ga_path: str = "best_ga_model.pth",
                   num_episodes: int = 100,
                   max_steps: int = 2000,
                   render_freq: float = 1.0):
    """
    Time-optimized model comparison with reduced overhead.
    
    Evaluates the performance of the DQN and GA models over a number of episodes,
    prints out key metrics, and creates visualizations comparing their performances.
    
    Parameters:
        dqn_path (str): File path for the DQN model.
        ga_path (str): File path for the GA model.
        num_episodes (int): Number of episodes for evaluation.
        max_steps (int): Maximum steps per episode.
        render_freq (float): Fraction of episodes to render.
    """
    # Validate that both model files exist
    if not all(map(os.path.exists, [dqn_path, ga_path])):
        raise FileNotFoundError("Missing model file(s)")  # Removed emoji from error message

    # Evaluate the DQN model and print status
    print("\nEvaluating DQN Model...")  # Removed emoji from print statement
    dqn_results = evaluate_model(dqn_path, num_episodes, max_steps, render_freq)
    
    # Evaluate the GA model and print status
    print("\nEvaluating GA Model...")  # Removed emoji from print statement
    ga_results = evaluate_model(ga_path, num_episodes, max_steps, render_freq)

    # Print final comparison results
    print("\nFinal Comparison Results:")  # Removed emoji from print statement
    print(f"{'Metric':<10} | {'DQN':<10} | {'GA':<10}")
    print("-" * 32)
    print(f"{'Average':<10} | {dqn_results['mean']:>10.2f} | {ga_results['mean']:>10.2f}")
    print(f"{'Best':<10} | {dqn_results['max']:>10.2f} | {ga_results['max']:>10.2f}")
    print(f"{'Stability':<10} | {dqn_results['std']:>10.2f} | {ga_results['std']:>10.2f}")

    # Create visualization for performance comparison
    plt.figure(figsize=(12, 5))
    
    # Plot performance scores across episodes for both models
    plt.subplot(1, 2, 1)
    plt.plot(dqn_results['scores'], label='DQN', alpha=0.7)
    plt.plot(ga_results['scores'], label='GA', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Performance Across Episodes")
    plt.legend()

    # Plot boxplots for statistical comparison of score distributions
    plt.subplot(1, 2, 2)
    plt.boxplot([dqn_results['scores'], ga_results['scores']], 
                labels=['DQN', 'GA'])
    plt.ylabel("Score")
    plt.title("Score Distribution Comparison")

    plt.tight_layout()
    # Save the visualization to a file
    plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
    print("\nComparison saved to model_comparison.png")  # Removed emoji from print statement
    plt.close()

if __name__ == "__main__":
    compare_models(
        dqn_path="best_model.pth",
        ga_path="best_ga_model.pth",
        num_episodes=4,
        max_steps=2000,
        render_freq=1.0
    )
