import torch
import os
import numpy as np
import time
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym
from agent.dqnAgent import DQNAgent

def dqn_evaluate(agent, env, max_steps, n_episodes=10, sleep_time=0.05):
    """
    Evaluate the DQN agent on the given environment over multiple episodes.
    
    Parameters:
        agent (DQNAgent): The trained DQN agent.
        env (FlappyBirdGym): The Flappy Bird environment.
        max_steps (int): Maximum number of steps to run in an episode.
        n_episodes (int): Number of episodes to evaluate.
        sleep_time (float): Delay between frames when rendering.
    
    Returns:
        float: The average reward over the evaluated episodes.
    """
    total_reward = 0.0  # Accumulate total reward over all episodes
    for ep in range(n_episodes):
        state = env.reset()  # Reset environment at the beginning of each episode
        episode_reward = 0.0  # Initialize reward for the episode
        
        while True:
            # Select an action using the agent's policy
            action = agent.select_action(state)
            # Execute the action and observe the next state, reward, and done flag
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward  # Accumulate reward for the episode
            state = next_state  # Update state
            
            # Render the environment if display is enabled
            if env.display:
                env.render()
                time.sleep(sleep_time)  # Slow down the gameplay for visualization
            
            # Exit the loop if the episode is finished
            if done:
                break
        
        total_reward += episode_reward  # Add the episode's reward to the total
        print(f"Episode {ep+1}: {episode_reward:.2f}")  # Log episode reward
    
    # Return the average reward over all episodes
    return total_reward / n_episodes

def evaluate_model(model_file, max_steps=2000, n_episodes=10, display=True, sleep_time=0.05):
    """
    Evaluate a trained model on the Flappy Bird environment.
    
    Parameters:
        model_file (str): Path to the saved model file.
        max_steps (int): Maximum steps per episode.
        n_episodes (int): Number of episodes to evaluate.
        display (bool): Whether to display the game during evaluation.
        sleep_time (float): Time to sleep between rendering frames.
    
    This function loads the model, creates the agent with minimal exploration,
    evaluates its performance, and prints the average score.
    """
    # Create the Flappy Bird environment with specified display settings
    env = FlappyBirdGym(display=display)
    
    # Define evaluation parameters; no exploration during evaluation
    eval_params = {
        "epsilon": 0.0,      # Force greedy policy (no exploration)
        "min_epsilon": 0.0   # Not used during evaluation
    }
    
    # Create a DQN agent with the minimal evaluation parameters
    agent = DQNAgent(state_dim=3, action_dim=2, **eval_params)
    
    # Check if the model file exists; if not, print an error and exit
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found")
        env.close()
        return
    
    # Load the trained model weights into the agent's online network and set to evaluation mode
    agent.online_net.load_state_dict(torch.load(model_file))
    agent.online_net.eval()
    
    # Evaluate the model using the dqn_evaluate function
    avg_score = dqn_evaluate(agent, env, max_steps, n_episodes, sleep_time)
    env.close()  # Close the environment after evaluation
    print(f"\nAverage evaluation score (over {n_episodes} episodes): {avg_score:.2f}")

if __name__ == "__main__":
    model_file = "best_model.pth"  # Ensure this file exists from training
    evaluate_model(model_file, max_steps=2000, n_episodes=10, display=True, sleep_time=0.05)
