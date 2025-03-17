import torch
import numpy as np
import os
from collections import deque
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym
from agent.dqnAgent import DQNAgent

import warnings
warnings.simplefilter("ignore", UserWarning)

# Global tracking variables
best_score = -np.inf  # Initialize the best score to negative infinity
episode_rewards = deque(maxlen=100)  # Track the rewards of the last 100 episodes

def dqn_training(agent, env, max_steps):
    """
    Run a single training episode for the DQN agent.

    Parameters:
        agent (DQNAgent): The agent interacting with the environment.
        env (FlappyBirdGym): The Flappy Bird environment.
        max_steps (int): Maximum number of steps per episode.

    Returns:
        float: Total reward accumulated during the episode.
    """
    state = env.reset()  # Reset the environment and obtain the initial state
    total_reward = 0.0   # Initialize the total reward for the episode
    
    # Run the episode for a maximum number of steps
    for _ in range(max_steps):
        action = agent.select_action(state)  # Agent selects an action based on the current state
        next_state, reward, done, _ = env.step(action)  # Execute action in the environment
        
        # Add a survival bonus to the reward if the episode is not finished
        modified_reward = reward + 0.1 * (not done)
        agent.store_transition(state, action, modified_reward, next_state, done)  # Store the transition in memory
        
        # Train the agent if the replay memory has sufficient samples
        if len(agent.memory) > 5000:
            agent.train(batch_size=64)
            
        state = next_state  # Update the state
        total_reward += reward  # Accumulate the reward
        
        if done:
            break  # End the episode if the environment signals termination

    return total_reward

def run_dqn(num_episodes, max_steps, params, model_file="best_model.pth", display=False):
    """
    Train the DQN agent over multiple episodes and save the best performing model.

    Parameters:
        num_episodes (int): Total number of training episodes.
        max_steps (int): Maximum steps allowed per episode.
        params (dict): Hyperparameters for training the agent.
        model_file (str): Filename to save the best model.
        display (bool): Flag to enable or disable visual display during training.
    """
    global best_score  # Use the global best_score variable to track progress
    env = FlappyBirdGym(display=display)  # Create the Flappy Bird environment
    agent = DQNAgent(state_dim=3, action_dim=2, **params)  # Initialize the DQN agent with given hyperparameters

    # Calculate linear epsilon decay: fixed decrement per episode
    epsilon_step = (agent.epsilon - agent.min_epsilon) / num_episodes
    print(f"Using linear epsilon decay: epsilon_step = {epsilon_step:.6f}")

    try:
        for ep in range(num_episodes):
            score = dqn_training(agent, env, max_steps)  # Run a training episode and obtain the score
            episode_rewards.append(score)  # Append the score to the tracking deque
            
            # Update epsilon linearly after each episode
            agent.epsilon = max(agent.min_epsilon, agent.epsilon - epsilon_step)
            
            avg_score = np.mean(episode_rewards)  # Compute average score over the recent episodes
            if avg_score > best_score:
                best_score = avg_score  # Update the best score if current average is higher
                torch.save(agent.online_net.state_dict(), model_file)  # Save the model weights
                print(f"New best avg: {avg_score:.2f} (ε={agent.epsilon:.4f})")
                
            # Print training progress every 1000 episodes
            if (ep + 1) % 1000 == 0:
                print(f"Ep {ep+1:5d} | Train avg: {avg_score:.2f} | ε: {agent.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
    finally:
        env.close()  # Close the environment
        torch.save(agent.online_net.state_dict(), "final_model.pth")  # Save the final model weights
        print(f"\nTraining completed. Best avg score: {best_score:.2f}")

if __name__ == "__main__":
    # Training configuration (parameters remain the same)
    train_config = {
        "lr": 0.00025,               # Learning rate for the optimizer
        "gamma": 0.99,               # Discount factor for future rewards
        "epsilon": 1.0,              # Initial exploration rate
        "min_epsilon": 0.01,         # Minimum exploration rate after decay
        "decay": 0.9999995,          # Decay factor (not used with linear decay)
        "target_update_freq": 500,   # Frequency for updating the target network
        "replay_size": 500000        # Maximum size of the replay buffer
    }

    run_dqn(
        num_episodes=150000,  # Total number of episodes for training
        max_steps=2000,       # Maximum steps per episode
        params=train_config,  # Hyperparameters for training
        model_file="best_model.pth",  # File to save the best model
        display=False       # Disable display to speed up training
    )
