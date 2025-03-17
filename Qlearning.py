from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym  # Import Flappy Bird environment
from agent.qlAgent import QLearningAgent  # Import Flappy Bird Q-learning Agent
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import random
import copy
import time
import time
import time  # Multiple imports of time are present; kept as in original code

def q_learning(agent, env, max_steps, train=True):
    """
    Runs Q-learning for Flappy Bird.
    
    - If `train=True`, the agent learns (updates its Q-table).
    - If `train=False`, the agent plays using a greedy policy (evaluation).
    
    Parameters:
        agent (QLearningAgent): The Q-learning agent.
        env (FlappyBirdGym): The Flappy Bird environment.
        max_steps (int): Maximum number of steps to run in a single run.
        train (bool): Flag to indicate training or evaluation mode.
    
    Returns:
        float: Total discounted return accumulated over the run.
    """
    state = env.reset()  # Reset environment and get initial state
    agent.init_state(state)  # Initialize the state in the agent's Q-table

    total_return = 0.0  # Initialize total discounted return
    done = False

    for i in range(max_steps):
        # Select action: use exploration if training, otherwise select greedy action
        action = agent.select_action(state) if train else agent.select_greedy(state)
        next_state, reward, done, _ = env.step(action)  # Execute action and observe outcome

        if train:
            # Update Q-table only during training mode
            agent.update_Qtable(state, action, reward, next_state)

        state = next_state  # Transition to next state
        total_return += pow(agent.gamma, i) * reward  # Accumulate discounted reward

        # Render environment only if display is enabled
        if env.display:
            env.render()

        # Slow down during evaluation to allow natural movement
        if not train and env.display:
            time.sleep(0.05)

        if done:
            state = env.reset()  # Reset environment if episode terminates

    return total_return


def run_ql(max_runs, max_steps, params, qtable_file, display=False, train=False):
    """
    Runs multiple Q-learning runs for Flappy Bird and optionally trains or evaluates the agent.
    
    Parameters:
        max_runs (int): Number of runs to execute.
        max_steps (int): Maximum steps per run.
        params (dict): Hyperparameters for the QLearningAgent.
        qtable_file (str): File path to save/load the Q-table.
        display (bool): Whether to display the environment.
        train (bool): Flag indicating training (True) or evaluation (False).
    
    Returns:
        list: List of total returns for each run.
    """
    results_list = []  # List to store the total return from each run
    best_return = float('-inf')  # Track the best total return
    best_qtable = None  # Track the best Q-table

    for run in range(max_runs):
        env = FlappyBirdGym(display=display)  # Create a new environment instance
        agent = QLearningAgent(env, params)  # Initialize the Q-learning agent with given parameters
        
        if not train and qtable_file is not None:
            # Load the Q-table from file if in evaluation mode
            agent.load_qtable(qtable_file)

        total_return = q_learning(agent, env, max_steps, train)  # Run Q-learning for the current run
        results_list.append(total_return)  # Record the total return

        env.close()  # Close the environment
        print(f"Run {run + 1}: Total Reward = {total_return}")

        if train and total_return > best_return:
            best_return = total_return  # Update best return if current run is better
            best_qtable = agent.Q  # Save the corresponding Q-table

    if train:
        # After training, save the best Q-table to file
        agent.Q = best_qtable
        agent.save_qtable(qtable_file)

    return results_list


## ================================
## Run Training or Evaluation
## ================================
num_runs = 10      
num_steps = 1000  # Maximum steps per run

params = {
    "gamma": 0.99,         # Discount factor for future rewards
    "alpha": 0.5,          # Learning rate for updating Q-values
    "epsilon": 0.5,        # Initial exploration probability
    "epsilon_min": 0.01,   # Minimum exploration probability after decay
    "epsilon_decay": 0.995 # Decay rate for exploration probability
}

qtable_file = "flappy_qtable.csv"

# Train the Q-learning agent and save the best Q-table
results_list = run_ql(num_runs, num_steps, params, qtable_file, display=False, train=True)

# Uncomment the following line to run evaluation instead of training
# results_list = run_ql(num_runs, num_steps, params, qtable_file, display=True, train=False)
