import torch
import time
from agent.dqnAgent import DQNAgent
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym

def visualize_agent(model_file, agent_name="Agent", display=True):
    """
    Loads the given model and plays Flappy Bird while rendering.
    
    Parameters:
        model_file (str): Path to the model file.
        agent_name (str): Name of the agent (for display purposes).
        display (bool): Whether to render the game screen.
    """
    # Print message indicating the model is being loaded (emoji removed)
    print(f"\nLoading {agent_name} model from {model_file}...")

    # Initialize the DQN agent with specified hyperparameters
    # epsilon is set to 0.0 for a fully greedy policy during visualization
    agent = DQNAgent(state_dim=3, action_dim=2, lr=0.001, gamma=0.99, epsilon=0.0)
    # Load model weights into the agent's online network
    agent.online_net.load_state_dict(torch.load(model_file))
    # Ensure the target network has the same weights as the online network
    agent.target_net.load_state_dict(agent.online_net.state_dict())

    # Create the Flappy Bird environment with rendering as specified by the display flag
    env = FlappyBirdGym(display=display)
    # Reset the environment to get the initial state
    state = env.reset()
    total_reward = 0  # Initialize total reward counter
    done = False  # Flag to track when the episode is finished

    # Game loop: run until the game signals the episode is done
    while not done:
        # Agent selects an action based on the current state
        action = agent.select_action(state)
        # Execute the action in the environment and get the next state and reward
        state, reward, done, _ = env.step(action)
        total_reward += reward  # Accumulate reward

        # Render the current game state for visualization
        env.render()  # Show the game
        # Introduce a small delay to slow down the game for better visualization
        time.sleep(0.03)

    # Close the environment after the episode is over
    env.close()
    # Print the total reward achieved by the agent (emoji removed)
    print(f"{agent_name} finished with total reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Visualize the performance of the DQN model
    visualize_agent("best_dqn_model.pth", agent_name="DQN Agent")

    # Visualize the performance of the GA-evolved model
    visualize_agent("best_ga_model.pth", agent_name="GA Agent")
