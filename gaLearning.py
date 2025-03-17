import numpy as np
import torch
import os
import time
from tqdm import tqdm

from agent.gaAgent import (
    flatten_params, unflatten_params,
    crossover, mutate, selection, set_seeds
)
from agent.dqnAgent import DQNAgent
from Busy_bird.Busy_bird.flappyGym import FlappyBirdGym

# Set global seeds for reproducibility
set_seeds(42)

###################################################
# 1) Enhanced Individual Evaluation
###################################################
def evaluate_individual(individual, max_steps=500, render=False, n_evals=3):
    """
    Robust evaluation with multiple episodes and frame skipping.
    
    This function evaluates a GA individual (a flattened parameter vector) by
    unflattening its parameters into a DQN agent and running it over multiple episodes.
    
    Parameters:
        individual (np.ndarray): Flattened parameter vector representing a model.
        max_steps (int): Maximum number of steps per episode.
        render (bool): Whether to render the environment.
        n_evals (int): Number of episodes to average over.
    
    Returns:
        float: The average score over n_evals episodes.
    """
    # Initialize a DQN agent with deterministic behavior (epsilon = 0)
    agent = DQNAgent(state_dim=3, action_dim=2, epsilon=0.0)
    # Load the individual's parameters into the agent's online network
    unflatten_params(agent.online_net, individual)
    
    total_score = 0.0
    # Evaluate the individual over multiple episodes
    for _ in range(n_evals):
        env = FlappyBirdGym(display=render)
        state = env.reset()
        episode_score = 0.0
        frame_skip = 2  # Process every nth frame
        
        for step in range(max_steps):
            if step % frame_skip == 0:
                action = agent.select_action(state)
            
            state, reward, done, _ = env.step(action)
            episode_score += reward
            
            if done:
                break
                
        env.close()
        total_score += episode_score
    
    # Return the average score over the evaluation episodes
    return total_score / n_evals

###################################################
# 2) Optimized Genetic Algorithm Loop (CPU-only)
###################################################
def run_ga_evolution(pop_size=20,
                     n_generations=50,
                     max_steps=1000,
                     elite_frac=0.3,
                     mutation_rate=0.04,
                     mutation_scale=0.05,
                     base_model_file="best_dqn_model.pth",
                     output_ga_model="best_ga_model.pth"):
    """
    Optimized genetic algorithm loop to evolve model parameters.
    
    This function evolves a population of individuals (flattened parameter vectors)
    using selection, crossover, and mutation. It starts from a base DQN model.
    
    Parameters:
        pop_size (int): Number of individuals in the population.
        n_generations (int): Number of generations to run.
        max_steps (int): Maximum steps to evaluate an individual.
        elite_frac (float): Fraction of the population to keep as elites.
        mutation_rate (float): Mutation rate for genetic variation.
        mutation_scale (float): Scale of mutation noise.
        base_model_file (str): Path to the base DQN model file.
        output_ga_model (str): Path to save the best evolved GA model.
    """
    # Validate base model file existence
    if not os.path.exists(base_model_file):
        raise FileNotFoundError(f"Base DQN model '{base_model_file}' not found")
    
    print(f"Starting GA evolution with {pop_size} individuals over {n_generations} generations")
    
    # Load base model on CPU
    base_agent = DQNAgent(3, 2)
    base_agent.online_net.load_state_dict(
        torch.load(base_model_file, map_location='cpu')
    )
    base_agent.target_net.load_state_dict(base_agent.online_net.state_dict())
    
    # Initialize population by mutating the base model's parameters
    base_params = flatten_params(base_agent.online_net)
    population = [mutate(base_params.copy(), 0.1, 0.05) for _ in range(pop_size)]
    
    best = {
        "individual": base_params.copy(),
        "fitness": -np.inf,
        "generation": 0
    }
    
    # Evolution loop over generations
    for gen in range(1, n_generations+1):
        start_time = time.time()
        
        # Evaluate each individual in the population
        fitnesses = []
        for ind in tqdm(population, desc=f"Gen {gen}", leave=False):
            fitness = evaluate_individual(ind, max_steps, render=False, n_evals=3)
            fitnesses.append(fitness)
        
        # Track the best performer in the current generation
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best["fitness"]:
            best["fitness"] = fitnesses[gen_best_idx]
            best["individual"] = population[gen_best_idx].copy()
            best["generation"] = gen
        
        # Log progress for the current generation
        print(f"\nGeneration {gen}/{n_generations} | Best: {best['fitness']:.2f} | Avg: {np.mean(fitnesses):.2f} | Time: {time.time()-start_time:.1f}s")
        
        # Selection: choose elite individuals based on fitness
        elites = selection(population, fitnesses, elite_frac)
        new_pop = elites.copy()
        
        # Reproduction: generate offspring until population is replenished
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(len(elites), 2, replace=False)
            child = crossover(elites[p1], elites[p2], 0.6)
            child = mutate(child, mutation_rate, mutation_scale)
            new_pop.append(child)
            
        population = new_pop

    # Save the best evolved model
    best_agent = DQNAgent(3, 2)
    unflatten_params(best_agent.online_net, best["individual"])
    torch.save(best_agent.online_net.state_dict(), output_ga_model)
    print(f"\nEvolution complete! Best fitness {best['fitness']:.2f} (gen {best['generation']}) saved to {output_ga_model}")

###################################################
# 3) Simplified Example Usage
###################################################
if __name__ == "__main__":
    run_ga_evolution(
        pop_size=50,
        n_generations=100,
        max_steps=2000,
        elite_frac=0.2,
        mutation_rate=0.1,
        mutation_scale=0.15,
        base_model_file="best_model.pth",
        output_ga_model="best_ga_model.pth"
    )

    # Final evaluation 
    agent = DQNAgent(3, 2)
    agent.online_net.load_state_dict(torch.load("best_ga_model.pth"))
    env = FlappyBirdGym(display=True)
    scores = [evaluate_individual(flatten_params(agent.online_net), max_steps=1000, render=True) for _ in range(5)]
    env.close()
    
    print(f"Final Performance: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
