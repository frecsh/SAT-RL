"""
Curriculum Learning for SAT problems.
Tackles the phase transition problem by gradually increasing problem difficulty.
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from deep_q_sat_agent import DeepQLearningAgent
from improved_sat_gan import ImprovedSATGAN
from anytime_sat_solver import AnytimeEnsembleSolver
import tensorflow as tf


class CurriculumSATLearner:
    def __init__(self, n_vars, initial_ratio=3.0, target_ratio=4.2, step_size=0.2):
        """
        Initialize a Curriculum Learning approach for SAT problems.
        
        Args:
            n_vars: Number of variables in the SAT problem
            initial_ratio: Starting clause-to-variable ratio (easier)
            target_ratio: Target clause-to-variable ratio (harder, typically 4.2)
            step_size: Initial step size for curriculum progression
        """
        self.n_vars = n_vars
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.initial_step_size = step_size
        
        # Generate target problem at the desired difficulty
        self.target_clauses = self.generate_sat_problem(n_vars, target_ratio)
        
        # Track progress through curriculum
        self.progress_history = []
        
        # Solution pool for diversity
        self.solution_pool = []
        self.max_pool_size = 10
        
        # Best solution found so far
        self.best_solution = None
        self.best_satisfied = 0
    
    def generate_sat_problem(self, n_vars, ratio, unique_clauses=True):
        """Generate a random 3-SAT problem with given clause-to-variable ratio"""
        n_clauses = int(ratio * n_vars)
        clauses = []
        clause_set = set()
        
        while len(clauses) < n_clauses:
            # Generate a random clause with 3 literals (3-SAT)
            vars_in_clause = random.sample(range(1, n_vars + 1), 3)
            
            # Randomly negate some variables
            clause = [var if random.random() > 0.5 else -var for var in vars_in_clause]
            
            # Convert to tuple for set operations
            clause_tuple = tuple(sorted(clause, key=abs))
            
            # Add clause if unique or if not enforcing uniqueness
            if not unique_clauses or clause_tuple not in clause_set:
                clauses.append(clause)
                if unique_clauses:
                    clause_set.add(clause_tuple)
        
        return clauses
    
    def determine_best_agent(self, ratio, attempt):
        """Intelligently select agent based on problem characteristics and attempt number"""
        # For easier problems, DQN tends to work well
        if ratio < 3.5:
            if attempt <= 2:
                return 'dqn'
            else:
                return 'gan'
        # For medium difficulty, alternate more quickly
        elif ratio < 4.0:
            return 'dqn' if attempt % 2 == 1 else 'gan'
        # Near phase transition, try GAN first
        else:
            if attempt == 1:
                return 'gan'
            elif attempt == 2: 
                return 'dqn'
            else:
                # For later attempts, use ensemble approach
                return 'ensemble'
    
    def get_agent_config(self, agent_type, ratio, attempt, clauses):
        """Configure agent parameters based on difficulty level and attempt number"""
        config = {}
        
        if agent_type == 'dqn':
            # Configure exploration parameters based on curriculum level
            if ratio < 3.5:
                config['epsilon'] = 1.0
                config['epsilon_decay'] = 0.995
                config['learning_rate'] = 0.001
            elif ratio < 4.0:
                config['epsilon'] = 1.0
                config['epsilon_decay'] = 0.99  # Slower decay for harder problems
                config['learning_rate'] = 0.0005
            else:
                config['epsilon'] = 1.0  
                config['epsilon_decay'] = 0.98  # Even slower decay near phase transition
                config['learning_rate'] = 0.0002
            
            # Increase exploration for retry attempts
            if attempt > 1:
                config['epsilon'] = min(1.0, 0.5 + 0.1 * attempt)
                
        elif agent_type == 'gan':
            # Configure GAN parameters based on difficulty
            if ratio < 3.5:
                config['epochs'] = 100
                config['latent_dim'] = 32
            elif ratio < 4.0:
                config['epochs'] = 150
                config['latent_dim'] = 48
            else:
                config['epochs'] = 200
                config['latent_dim'] = 64
                
        elif agent_type == 'ensemble':
            # Configure ensemble parameters
            config['time_limit'] = 30 + int(10 * (ratio - self.initial_ratio))
            config['strategies'] = [
                {'name': 'random_walk', 'weight': 1.0},
                {'name': 'greedy', 'weight': 2.0},
                {'name': 'annealing', 'weight': 3.0}
            ]
            
        return config
    
    def solve_with_agent(self, agent_type, clauses, current_ratio, attempt, max_episodes=20):
        """Solve SAT problem using the specified agent type with appropriate configuration"""
        print(f"Attempt {attempt}/5 with {agent_type}")
        
        # Get configuration specific to this difficulty level and attempt
        config = self.get_agent_config(agent_type, current_ratio, attempt, clauses)
        
        # Choose starting point from solution pool if available
        starting_point = self.get_diverse_starting_point() if hasattr(self, 'solution_pool') and self.solution_pool else None
        
        # Apply restart mechanism for DQN agent
        restart_callback = None
        if agent_type == 'dqn':
            def restart_callback_fn(episode, best_satisfied):
                # If 5 episodes without improvement, restart with new initialization
                if episode > 0 and episode % 5 == 0 and best_satisfied < len(clauses):
                    print(f"Restarting with new initialization at episode {episode}")
                    return True
                return False
            
            restart_callback = restart_callback_fn
        
        # Initialize appropriate agent with configuration
        if agent_type == 'dqn':
            agent = DeepQLearningAgent(
                self.n_vars, 
                clauses,
                epsilon=config.get('epsilon', 1.0),
                epsilon_decay=config.get('epsilon_decay', 0.995),
                epsilon_min=config.get('epsilon_min', 0.1),
                learning_rate=config.get('learning_rate', 0.001),
                restart_callback=restart_callback
            )
            solution, stats = agent.solve(max_episodes=max_episodes)
            solved = stats['best_satisfied'] == len(clauses)
            satisfied = stats['best_satisfied']
            
            # Store agent for potential knowledge transfer
            self.last_agent = agent
            self.last_agent_type = 'dqn'
            
        elif agent_type == 'gan':
            agent = ImprovedSATGAN(
                self.n_vars, 
                clauses,
                latent_dim=config.get('latent_dim', 32),
                epochs=config.get('epochs', 100),
                initial_solutions=[starting_point] if starting_point is not None else None
            )
            agent.train_with_experience_replay()
            solution = agent.solve(max_generations=30)
            
            # Evaluate solution
            satisfied = 0
            for clause in clauses:
                for literal in clause:
                    var = abs(literal) - 1  # Convert to 0-indexed
                    val = solution[var]
                    if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                        satisfied += 1
                        break
            
            solved = satisfied == len(clauses)
            
            # Store agent for potential knowledge transfer
            self.last_agent = agent
            self.last_agent_type = 'gan'
            
        elif agent_type == 'ensemble':
            agent = AnytimeEnsembleSolver(
                self.n_vars,
                clauses,
                time_limit=config.get('time_limit', 30),
                strategies=config.get('strategies', None)
            )
            solution, stats = agent.solve()
            satisfied = stats['satisfied']
            solved = satisfied == len(clauses)
            
            # Ensemble doesn't support direct knowledge transfer
            self.last_agent = None
            self.last_agent_type = None
        
        # Update solution pool with this solution
        self.add_to_solution_pool(solution, satisfied)
        
        # Update best solution if better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_solution = solution
            
        # Track curriculum progress
        self.track_curriculum_progress(current_ratio, attempt, satisfied, len(clauses))
        
        return solved, solution, satisfied
    
    def transfer_knowledge(self, new_agent, current_ratio, next_ratio):
        """Enhanced knowledge transfer between difficulty levels"""
        if self.last_agent is None or self.last_agent_type is None:
            return False
            
        # Skip if agent types don't match
        if (self.last_agent_type == 'dqn' and not isinstance(new_agent, DeepQLearningAgent)) or \
           (self.last_agent_type == 'gan' and not isinstance(new_agent, ImprovedSATGAN)):
            return False
        
        try:
            # Calculate transfer factor based on difficulty jump
            ratio_difference = next_ratio - current_ratio
            # Less aggressive transfer for bigger jumps in difficulty
            transfer_factor = max(0.5, 1.0 - ratio_difference)
            
            if self.last_agent_type == 'dqn' and hasattr(self.last_agent, 'model') and hasattr(new_agent, 'model'):
                # Get weights from source model
                source_weights = self.last_agent.model.get_weights()
                
                # Get target model's initial weights
                target_weights = new_agent.model.get_weights()
                
                # Create blended weights with regularization to prevent forgetting
                blended_weights = []
                for i, (sw, tw) in enumerate(zip(source_weights, target_weights)):
                    # More transfer for earlier layers (feature extraction)
                    # Less transfer for final layers (decision making)
                    if i < len(source_weights) - 2:
                        blend = sw * transfer_factor + tw * (1 - transfer_factor)
                    else:
                        blend = sw * (transfer_factor * 0.5) + tw * (1 - transfer_factor * 0.5)
                    blended_weights.append(blend)
                    
                # Set blended weights to target model
                new_agent.model.set_weights(blended_weights)
                print(f"Transferred neural network knowledge with factor {transfer_factor:.2f}")
                return True
                
            elif self.last_agent_type == 'gan' and hasattr(self.last_agent, 'experience_buffer') and \
                 hasattr(new_agent, 'experience_buffer'):
                # Transfer promising solutions from buffer
                if len(self.last_agent.experience_buffer) > 0:
                    transfer_count = min(20, len(self.last_agent.experience_buffer))
                    experiences = random.sample(list(self.last_agent.experience_buffer), transfer_count)
                    
                    for exp in experiences:
                        new_agent.experience_buffer.append(exp)
                        
                    print(f"Transferred {transfer_count} promising solutions to new GAN buffer")
                    return True
                    
            return False
        except Exception as e:
            print(f"Knowledge transfer failed: {str(e)}")
            return False
    
    def add_to_solution_pool(self, solution, satisfied):
        """Add solution to diversity pool if sufficiently different or better"""
        if solution is None:
            return False
            
        # Initialize solution pool if needed
        if not hasattr(self, 'solution_pool'):
            self.solution_pool = []
            
        if len(self.solution_pool) < self.max_pool_size:
            self.solution_pool.append((solution.copy(), satisfied))
            return True
            
        # Find most similar solution in pool
        min_distance = float('inf')
        min_idx = 0
        
        for i, (pool_solution, _) in enumerate(self.solution_pool):
            # Hamming distance
            distance = sum(s1 != s2 for s1, s2 in zip(solution, pool_solution))
            
            if distance < min_distance:
                min_distance = distance
                min_idx = i
                
        # If different enough or better than most similar solution
        pool_satisfied = self.solution_pool[min_idx][1]
        if min_distance > self.n_vars * 0.2 or satisfied > pool_satisfied:
            if satisfied >= pool_satisfied:  # Replace worse solution
                self.solution_pool[min_idx] = (solution.copy(), satisfied)
                return True
                
        return False
        
    def get_diverse_starting_point(self):
        """Get diverse starting point from solution pool"""
        if not hasattr(self, 'solution_pool') or not self.solution_pool:
            return None
            
        # Randomly select from pool with preference for better solutions
        weights = [sat for _, sat in self.solution_pool]
        chosen_idx = random.choices(range(len(self.solution_pool)), 
                                weights=weights, k=1)[0]
        return self.solution_pool[chosen_idx][0]
    
    def track_curriculum_progress(self, ratio, attempt, satisfied, total):
        """Track detailed progress metrics"""
        if not hasattr(self, 'progress_history'):
            self.progress_history = []
            
        self.progress_history.append({
            'ratio': ratio,
            'attempt': attempt,
            'satisfied': satisfied,
            'total': total,
            'percentage': satisfied/total,
            'timestamp': time.time()
        })
    
    def visualize_learning_curve(self):
        """Visualize learning curve for curriculum progress"""
        if not hasattr(self, 'progress_history') or len(self.progress_history) < 2:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Extract data from history
        ratios = [entry['ratio'] for entry in self.progress_history]
        percentages = [entry['percentage'] for entry in self.progress_history]
        attempts = [entry['attempt'] for entry in self.progress_history]
        
        # Plot satisfaction percentage over time
        plt.plot(range(len(percentages)), percentages, 'b-', label='Satisfaction %')
        
        # Color-code by ratio
        unique_ratios = sorted(list(set(ratios)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ratios)))
        
        for i, ratio in enumerate(unique_ratios):
            indices = [j for j, r in enumerate(ratios) if r == ratio]
            plt.scatter([idx for idx in indices], [percentages[idx] for idx in indices], 
                       color=colors[i], label=f'Ratio {ratio:.2f}')
        
        # Mark attempt changes
        for i in range(1, len(attempts)):
            if attempts[i] != attempts[i-1]:
                plt.axvline(x=i, linestyle='--', color='gray', alpha=0.5)
        
        plt.xlabel('Curriculum Step')
        plt.ylabel('Clause Satisfaction')
        plt.title('Curriculum Learning Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('curriculum_progress.png')
        plt.close()
        print("Learning curve visualization saved to curriculum_progress.png")
    
    def solve_with_curriculum(self, max_attempts=5):
        """
        Solve the SAT problem using curriculum learning.
        Gradually increases problem difficulty from initial_ratio to target_ratio.
        """
        print(f"Starting curriculum learning from ratio {self.initial_ratio} to {self.target_ratio}")
        
        current_ratio = self.initial_ratio
        current_step = self.initial_step_size
        level = 1
        
        # Continue until we solve the target problem
        while current_ratio <= self.target_ratio:
            print(f"\n--- Curriculum Level {level}: Ratio = {current_ratio:.2f} ---")
            
            # Generate problem at current difficulty
            current_clauses = self.generate_sat_problem(self.n_vars, current_ratio)
            
            # Try to solve with different agent types
            solved = False
            attempt = 1
            
            while not solved and attempt <= max_attempts:
                # Select agent type based on difficulty and attempt number
                agent_type = self.determine_best_agent(current_ratio, attempt)
                
                # Try to solve with the selected agent
                solved, solution, satisfied = self.solve_with_agent(
                    agent_type, current_clauses, current_ratio, attempt
                )
                
                # If solved, update best solution
                if solved:
                    print(f"Solved level {level} (ratio {current_ratio:.2f}) with {agent_type}!")
                    self.best_solution = solution
                    self.best_satisfied = len(current_clauses)
                    break
                
                attempt += 1
            
            # Dynamic difficulty adjustment based on performance
            next_ratio = current_ratio
            if solved:
                # Progress to next level
                next_ratio = current_ratio + current_step
                print(f"Advancing to ratio {next_ratio:.2f}")
            else:
                # Assess how close we are to solution
                percentage_solved = satisfied / len(current_clauses)
                
                if percentage_solved > 0.95:  # Very close to solution
                    # Small step forward despite not fully solving
                    next_ratio = current_ratio + current_step * 0.5
                    print(f"Close enough ({percentage_solved:.2f}), advancing to ratio {next_ratio:.2f}")
                elif percentage_solved > 0.85:  # Making decent progress
                    # Reduce step size but still advance slightly
                    current_step *= 0.75
                    next_ratio = current_ratio + current_step * 0.25
                    print(f"Making progress ({percentage_solved:.2f}), small advance to {next_ratio:.2f}")
                else:  # Struggling significantly
                    # Reduce step size but don't change ratio
                    current_step *= 0.5
                    print(f"Struggling ({percentage_solved:.2f}), reducing step size to {current_step:.3f}")
            
            # Check if we've reached the target ratio
            if current_ratio >= self.target_ratio:
                print("Reached target difficulty!")
                break
                
            # Check if step size has become too small
            if current_step < 0.01:
                print("Step size too small. Resetting to larger step and continuing.")
                current_step = self.initial_step_size * 0.5
            
            # Prepare for next level by performing knowledge transfer
            if next_ratio > current_ratio:
                level += 1
                
                # Create fresh agent for next difficulty level
                next_clauses = self.generate_sat_problem(self.n_vars, next_ratio)
                next_agent_type = self.determine_best_agent(next_ratio, 1)
                
                if next_agent_type == 'dqn':
                    config = self.get_agent_config('dqn', next_ratio, 1, next_clauses)
                    next_agent = DeepQLearningAgent(
                        self.n_vars, 
                        next_clauses,
                        epsilon=config.get('epsilon', 1.0),
                        epsilon_decay=config.get('epsilon_decay', 0.995),
                        learning_rate=config.get('learning_rate', 0.001)
                    )
                    
                    # Transfer knowledge from current to next level
                    self.transfer_knowledge(next_agent, current_ratio, next_ratio)
                    
                elif next_agent_type == 'gan':
                    config = self.get_agent_config('gan', next_ratio, 1, next_clauses)
                    next_agent = ImprovedSATGAN(
                        self.n_vars, 
                        next_clauses,
                        latent_dim=config.get('latent_dim', 32),
                        epochs=config.get('epochs', 100)
                    )
                    
                    # Transfer knowledge from current to next level
                    self.transfer_knowledge(next_agent, current_ratio, next_ratio)
            
            # Update ratio for next level
            current_ratio = next_ratio
        
        # Once curriculum is complete, try to solve target problem with best approach
        if self.best_satisfied < len(self.target_clauses):
            print("\n--- Final Level: Attempting target problem ---")
            
            # Try each agent type on the target problem
            best_satisfied = 0
            for agent_type in ['dqn', 'gan', 'ensemble']:
                print(f"Trying {agent_type} on target problem")
                _, solution, satisfied = self.solve_with_agent(
                    agent_type, self.target_clauses, self.target_ratio, 1
                )
                
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    self.best_solution = solution
            
            self.best_satisfied = best_satisfied
            
        # Visualize final learning curve
        self.visualize_learning_curve()
        
        return self.best_solution, self.best_satisfied