"""
Improved Curriculum Learning for SAT Solving
This module implements adaptive curriculum learning strategies with
distribution matching and dynamic difficulty adjustment.
"""

import numpy as np
import time
import random
import os
import json
import matplotlib.pyplot as plt
from collections import deque

from anytime_sat_solver import AnytimeEnsembleSolver
from improved_sat_gan import ImprovedSATGAN


class ImprovedCurriculumLearning:
    def __init__(self, base_agent, config=None):
        """
        Initialize the improved curriculum learning module.
        
        Args:
            base_agent: The base RL agent to train
            config: Configuration dictionary with parameters
        """
        self.base_agent = base_agent
        
        # Default configuration
        default_config = {
            'min_vars': 10,
            'max_vars': 100,
            'min_clauses_per_var': 2.5,
            'max_clauses_per_var': 5.0,
            'difficulty_levels': 10,
            'episodes_per_level': 50,
            'success_threshold': 0.75,
            'failure_threshold': 0.3,
            'patience': 3,
            'dynamic_adjustment': True,
            'difficulty_decay': 0.9,
            'use_experience_replay': True,
            'replay_buffer_size': 10000,
            'replay_batch_size': 32,
            'use_ensemble_fallback': True,
            'ensemble_timeout': 30,
            'use_gan_generation': True,
            'gan_training_interval': 200,
            'save_checkpoints': True,
            'checkpoint_dir': 'checkpoints',
            'checkpoint_interval': 100,
            'log_dir': 'logs',
        }
        
        # Override defaults with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize state
        self.current_level = 0
        self.episodes_at_level = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.training_history = []
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=self.config['replay_buffer_size'])
        
        # Generate initial difficulty distribution
        self.difficulty_distribution = self._initialize_difficulty_distribution()
        
        # Initialize the GAN for generating new problems
        if self.config['use_gan_generation']:
            self.sat_gan = None  # Will be initialized when needed
            self.gan_examples = []
            self.gan_training_steps = 0
        
        # Create directories if they don't exist
        if self.config['save_checkpoints']:
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
    
    def _initialize_difficulty_distribution(self):
        """Initialize the difficulty distribution for curriculum learning"""
        levels = self.config['difficulty_levels']
        # Linear distribution from easiest to hardest
        distribution = np.ones(levels) / levels
        return distribution
    
    def _update_difficulty_distribution(self, success_rate):
        """Update difficulty distribution based on agent's success rate"""
        if not self.config['dynamic_adjustment']:
            return
        
        # Current level's index
        idx = self.current_level
        levels = self.config['difficulty_levels']
        
        # Update based on success rate
        if success_rate > self.config['success_threshold']:
            # Increase probability of harder levels
            for i in range(levels):
                if i <= idx:
                    # Decrease probability of easier levels
                    self.difficulty_distribution[i] *= self.config['difficulty_decay']
                else:
                    # Increase probability of harder levels
                    self.difficulty_distribution[i] /= self.config['difficulty_decay']
        else:
            # Increase probability of easier levels
            for i in range(levels):
                if i <= idx:
                    # Increase probability of easier levels
                    self.difficulty_distribution[i] /= self.config['difficulty_decay']
                else:
                    # Decrease probability of harder levels
                    self.difficulty_distribution[i] *= self.config['difficulty_decay']
        
        # Normalize to ensure it's a valid probability distribution
        self.difficulty_distribution /= np.sum(self.difficulty_distribution)
    
    def _generate_problem_for_level(self, level):
        """Generate a SAT problem with appropriate difficulty for the level"""
        # Scale number of variables based on level
        min_vars = self.config['min_vars']
        max_vars = self.config['max_vars']
        vars_range = max_vars - min_vars
        
        # Calculate linear interpolation based on level
        proportion = level / (self.config['difficulty_levels'] - 1)
        n_vars = int(min_vars + proportion * vars_range)
        
        # Determine clause-to-var ratio - gets closer to phase transition as level increases
        min_ratio = self.config['min_clauses_per_var']
        max_ratio = self.config['max_clauses_per_var']
        ratio_range = max_ratio - min_ratio
        
        # Make ratio approach the phase transition (4.26) as level increases
        target_ratio = 4.26  # The SAT phase transition point
        
        if proportion < 0.5:
            # First half of levels: easy problems (underconstrained)
            ratio = min_ratio + proportion * 2 * (target_ratio - min_ratio)
        else:
            # Second half of levels: hard problems (overconstrained)
            ratio = target_ratio + (proportion - 0.5) * 2 * (max_ratio - target_ratio)
        
        n_clauses = int(n_vars * ratio)
        
        # If using GAN for generation and we have trained it
        if self.config['use_gan_generation'] and self.sat_gan and self.gan_training_steps > 0:
            # Try to use GAN with some probability
            if random.random() < min(0.8, self.gan_training_steps / 500):
                try:
                    problem = self._generate_problem_with_gan(n_vars, n_clauses)
                    if problem:
                        return problem
                except Exception as e:
                    print(f"Error generating problem with GAN: {e}")
                    # Fall back to random generation
        
        # Random generation
        return self._generate_random_problem(n_vars, n_clauses)
    
    def _generate_random_problem(self, n_vars, n_clauses):
        """Generate a random SAT problem with the given parameters"""
        clauses = []
        for _ in range(n_clauses):
            # Generate a random clause with 3 literals
            clause = []
            
            while len(clause) < 3:
                # Select a random variable (1 to n_vars)
                var = random.randint(1, n_vars)
                
                # Randomly negate with 50% probability
                if random.random() < 0.5:
                    var = -var
                
                # Avoid duplicates in the same clause
                if var not in clause and -var not in clause:
                    clause.append(var)
            
            clauses.append(clause)
            
        return {
            'n_vars': n_vars,
            'clauses': clauses
        }
    
    def _generate_problem_with_gan(self, n_vars, n_clauses):
        """Generate a problem using the trained GAN"""
        if self.sat_gan is None:
            self._initialize_gan()
        
        # Generate a problem that's close to the target size
        min_acceptable = 0.8
        max_acceptable = 1.2
        
        # Try a few times to get a problem close to desired size
        for attempt in range(5):
            generated = self.sat_gan.generate_problem()
            
            # Check if the generated problem has appropriate size
            gen_vars = generated['n_vars']
            gen_clauses = len(generated['clauses'])
            
            if (min_acceptable * n_vars <= gen_vars <= max_acceptable * n_vars and
                min_acceptable * n_clauses <= gen_clauses <= max_acceptable * n_clauses):
                return generated
        
        # If no suitable problem was generated, create a new one manually
        return self._generate_random_problem(n_vars, n_clauses)
    
    def _initialize_gan(self):
        """Initialize the GAN for problem generation"""
        # Generate some initial problems for GAN training
        initial_examples = []
        for _ in range(50):  # Start with 50 examples
            level = random.randint(0, self.config['difficulty_levels'] - 1)
            problem = self._generate_random_problem(
                n_vars=random.randint(self.config['min_vars'], self.config['max_vars']),
                n_clauses=random.randint(
                    int(self.config['min_vars'] * self.config['min_clauses_per_var']),
                    int(self.config['max_vars'] * self.config['max_clauses_per_var'])
                )
            )
            initial_examples.append(problem)
        
        print("Initializing SAT-GAN...")
        self.sat_gan = ImprovedSATGAN(initial_examples)
        self.gan_examples = initial_examples.copy()
        self.gan_training_steps = 0
    
    def _update_gan(self, new_problems=None):
        """Update the GAN with new problem examples"""
        if not self.config['use_gan_generation']:
            return
            
        if self.sat_gan is None:
            self._initialize_gan()
        
        # Add new problems to training set
        if new_problems:
            self.gan_examples.extend(new_problems)
            
            # Limit the size of the example set
            max_examples = 500
            if len(self.gan_examples) > max_examples:
                # Keep more recent examples
                self.gan_examples = self.gan_examples[-max_examples:]
        
        # Train for some iterations
        if len(self.gan_examples) >= 20:  # Need enough examples
            print(f"Training GAN with {len(self.gan_examples)} examples...")
            self.sat_gan.train(
                problems=self.gan_examples,
                epochs=5,
                batch_size=min(16, len(self.gan_examples))
            )
            self.gan_training_steps += 1
    
    def _sample_from_replay_buffer(self, batch_size):
        """Sample transitions from the experience replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return []
            
        return random.sample(self.replay_buffer, batch_size)
    
    def _store_experience(self, problem, state, action, reward, next_state, done):
        """Store experience in the replay buffer"""
        if not self.config['use_experience_replay']:
            return
            
        # Store the transition
        self.replay_buffer.append({
            'problem': problem,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def _use_ensemble_solver(self, problem):
        """Use the ensemble solver to solve a difficult problem"""
        print("Attempting to solve with ensemble solver...")
        n_vars = problem['n_vars']
        clauses = problem['clauses']
        
        # Create the ensemble solver
        solver = AnytimeEnsembleSolver(
            n_vars=n_vars,
            clauses=clauses,
            time_limit=self.config['ensemble_timeout']
        )
        
        # Solve the problem
        solution, stats = solver.solve()
        
        return solution, stats
    
    def _implement_restart_callback(self):
        """Set up restart callback for the agent"""
        def restart_callback(problem, episode, timestep, rewards):
            # Check if we should restart
            if timestep > 100 and np.mean(rewards[-20:]) < 0.01:
                return True  # Signal restart
            return False  # Continue
        
        # Set callback in the agent
        self.base_agent.set_restart_callback(restart_callback)
    
    def train(self, total_episodes=1000):
        """Train the agent using curriculum learning"""
        # Implement restart callback
        self._implement_restart_callback()
        
        # Training metrics
        episode_rewards = []
        success_rates = []
        level_history = []
        difficulty_history = []
        
        # Initial success rate
        success_rate = 0
        
        for episode in range(total_episodes):
            if episode % 10 == 0:
                print(f"\nEpisode {episode}/{total_episodes} | Level: {self.current_level}")
                print(f"Success rate: {success_rate:.2f} | Difficulty distribution: {np.round(self.difficulty_distribution, 2)}")
            
            # Sample a difficulty level based on the distribution
            sampled_level = np.random.choice(
                range(self.config['difficulty_levels']),
                p=self.difficulty_distribution
            )
            
            # Generate a problem of the sampled difficulty
            problem = self._generate_problem_for_level(sampled_level)
            
            # Try to solve with the agent
            start_time = time.time()
            solution, reward, success = self.base_agent.solve(problem)
            solve_time = time.time() - start_time
            
            # Store metrics
            episode_rewards.append(reward)
            level_history.append(sampled_level)
            difficulty_history.append(problem['n_vars'] * len(problem['clauses']))
            
            # Train with experience replay
            if self.config['use_experience_replay'] and episode % 10 == 0:
                replay_batch = self._sample_from_replay_buffer(self.config['replay_batch_size'])
                if replay_batch:
                    for experience in replay_batch:
                        # Replay the experience
                        self.base_agent.learn_from_experience(
                            experience['state'], 
                            experience['action'],
                            experience['reward'],
                            experience['next_state'], 
                            experience['done']
                        )
            
            # Update GAN periodically
            if self.config['use_gan_generation'] and episode % self.config['gan_training_interval'] == 0:
                # Extract problems from the replay buffer to train the GAN
                replay_problems = []
                for exp in self._sample_from_replay_buffer(min(50, len(self.replay_buffer))):
                    if 'problem' in exp:
                        replay_problems.append(exp['problem'])
                
                self._update_gan(replay_problems)
            
            # If agent failed but we're using ensemble fallback
            if not success and self.config['use_ensemble_fallback'] and sampled_level > self.config['difficulty_levels'] // 2:
                print(f"Agent failed. Trying ensemble solver...")
                ensemble_solution, ensemble_stats = self._use_ensemble_solver(problem)
                
                if ensemble_stats['satisfied'] == len(problem['clauses']):
                    print("Ensemble solver found a solution!")
                    
                    # Use this solution to teach the agent
                    self.base_agent.learn_from_solution(problem, ensemble_solution)
                    
                    # Count as a partial success
                    success = 0.5
            
            # Update counters for level progression
            if success:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
                
                # Store this successful experience for later replay
                # (simplified - in practice would store actual experience tuples)
                if self.config['use_experience_replay'] and isinstance(success, bool):
                    self._store_experience(problem, None, None, reward, None, True)
            else:
                self.consecutive_successes = 0
                self.consecutive_failures += 1
                
            # Update episodes at current level
            self.episodes_at_level += 1
            
            # Check if we should move to the next level
            if (self.consecutive_successes >= self.config['patience'] and 
                self.current_level < self.config['difficulty_levels'] - 1):
                self.current_level += 1
                self.episodes_at_level = 0
                self.consecutive_successes = 0
                self.consecutive_failures = 0
                print(f"\n=== Advancing to level {self.current_level} ===")
            
            # Check if we should move back a level
            elif (self.consecutive_failures >= self.config['patience'] and
                  self.current_level > 0):
                self.current_level -= 1
                self.episodes_at_level = 0
                self.consecutive_successes = 0
                self.consecutive_failures = 0
                print(f"\n=== Moving back to level {self.current_level} ===")
            
            # Calculate success rate over recent episodes
            window = 50
            if episode >= window:
                success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-window:]])
            else:
                success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards])
            
            success_rates.append(success_rate)
            
            # Update difficulty distribution
            self._update_difficulty_distribution(success_rate)
            
            # Save checkpoint periodically
            if self.config['save_checkpoints'] and episode % self.config['checkpoint_interval'] == 0:
                self._save_checkpoint(episode)
                
        # Final metrics
        self.training_history = {
            'episode_rewards': episode_rewards,
            'success_rates': success_rates,
            'level_history': level_history,
            'difficulty_history': difficulty_history
        }
        
        # Save final visualization
        self._visualize_training_progress()
        
        return self.training_history
    
    def _save_checkpoint(self, episode):
        """Save a checkpoint of the current training state"""
        if not self.config['save_checkpoints']:
            return
            
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f"checkpoint_ep{episode}.json"
        )
        
        model_path = os.path.join(
            self.config['checkpoint_dir'],
            f"agent_model_ep{episode}"
        )
        
        # Save curriculum learning state
        state = {
            'episode': episode,
            'current_level': self.current_level,
            'difficulty_distribution': self.difficulty_distribution.tolist(),
            'config': self.config,
            'training_metrics': {
                'success_rates': self.training_history.get('success_rates', [])[-100:],
                'episode_rewards': self.training_history.get('episode_rewards', [])[-100:],
                'level_history': self.training_history.get('level_history', [])[-100:]
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        # Save agent model
        try:
            self.base_agent.save_model(model_path)
        except Exception as e:
            print(f"Error saving agent model: {e}")
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _visualize_training_progress(self):
        """Visualize the training progress"""
        if not self.training_history:
            print("No training history to visualize")
            return
            
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot episode rewards
        axs[0].plot(self.training_history['episode_rewards'])
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)
        
        # Plot success rates
        axs[1].plot(self.training_history['success_rates'])
        axs[1].set_title('Success Rate')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Success Rate')
        axs[1].set_ylim([0, 1])
        axs[1].grid(True)
        
        # Plot difficulty level
        axs[2].scatter(
            range(len(self.training_history['level_history'])), 
            self.training_history['level_history'],
            alpha=0.5
        )
        axs[2].set_title('Difficulty Levels')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Level')
        axs[2].set_ylim([-0.5, self.config['difficulty_levels'] - 0.5])
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'curriculum_learning_progress.png'))
        plt.close()
        
        print("Training visualization saved to curriculum_learning_progress.png")
        
    def load_checkpoint(self, checkpoint_path, model_path=None):
        """Load a training checkpoint to resume training"""
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
            
        self.current_level = state['current_level']
        self.difficulty_distribution = np.array(state['difficulty_distribution'])
        
        # Update config if provided
        if 'config' in state:
            self.config.update(state['config'])
            
        # Update training history if available
        if 'training_metrics' in state:
            metrics = state['training_metrics']
            self.training_history = {
                'episode_rewards': metrics.get('episode_rewards', []),
                'success_rates': metrics.get('success_rates', []),
                'level_history': metrics.get('level_history', [])
            }
        
        # Load agent model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self.base_agent.load_model(model_path)
                print(f"Loaded agent model from {model_path}")
            except Exception as e:
                print(f"Error loading agent model: {e}")
        
        print(f"Loaded checkpoint from {checkpoint_path}, resuming at level {self.current_level}")
        return state['episode']