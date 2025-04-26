"""
Deep Q-Learning for SAT problems.
This agent uses neural networks for function approximation instead of Q-tables.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import random
from collections import deque
import time


class DeepQLearningAgent:
    def __init__(self, n_vars, clauses, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.1, learning_rate=0.001, gamma=0.95, restart_callback=None):
        """
        Initialize a Deep Q-Learning agent for SAT problems.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for exploration
            epsilon_min: Minimum exploration rate
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
            restart_callback: Function to call to decide whether to restart (episode, best_satisfied) -> bool
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = 32
        self.restart_callback = restart_callback
        
        # Double the action space to handle setting vars to 0 or 1 explicitly
        self.action_size = 2 * n_vars  # For each var, can set to 0 or 1
        
        # Memory for experience replay
        self.memory = deque(maxlen=2000)
        
        # Initialize model
        self.model = self._build_model()
        
        # Track the best solution found
        self.best_assignment = None
        self.best_satisfied = 0

        # Performance tracking
        self.performance_history = {
            'episodes': [],
            'steps': [],
            'satisfied_clauses': [],
            'epsilon': [],
            'time_per_episode': []
        }
        self.last_improvement_episode = 0
    
    def _build_model(self):
        """Build deep Q-network using the Functional API"""
        inputs = Input(shape=(self.n_vars,))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.1)(x)  # Add dropout to prevent overfitting
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def reset_model(self):
        """Reset the model while keeping the same architecture"""
        print("Resetting model weights to random initialization")
        self.model = self._build_model()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape state for prediction
        state_reshaped = np.reshape(state, [1, self.n_vars])
        
        # Get action values
        act_values = self.model.predict(state_reshaped, verbose=0)
        
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the network using experience replay"""
        # Need enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Reshape states for prediction
            state_reshaped = np.reshape(state, [1, self.n_vars])
            next_state_reshaped = np.reshape(next_state, [1, self.n_vars])
            
            # If done, target is just the reward, otherwise include future discounted reward
            if done:
                target = reward
            else:
                # Double DQN approach for more stable learning
                a = np.argmax(self.model.predict(next_state_reshaped, verbose=0)[0])
                target = reward + self.gamma * self.model.predict(next_state_reshaped, verbose=0)[0][a]
            
            # Update the target for the chosen action
            target_f = self.model.predict(state_reshaped, verbose=0)
            target_f[0][action] = target
            
            # Train the network
            self.model.fit(state_reshaped, target_f, epochs=1, verbose=0)
        
        # Decay epsilon after each replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def apply_action(self, state, action):
        """Apply the chosen action to the state"""
        var_idx = action // 2  # Which variable to change
        new_value = action % 2  # 0 or 1
        
        # Create a copy of the state
        new_state = state.copy()
        
        # Apply the action
        new_state[var_idx] = new_value
        
        return new_state
    
    def count_satisfied_clauses(self, state):
        """Count how many clauses are satisfied by the current assignment"""
        satisfied = 0
        
        for clause in self.clauses:
            # Check if any literal in the clause is satisfied
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = state[var]
                
                # Check if the literal is satisfied
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        
        # Update best solution if better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_assignment = state.copy()
        
        return satisfied
    
    def get_reward(self, state):
        """Calculate reward based on satisfied clauses"""
        satisfied = self.count_satisfied_clauses(state)
        
        # Normalized reward between 0 and 1
        return satisfied / len(self.clauses)
    
    def should_restart(self, episode, best_satisfied):
        """Determine if a restart is needed based on progress"""
        if self.restart_callback:
            return self.restart_callback(episode, best_satisfied)
            
        # Default restart logic - no improvement for 5 episodes
        if episode > 5:
            episodes_without_improvement = episode - self.last_improvement_episode
            if episodes_without_improvement >= 5:
                return True
        return False
    
    def solve(self, max_episodes=1000, max_steps=200):
        """Solve SAT problem using Deep Q-Learning"""
        import time
        start_time = time.time()
        last_update = time.time()
        
        stats = {
            'episodes': 0,
            'steps': [],
            'rewards': [],
            'best_satisfied': 0,
            'time_per_episode': []
        }
        
        for episode in range(max_episodes):
            episode_start = time.time()
            
            # Check if restart is needed
            if self.should_restart(episode, self.best_satisfied):
                self.reset_model()
                # Bump up exploration after restart
                self.epsilon = min(0.8, self.epsilon * 2.0)
                print(f"Restarting with epsilon={self.epsilon:.4f}")
            
            # Initialize state (random assignment)
            state = np.random.randint(0, 2, size=self.n_vars)
            total_reward = 0
            
            for step in range(max_steps):
                # Choose action
                action = self.act(state)
                
                # Apply action and get next state
                next_state = self.apply_action(state, action)
                
                # Calculate reward
                reward = self.get_reward(next_state)
                total_reward += reward
                
                # Check if solution is found
                satisfied = self.count_satisfied_clauses(next_state)
                done = satisfied == len(self.clauses)
                
                # Store experience in memory
                self.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Train the network 
                # Only train every few steps to reduce computational overhead
                if step % 5 == 0 or done:
                    self.replay()
                
                # If solution found, terminate episode
                if done:
                    break
                
                # Add periodic progress indicator during long episodes
                current_time = time.time()
                if current_time - last_update > 30:  # Print every 30 seconds
                    print(f"  Still working... Step {step}/{max_steps}, "
                          f"Best so far: {self.best_satisfied}/{len(self.clauses)}, "
                          f"Elapsed: {current_time - start_time:.1f}s")
                    last_update = current_time
            
            # Update statistics
            episode_time = time.time() - episode_start
            stats['episodes'] += 1
            stats['steps'].append(step + 1)
            stats['rewards'].append(total_reward)
            stats['best_satisfied'] = self.best_satisfied
            stats['time_per_episode'].append(episode_time)
            
            # Track if we improved this episode
            if satisfied == self.best_satisfied:
                self.last_improvement_episode = episode
                
            # Update performance history
            self.performance_history['episodes'].append(episode)
            self.performance_history['steps'].append(step + 1)
            self.performance_history['satisfied_clauses'].append(self.best_satisfied)
            self.performance_history['epsilon'].append(self.epsilon)
            self.performance_history['time_per_episode'].append(episode_time)
            
            # Print progress
            print(f"Episode: {episode}, Steps: {step+1}, "
                  f"Best satisfied: {self.best_satisfied}/{len(self.clauses)}, "
                  f"Epsilon: {self.epsilon:.4f}, "
                  f"Time: {episode_time:.2f}s")
            
            # If solution found, terminate
            if self.best_satisfied == len(self.clauses):
                print(f"Solution found after {episode+1} episodes!")
                break
                
            # Early stopping if we've converged
            if episode >= 20 and episode - self.last_improvement_episode >= 15:
                print(f"Stopping early: No improvement for {episode - self.last_improvement_episode} episodes")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Average time per episode: {total_time/stats['episodes']:.2f} seconds")
        
        return self.best_assignment, stats
    
    def visualize_learning_curve(self):
        """Visualize the learning progress"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Satisfied clauses over episodes
        plt.subplot(2, 2, 1)
        plt.plot(self.performance_history['episodes'], 
                 self.performance_history['satisfied_clauses'])
        plt.title('Satisfied Clauses vs Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Satisfied Clauses')
        plt.grid(True)
        
        # Plot 2: Steps per episode
        plt.subplot(2, 2, 2)
        plt.plot(self.performance_history['episodes'], 
                 self.performance_history['steps'])
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        # Plot 3: Epsilon decay
        plt.subplot(2, 2, 3)
        plt.plot(self.performance_history['episodes'], 
                 self.performance_history['epsilon'])
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        # Plot 4: Time per episode
        plt.subplot(2, 2, 4)
        plt.plot(self.performance_history['episodes'], 
                 self.performance_history['time_per_episode'])
        plt.title('Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Time (s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dqn_learning_curve.png')
        plt.close()
        print("Learning curve visualization saved to dqn_learning_curve.png")