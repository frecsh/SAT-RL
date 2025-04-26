"""
Deep Q-Learning Agent for SAT solving using neural network function approximation.
This improves scalability over tabular Q-learning approaches.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random
from collections import deque


class DeepQLearningAgent:
    def __init__(self, n_vars, clauses, learning_rate=0.001, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        """
        Initialize a Deep Q-Learning agent for SAT problems.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            learning_rate: Learning rate for the neural network
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which to decay epsilon
            epsilon_min: Minimum exploration rate
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Input dimension is the assignment state (n_vars)
        # Output dimension is 2*n_vars (flip each variable to 0 or 1)
        self.state_dim = n_vars
        self.action_dim = 2 * n_vars
        
        # Create experience replay buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Build Q-network
        self.model = self._build_network(learning_rate)
        
        # Track best assignment found
        self.best_assignment = None
        self.best_satisfied = 0
    
    def _build_network(self, learning_rate):
        """Build neural network for Q-value function approximation"""
        # Use Input layer explicitly instead of passing input_shape to Dense
        inputs = Input(shape=(self.state_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return np.random.randint(self.action_dim)
        else:
            # Exploitation: choose best action according to Q-values
            q_values = self.model.predict(np.array([state]), verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        
        # Predict Q-values for current and next states
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Update Q-values with the Bellman equation
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(next_q_values[i])
            
            q_values[i][action] = target
        
        # Train the model with updated Q-values
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def count_satisfied_clauses(self, assignment):
        """Count number of clauses satisfied by an assignment"""
        satisfied = 0
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = assignment[var]
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        return satisfied
    
    def get_reward(self, assignment):
        """Calculate reward based on number of satisfied clauses"""
        satisfied = self.count_satisfied_clauses(assignment)
        
        # Update best assignment if current is better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_assignment = assignment.copy()
        
        # Return normalized reward
        return satisfied / len(self.clauses)
    
    def apply_action(self, state, action):
        """Apply action to state to get next state"""
        var_idx = action // 2
        new_value = action % 2
        
        next_state = state.copy()
        next_state[var_idx] = new_value
        
        return next_state
    
    def solve(self, max_episodes=1000, max_steps=200):
        """Solve SAT problem using Deep Q-Learning"""
        stats = {
            'episodes': 0,
            'steps': [],
            'rewards': [],
            'best_satisfied': 0
        }
        
        for episode in range(max_episodes):
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
                self.replay()
                
                # If solution found, terminate episode
                if done:
                    break
            
            # Update statistics
            stats['episodes'] += 1
            stats['steps'].append(step + 1)
            stats['rewards'].append(total_reward)
            stats['best_satisfied'] = self.best_satisfied
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Episode: {episode}, Steps: {step+1}, "
                      f"Best satisfied: {self.best_satisfied}/{len(self.clauses)}, "
                      f"Epsilon: {self.epsilon:.4f}")
            
            # If solution found, terminate
            if self.best_satisfied == len(self.clauses):
                print(f"Solution found after {episode+1} episodes!")
                break
        
        return self.best_assignment, stats