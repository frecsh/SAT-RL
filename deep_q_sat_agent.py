"""
Deep Q-Learning for SAT problems.
This agent uses neural networks for function approximation instead of Q-tables.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
from sat_rl_logger import SATRLLogger, wrap_agent_step


class DeepQSATAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=2000, batch_size=32, enable_logging=True, logs_dir='sat_rl_logs/dqn'):
        """
        Initialize a Deep Q-Learning agent for SAT problems.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            hidden_size: Number of hidden units in the neural network
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Maximum size of the replay memory
            batch_size: Batch size for training
            enable_logging: Whether to enable logging
            logs_dir: Directory to save logs
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Initialize model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize logger if enabled
        self.enable_logging = enable_logging
        self.logger = None
        self.original_step = None
        if enable_logging:
            self.logger = SATRLLogger(max_entries=10000, log_to_file=True, logs_dir=logs_dir)
            # Store the original step function before wrapping
            self.original_step = self.step
            # Wrap the step function
            _, self.logger = wrap_agent_step(self, None, self.logger)
    
    def _build_model(self):
        """Build the neural network model"""
        model = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        return model
    
    def update_target_model(self):
        """Update the target model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def step(self, env, state, episode=None):
        """Perform a single step in the environment"""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            action = torch.argmax(action_values).item()
        
        next_state, reward, done, _ = env.step(action)
        next_state = self._preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
        return action, next_state, reward, done
    
    def _preprocess_state(self, state):
        """Preprocess the state (e.g., normalization)"""
        return np.array(state)
    
    def replay(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        
        # Compute target values
        with torch.no_grad():
            target_q_values = self.target_model(next_states_tensor)
            max_target_q_values = torch.max(target_q_values, dim=1)[0]
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_target_q_values
        
        # Compute current Q values
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env, episodes, max_steps_per_episode=1000, verbose=True):
        """Train the agent on the given environment"""
        all_rewards = []
        all_steps = []
        start_time = time.time()
        
        for episode in range(episodes):
            state = env.reset()
            state = self._preprocess_state(state)
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                # Use the step function (will use wrapped version if logging is enabled)
                action, next_state, reward, done = self.step(env, state, episode=episode)
                
                self.replay()
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update epsilon and track statistics
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            all_rewards.append(episode_reward)
            all_steps.append(step + 1)
            
            if verbose:
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Steps: {step + 1}, Epsilon: {self.epsilon:.4f}")
            
            # Export logs periodically if logging is enabled
            if self.enable_logging and episode > 0 and episode % 10 == 0:
                log_path = self.logger.export_traces_to_csv(f"dqn_train_episode_{episode}.csv")
                if verbose:
                    print(f"Exported logs to {log_path}")
        
        # Final export of logs
        if self.enable_logging:
            final_log_path = self.logger.export_traces_to_csv(f"dqn_train_final.csv")
            if verbose:
                print(f"Final logs exported to {final_log_path}")
                
                # Print statistics from logger
                stats = self.logger.get_statistics()
                print("\nTraining Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        return all_rewards, all_steps
    
    def save_model(self, path):
        """Save the model weights to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path):
        """Load the model weights from a file"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")
            
    def disable_logging(self):
        """Disable logging and restore original step function"""
        if self.enable_logging and self.original_step is not None:
            self.step = self.original_step
            self.enable_logging = False
            print("Logging disabled, original step function restored")
    
    def enable_logging_with_new_logger(self, logs_dir=None):
        """Enable logging with a new logger instance"""
        if logs_dir is None:
            logs_dir = f"sat_rl_logs/dqn_{int(time.time())}"
        
        self.logger = SATRLLogger(max_entries=10000, log_to_file=True, logs_dir=logs_dir)
        # Store original step if not already stored
        if self.original_step is None:
            self.original_step = self.step
        # Wrap the step function
        _, self.logger = wrap_agent_step(self, None, self.logger)
        self.enable_logging = True
        print(f"Logging enabled with new logger to {logs_dir}")