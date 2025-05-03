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


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Main Q-Network
        self.q_network = QNetwork(state_size, action_size)
        
        # Target Q-Network for stable learning targets
        self.target_network = QNetwork(state_size, action_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Track performance for adaptive training
        self.performance_history = []
        self.update_counter = 0
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return torch.argmax(action_values).item()
    
    def learn(self, batch_size=64):
        """Update Q-Network weights using sampled experience"""
        if len(self.memory) < batch_size:
            return
        
        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([experience[0] for experience in minibatch])
        actions = torch.LongTensor([experience[1] for experience in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch])
        dones = torch.FloatTensor([experience[4] for experience in minibatch]).unsqueeze(1)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update weights
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the Q-network model"""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a trained Q-network model"""
        self.q_network.load_state_dict(torch.load(filepath))
        self.update_target_network()
    
    def report_progress(self, episode, total_reward, avg_reward):
        """Report training progress"""
        self.performance_history.append(avg_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Avg Reward: {avg_reward}, Epsilon: {self.epsilon:.4f}")


# DeepQSATAgent - Specific implementation for SAT problems
class DeepQSATAgent:
    """Deep Q-Learning Agent for SAT problems"""
    
    def __init__(self, state_size, action_size, enable_logging=False, logs_dir=None, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  # Experience replay buffer
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Main Q-Network
        self.q_network = QNetwork(state_size, action_size)
        
        # Target Q-Network for stable learning targets
        self.target_network = QNetwork(state_size, action_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Tracking metrics
        self.performance_history = []
        self.update_counter = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.best_solution = None
        self.best_satisfaction_ratio = 0.0
        
        # Logging setup
        self.logger = None
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = SATRLLogger(logs_dir or "sat_rl_logs/dqn")
            self.step = wrap_agent_step(self.step, self.logger)
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return torch.argmax(action_values).item()
    
    def learn(self, batch_size=64):
        """Update Q-Network weights using sampled experience"""
        if len(self.memory) < batch_size:
            return
        
        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([experience[0] for experience in minibatch])
        actions = torch.LongTensor([experience[1] for experience in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch])
        dones = torch.FloatTensor([experience[4] for experience in minibatch]).unsqueeze(1)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update weights
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def step(self, env, state):
        """Take a step in the environment"""
        # Choose an action
        action = self.choose_action(state)
        
        # Take action in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Remember the experience
        self.remember(state, action, reward, next_state, done)
        
        # Learn from experiences
        self.learn()
        
        # Track total steps
        self.total_steps += 1
        
        return action, next_state, reward, done
    
    def train(self, env=None, episodes=1000, max_steps=1000, batch_size=64, 
              update_frequency=4, target_update=100, verbose=False):
        """Train the agent on the SAT environment"""
        if env is None:
            from src.sat_problems import SATEnvironment
            env = SATEnvironment()
        
        rewards = []
        steps_list = []
        
        for episode in range(1, episodes+1):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Take a step in the environment
                action, next_state, reward, done = self.step(env, state)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                # Update target network periodically
                if steps % target_update == 0:
                    self.update_target_network()
            
            # After each episode
            rewards.append(total_reward)
            steps_list.append(steps)
            self.episode_rewards.append(total_reward)
            
            # Track and report progress
            if verbose and episode % 10 == 0:
                avg_reward = sum(rewards[-10:]) / min(10, len(rewards))
                print(f"Episode: {episode}/{episodes}, Reward: {total_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, Steps: {steps}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return rewards, steps_list
    
    def save_model(self, filepath):
        """Save the Q-network model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
            'performance_history': self.performance_history
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained Q-network model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'memory' in checkpoint:
            self.memory = checkpoint['memory']
        if 'performance_history' in checkpoint:
            self.performance_history = checkpoint['performance_history']
        
        print(f"Model loaded from {filepath}")