#!/usr/bin/env python3
"""
SAT RL Logger - Comprehensive logging utility for SAT RL agents

This module provides a specialized logger for tracking the behavior of
reinforcement learning agents solving SAT problems, including:
- State transitions
- Action selection
- Rewards
- Performance metrics
- Training progress
"""

import os
import time
import json
import csv
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import functools
import matplotlib.pyplot as plt


class SATRLLogger:
    """Logger for SAT RL agents that tracks states, actions, rewards and metrics."""
    
    def __init__(self, max_entries=10000, log_to_file=True, logs_dir="sat_rl_logs"):
        """
        Initialize the logger.
        
        Args:
            max_entries: Maximum number of transitions to store in memory
            log_to_file: Whether to write logs to disk
            logs_dir: Directory to store log files
        """
        # Storage for traces
        self.traces = deque(maxlen=max_entries)
        self.episodes = defaultdict(list)  # Episode ID -> list of trace indices
        self.current_episode = 0
        
        # File logging configuration
        self.log_to_file = log_to_file
        self.logs_dir = logs_dir
        self.log_file = None
        self.csv_writer = None
        
        # Tracking various metrics
        self.start_time = time.time()
        self.total_steps = 0
        self.total_rewards = 0
        self.solved_episodes = 0
        self.total_episodes = 0
        
        # Statistics
        self.action_counts = defaultdict(int)
        self.state_transition_counts = defaultdict(int)
        self.reward_history = []
        
        # Initialize logging directory and files if needed
        if log_to_file:
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_file_path = os.path.join(logs_dir, f"sat_rl_log_{timestamp}.csv")
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the log file with headers."""
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp', 'episode', 'step', 'state', 'action', 
            'reward', 'next_state', 'done', 'info'
        ])
        self.log_file.flush()
    
    def log_step(self, state, action, reward, next_state, done, info=None, episode=None):
        """
        Log a single step of the agent.
        
        Args:
            state: Current state representation
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode is complete
            info: Additional information dictionary (optional)
            episode: Episode ID (if None, uses internal counter)
        """
        if episode is None:
            episode = self.current_episode
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.tolist()
            
        timestamp = time.time()
        
        # Create the trace entry
        trace = {
            'timestamp': timestamp,
            'episode': episode,
            'step': len(self.episodes[episode]),
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }
        
        # Add to in-memory storage
        self.traces.append(trace)
        self.episodes[episode].append(len(self.traces) - 1)
        
        # Update statistics
        self.total_steps += 1
        self.total_rewards += reward
        self.action_counts[action] += 1
        
        # Track state transitions (simplified for high-dimensional states)
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        self.state_transition_counts[(state_key, action, next_state_key)] += 1
        
        # Write to log file if enabled
        if self.log_to_file and self.csv_writer:
            self.csv_writer.writerow([
                timestamp, episode, trace['step'],
                json.dumps(state), action, reward,
                json.dumps(next_state), done,
                json.dumps(info) if info else ''
            ])
            # Periodically flush to ensure data is written
            if self.total_steps % 100 == 0:
                self.log_file.flush()
        
        # If episode is done, update episode statistics
        if done:
            if episode == self.current_episode:
                self.current_episode += 1
            self.total_episodes += 1
            self.reward_history.append(sum(t['reward'] for t in 
                                      [self.traces[i] for i in self.episodes[episode]]))
            
            # Check if the episode was successful (SAT was solved)
            if info and info.get('solved', False):
                self.solved_episodes += 1
    
    def _get_state_key(self, state):
        """
        Convert a state to a hashable key for counting transitions.
        For high-dimensional states, this might use dimensionality reduction.
        """
        if isinstance(state, (list, np.ndarray)) and len(state) > 10:
            # For large states, we'll use a hash of the state
            state_arr = np.array(state)
            return hash(state_arr.tobytes())
        return str(state)
    
    def log_episode_end(self, episode_metrics=None):
        """
        Log the end of an episode with additional metrics.
        
        Args:
            episode_metrics: Dictionary of metrics for the episode
        """
        if episode_metrics:
            # Store metrics in the episode's information
            episode_id = self.current_episode - 1
            for trace_idx in self.episodes[episode_id]:
                self.traces[trace_idx]['info'].update({
                    'episode_metrics': episode_metrics
                })
        
        # Log a summary to file if enabled
        if self.log_to_file and self.csv_writer:
            # Add a separator line for readability
            self.csv_writer.writerow(['---Episode End---', '', '', '', '', '', '', '', ''])
            self.log_file.flush()
    
    def get_statistics(self):
        """Return a dictionary of overall statistics."""
        runtime = time.time() - self.start_time
        
        stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'solved_episodes': self.solved_episodes,
            'solve_rate': self.solved_episodes / max(1, self.total_episodes),
            'total_rewards': self.total_rewards,
            'avg_reward_per_step': self.total_rewards / max(1, self.total_steps),
            'avg_reward_per_episode': self.total_rewards / max(1, self.total_episodes),
            'avg_steps_per_episode': self.total_steps / max(1, self.total_episodes),
            'runtime_seconds': runtime,
            'steps_per_second': self.total_steps / max(0.001, runtime),
        }
        
        # Add action distribution statistics
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            action_distribution = {
                f'action_{action}': count / total_actions 
                for action, count in self.action_counts.items()
            }
            stats.update(action_distribution)
        
        # Add recent performance trend
        recent_rewards = self.reward_history[-10:] if len(self.reward_history) > 10 else self.reward_history
        if recent_rewards:
            stats['recent_avg_reward'] = sum(recent_rewards) / len(recent_rewards)
        
        return stats
    
    def export_traces_to_csv(self, filename=None):
        """
        Export all traces to a CSV file.
        
        Args:
            filename: Name of the CSV file to export to (if None, generates one)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"sat_rl_traces_{timestamp}.csv"
        
        export_path = os.path.join(self.logs_dir, filename)
        
        with open(export_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'episode', 'step', 'state', 'action', 
                'reward', 'next_state', 'done', 'info'
            ])
            
            for trace in self.traces:
                writer.writerow([
                    trace['timestamp'], trace['episode'], trace['step'],
                    json.dumps(trace['state']), trace['action'], trace['reward'],
                    json.dumps(trace['next_state']), trace['done'],
                    json.dumps(trace['info'])
                ])
        
        return export_path
    
    def load_traces_from_csv(self, filepath):
        """
        Load traces from a CSV file.
        
        Args:
            filepath: Path to the CSV file to load from
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header row
                
                # Clear existing traces
                self.traces.clear()
                self.episodes.clear()
                
                for row in reader:
                    if row[0] == '---Episode End---':
                        continue
                        
                    timestamp, episode, step = float(row[0]), int(row[1]), int(row[2])
                    state = json.loads(row[3])
                    action = int(row[4])
                    reward = float(row[5])
                    next_state = json.loads(row[6])
                    done = row[7].lower() == 'true'
                    info = json.loads(row[8]) if row[8] else {}
                    
                    trace = {
                        'timestamp': timestamp,
                        'episode': episode,
                        'step': step,
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'info': info
                    }
                    
                    self.traces.append(trace)
                    self.episodes[episode].append(len(self.traces) - 1)
                    
                    # Update statistics
                    self.action_counts[action] += 1
                    
                # Set current episode to max episode + 1
                if self.episodes:
                    self.current_episode = max(self.episodes.keys()) + 1
                else:
                    self.current_episode = 0
                    
                # Recalculate episode rewards
                self.reward_history = []
                for ep_id, trace_indices in self.episodes.items():
                    ep_reward = sum(self.traces[i]['reward'] for i in trace_indices)
                    self.reward_history.append(ep_reward)
                
                # Update overall statistics
                self.total_steps = len(self.traces)
                self.total_rewards = sum(t['reward'] for t in self.traces)
                self.total_episodes = len(self.episodes)
                self.solved_episodes = sum(
                    1 for ep_id, indices in self.episodes.items()
                    if any(self.traces[i]['info'].get('solved', False) for i in indices)
                )
                
            return True
            
        except Exception as e:
            print(f"Error loading traces from CSV: {e}")
            return False
    
    def visualize_rewards(self, save_path=None, show=False):
        """
        Visualize rewards over episodes.
        
        Args:
            save_path: Path to save the visualization (if None, doesn't save)
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot episode rewards
        episodes = list(range(len(self.reward_history)))
        plt.plot(episodes, self.reward_history, marker='o', linestyle='-', markersize=4)
        
        # Add a trendline
        if len(episodes) > 1:
            z = np.polyfit(episodes, self.reward_history, 1)
            p = np.poly1d(z)
            plt.plot(episodes, p(episodes), "r--", alpha=0.8)
            
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    def visualize_action_distribution(self, save_path=None, show=False):
        """
        Visualize distribution of actions taken.
        
        Args:
            save_path: Path to save the visualization (if None, doesn't save)
            show: Whether to display the plot
        """
        if not self.action_counts:
            return
            
        plt.figure(figsize=(10, 6))
        
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())
        
        # Sort by action ID for clearer visualization
        sorted_actions, sorted_counts = zip(*sorted(zip(actions, counts)))
        
        plt.bar(range(len(sorted_actions)), sorted_counts)
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.title("Action Distribution")
        
        # If there are many actions, limit the tick labels
        if len(sorted_actions) > 20:
            plt.xticks(range(0, len(sorted_actions), len(sorted_actions) // 10))
        else:
            plt.xticks(range(len(sorted_actions)), sorted_actions)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    def visualize_state_transitions(self, top_n=10, save_path=None, show=False):
        """
        Visualize most common state transitions.
        
        Args:
            top_n: Number of most frequent transitions to show
            save_path: Path to save the visualization (if None, doesn't save)
            show: Whether to display the plot
        """
        if not self.state_transition_counts:
            return
            
        # Get the top N most frequent transitions
        sorted_transitions = sorted(
            self.state_transition_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        plt.figure(figsize=(12, 6))
        
        # Format transition labels and get counts
        labels = [f"{s[:5]}→{a}→{ns[:5]}" for (s, a, ns), _ in sorted_transitions]
        counts = [count for _, count in sorted_transitions]
        
        plt.barh(range(len(labels)), counts)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Count")
        plt.ylabel("Transition (State→Action→NextState)")
        plt.title(f"Top {top_n} Most Frequent State Transitions")
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    def close(self):
        """Close the logger and all open files."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None


def wrap_agent_step(agent, env, logger=None):
    """
    Wraps an agent's step method to automatically log transitions.
    
    Args:
        agent: Agent object with a step method
        env: Environment object (used for getting episode info)
        logger: Logger instance (creates one if None)
        
    Returns:
        agent: The modified agent
        logger: The logger instance
    """
    if logger is None:
        logger = SATRLLogger()
    
    # Store original step method
    original_step = agent.step
    
    @functools.wraps(original_step)
    def wrapped_step(env, state, **kwargs):
        # Call the original step method
        action, next_state, reward, done, info = original_step(env, state, **kwargs)
        
        # Log the step
        episode = kwargs.get('episode', logger.current_episode)
        logger.log_step(state, action, reward, next_state, done, info, episode)
        
        # If episode ends, log the end
        if done:
            episode_metrics = {
                'total_reward': sum(t['reward'] for t in 
                               [logger.traces[i] for i in logger.episodes[episode]]),
                'steps': len(logger.episodes[episode]),
                'solved': info.get('solved', False) if info else False
            }
            logger.log_episode_end(episode_metrics)
        
        return action, next_state, reward, done, info
    
    # Replace the agent's step method with our wrapped version
    agent.step = wrapped_step
    agent.logger = logger
    
    return agent, logger