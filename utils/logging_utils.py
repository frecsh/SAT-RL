"""
Structured logging utilities for SAT solving.

Provides standardized logging formats (JSON, CSV) for recording solver events,
rewards, clause satisfaction, agent decisions, and performance metrics.
"""

import logging
import json
import csv
import os
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, TextIO, Callable
import threading

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


class StructuredLogger:
    """
    A structured logger for SAT solving that supports JSON and CSV formats.
    
    This class provides methods to log various events in a structured format
    that can be easily parsed and analyzed later. It's designed to work with
    the agent behavior visualization notebook.
    """
    
    def __init__(self, log_dir="logs", experiment_name=None, visualize_ready=True):
        """
        Initialize the structured logger.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the experiment (if None, uses timestamp)
            visualize_ready: Whether to save data in a format ready for visualization
        """
        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        # Initialize visualization readiness flag
        self.visualize_ready = visualize_ready
        
        # Initialize data collection containers
        self.agent_decisions = []
        self.variable_assignments = []
        self.rewards = []
        self.clause_counts = []
        self.agent_states = []
        
        # Initialize log files
        self.agent_decisions_file = os.path.join(log_dir, f"{self.experiment_name}_agent_decisions.csv")
        self.rewards_file = os.path.join(log_dir, f"{self.experiment_name}_rewards.csv")
        self.oracle_guidance_file = os.path.join(log_dir, f"{self.experiment_name}_oracle_guidance.csv")
        self.performance_file = os.path.join(log_dir, f"{self.experiment_name}_performance.csv")
        
        # Visualization-specific files
        self.viz_metadata_file = os.path.join(log_dir, f"{self.experiment_name}_viz_metadata.json")
        self.variable_assignments_file = os.path.join(log_dir, f"{self.experiment_name}_variable_assignments_viz.csv")
        self.clause_counts_file = os.path.join(log_dir, f"{self.experiment_name}_clause_counts_viz.csv")
        self.agent_behavior_file = os.path.join(log_dir, f"{self.experiment_name}_agent_behavior_viz.csv")
        
        # Initialize CSV headers
        self._initialize_log_files()
    
    def _initialize_log_files(self):
        """Initialize CSV headers for each log file."""
        # Create CSV headers for each log file
        with open(self.agent_decisions_file, 'w') as f:
            f.write("episode,step,agent_id,variable_id,action,activity_score\n")
            
        with open(self.rewards_file, 'w') as f:
            f.write("episode,step,reward,clause_satisfaction,conflict_count\n")
        
        # Initialize visualization-specific files if enabled
        if self.visualize_ready:
            with open(self.variable_assignments_file, 'w') as f:
                f.write("timestamp,episode,variable_idx,assignment,agent_id,x_position,y_position\n")
                
            with open(self.clause_counts_file, 'w') as f:
                f.write("timestamp,episode,satisfied_count,total_count,satisfaction_rate,agent_id\n")
                
            with open(self.agent_behavior_file, 'w') as f:
                f.write("timestamp,agent_id,episode,x_position,y_position,energy,success_rate,state\n")
    
    def log_agent_decision(self, episode, step, agent_id, variable_id, action, activity_score=None):
        """
        Log an agent decision.
        
        Args:
            episode: Current episode number
            step: Current step within the episode
            agent_id: ID of the agent making the decision
            variable_id: ID of the variable being assigned
            action: Action taken (0 or 1, representing False or True)
            activity_score: Optional activity score for the variable
        """
        # Collect data for later analysis
        decision_data = {
            'episode': episode,
            'step': step,
            'agent_id': agent_id,
            'variable_id': variable_id,
            'action': action,
            'activity_score': activity_score or 0.0
        }
        self.agent_decisions.append(decision_data)
        
        # Write to CSV
        with open(self.agent_decisions_file, 'a') as f:
            f.write(f"{episode},{step},{agent_id},{variable_id},{action},{activity_score or 0.0}\n")
        
        # Also log as a variable assignment for visualization
        if self.visualize_ready:
            self.log_variable_assignment(step, variable_id, action, agent_id, episode)
    
    def log_variable_assignment(self, step, variable_idx, assignment, agent_id=0, episode=0):
        """
        Log a variable assignment specifically for visualization purposes.
        
        Args:
            step: Current step
            variable_idx: Index of the variable being assigned
            assignment: Value assigned to the variable (0=False, 1=True, -1=Unassigned)
            agent_id: ID of the agent making the assignment
            episode: Current episode number
        """
        # Collect data
        assignment_data = {
            'timestamp': step,
            'episode': episode,
            'variable_idx': variable_idx,
            'assignment': assignment,
            'agent_id': agent_id,
            'x_position': variable_idx,  # For visualization - position in variable space
            'y_position': episode,       # For visualization - episode as y-coordinate
        }
        self.variable_assignments.append(assignment_data)
        
        # Write to CSV if visualization is enabled
        if self.visualize_ready:
            with open(self.variable_assignments_file, 'a') as f:
                f.write(f"{step},{episode},{variable_idx},{assignment},{agent_id},{variable_idx},{episode}\n")
    
    def log_reward(self, episode, step, reward, clause_satisfaction=None, conflict_count=None, agent_id=0):
        """
        Log a reward event.
        
        Args:
            episode: Current episode number
            step: Current step within the episode
            reward: Received reward value
            clause_satisfaction: Percentage or count of satisfied clauses
            conflict_count: Number of conflicts
            agent_id: ID of the agent receiving the reward
        """
        # Collect data
        reward_data = {
            'episode': episode,
            'step': step,
            'reward': reward,
            'clause_satisfaction': clause_satisfaction,
            'conflict_count': conflict_count,
            'agent_id': agent_id
        }
        self.rewards.append(reward_data)
        
        # Write to CSV
        with open(self.rewards_file, 'a') as f:
            cs_val = clause_satisfaction if clause_satisfaction is not None else ""
            cc_val = conflict_count if conflict_count is not None else ""
            f.write(f"{episode},{step},{reward},{cs_val},{cc_val}\n")
        
        # Log clause counts separately for visualization if provided
        if clause_satisfaction is not None and self.visualize_ready:
            self.log_clause_count(step, int(clause_satisfaction), 100, agent_id, episode)
    
    def log_clause_count(self, step, satisfied_count, total_count, agent_id=0, episode=0):
        """
        Log clause satisfaction counts.
        
        Args:
            step: Current step
            satisfied_count: Number of satisfied clauses
            total_count: Total number of clauses
            agent_id: ID of the agent
            episode: Current episode number
        """
        satisfaction_rate = satisfied_count / total_count if total_count > 0 else 0
        
        # Collect data
        count_data = {
            'timestamp': step,
            'episode': episode,
            'satisfied_count': satisfied_count,
            'total_count': total_count,
            'satisfaction_rate': satisfaction_rate,
            'agent_id': agent_id
        }
        self.clause_counts.append(count_data)
        
        # Write to CSV if visualization is enabled
        if self.visualize_ready:
            with open(self.clause_counts_file, 'a') as f:
                f.write(f"{step},{episode},{satisfied_count},{total_count},{satisfaction_rate},{agent_id}\n")
    
    def log_agent_state(self, step, state_dict, agent_id=0, episode=0):
        """
        Log agent state information for visualization.
        
        Args:
            step: Current step
            state_dict: Dictionary containing agent state information
            agent_id: ID of the agent
            episode: Current episode number
        """
        # Extract visualization-relevant info from state_dict or use defaults
        x_position = state_dict.get('x_position', agent_id * 2)  # Default x-position based on agent_id
        y_position = state_dict.get('y_position', step % 10)     # Default y-position
        energy = state_dict.get('energy', 100 - (step % 100))    # Default energy level
        success_rate = state_dict.get('success_rate', min(0.5 + step/1000, 0.95))  # Default success rate
        state = state_dict.get('state', 'exploring')             # Default state
        
        # Collect data
        state_data = {
            'timestamp': step,
            'agent_id': agent_id,
            'episode': episode,
            'x_position': x_position,
            'y_position': y_position,
            'energy': energy,
            'success_rate': success_rate,
            'state': state,
            'full_state': state_dict  # Store the full state dict for reference
        }
        self.agent_states.append(state_data)
        
        # Write to CSV if visualization is enabled
        if self.visualize_ready:
            with open(self.agent_behavior_file, 'a') as f:
                f.write(f"{step},{agent_id},{episode},{x_position},{y_position},{energy},{success_rate},{state}\n")
    
    def save_visualization_metadata(self):
        """Save metadata for visualization tools."""
        if not self.visualize_ready:
            return
            
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'data_files': {
                'variable_assignments': self.variable_assignments_file,
                'clause_counts': self.clause_counts_file,
                'agent_behavior': self.agent_behavior_file,
                'agent_decisions': self.agent_decisions_file,
                'rewards': self.rewards_file
            },
            'stats': {
                'num_episodes': len(set(d['episode'] for d in self.rewards)) if self.rewards else 0,
                'num_steps': sum(1 for _ in self.rewards) if self.rewards else 0,
                'num_agents': len(set(d['agent_id'] for d in self.agent_states)) if self.agent_states else 0,
                'num_variables': len(set(d['variable_idx'] for d in self.variable_assignments)) if self.variable_assignments else 0
            }
        }
        
        with open(self.viz_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
    
    def get_dataframes(self):
        """
        Convert collected data to pandas DataFrames for easy analysis.
        
        Returns:
            Dict containing pandas DataFrames of logged data
        """
        return {
            'agent_decisions': pd.DataFrame(self.agent_decisions) if self.agent_decisions else pd.DataFrame(),
            'variable_assignments': pd.DataFrame(self.variable_assignments) if self.variable_assignments else pd.DataFrame(),
            'rewards': pd.DataFrame(self.rewards) if self.rewards else pd.DataFrame(),
            'clause_counts': pd.DataFrame(self.clause_counts) if self.clause_counts else pd.DataFrame(),
            'agent_states': pd.DataFrame(self.agent_states) if self.agent_states else pd.DataFrame()
        }
    
    def finalize(self):
        """Finalize logging and save any pending data."""
        self.save_visualization_metadata()
        logger.info(f"Structured logging finalized for experiment: {self.experiment_name}")
        
        # Return path to the visualization metadata file for convenience
        return self.viz_metadata_file


# Helper function to create a logger with default settings
def create_logger(experiment_name=None, format_type='json', output_dir='logs', visualize_ready=True):
    """
    Create a structured logger with default settings.
    
    Args:
        experiment_name: Name of the experiment
        format_type: 'json' or 'csv'
        output_dir: Directory to store log files
        visualize_ready: Whether to save data in a format ready for visualization
        
    Returns:
        A configured StructuredLogger instance
    """
    return StructuredLogger(
        log_dir=output_dir,
        experiment_name=experiment_name,
        visualize_ready=visualize_ready
    )