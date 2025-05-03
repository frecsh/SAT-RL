"""
Base agent interface for SAT problem solving with RL.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AgentBase(ABC):
    """
    Abstract base class for SAT-solving agents.
    Defines the standard interface for all agents in the system.
    """
    
    def __init__(
        self, 
        num_vars: int,
        action_space_size: int,
        observation_shape: Dict[str, Any],
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent.
        
        Args:
            num_vars: Number of variables in the SAT problem
            action_space_size: Size of the action space
            observation_shape: Shape of the observation space
            agent_config: Configuration parameters for the agent
        """
        self.num_vars = num_vars
        self.action_space_size = action_space_size
        self.observation_shape = observation_shape
        self.agent_config = agent_config or {}
        
        # Initialize statistics
        self.stats = {
            'episodes_completed': 0,
            'steps_taken': 0,
            'successful_episodes': 0,
            'agent_type': self.__class__.__name__,
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'avg_satisfaction_ratio': 0.0
        }
    
    @abstractmethod
    def act(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> int:
        """
        Choose an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            deterministic: Whether to act deterministically (e.g., during evaluation)
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def train(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the agent using an experience tuple.
        
        Args:
            experience: Dictionary containing experience data
                - observation: Current observation
                - action: Action taken
                - reward: Reward received
                - next_observation: Next observation
                - done: Whether the episode is done
                - info: Additional information
                
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def end_episode(self, episode_info: Dict[str, Any]) -> None:
        """
        Signal the end of an episode to the agent.
        
        Args:
            episode_info: Information about the completed episode
        """
        # Update statistics
        self.stats['episodes_completed'] += 1
        self.stats['steps_taken'] += episode_info.get('steps_taken', 0)
        
        # Track success
        if episode_info.get('is_satisfied', False):
            self.stats['successful_episodes'] += 1
            
        # Update averages
        if self.stats['episodes_completed'] > 0:
            self.stats['avg_reward'] = ((self.stats['avg_reward'] * (self.stats['episodes_completed'] - 1) + 
                                        episode_info.get('total_reward', 0)) / 
                                        self.stats['episodes_completed'])
            
            self.stats['avg_steps'] = self.stats['steps_taken'] / self.stats['episodes_completed']
            
            self.stats['avg_satisfaction_ratio'] = ((self.stats['avg_satisfaction_ratio'] * 
                                                  (self.stats['episodes_completed'] - 1) + 
                                                  episode_info.get('satisfaction_ratio', 0)) / 
                                                  self.stats['episodes_completed'])
    
    def reset(self) -> None:
        """
        Reset the agent's state (e.g., at the start of a new episode).
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to disk.
        
        Args:
            path: Path to save the agent state
        """
        logger.info(f"Saving agent state to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from disk.
        
        Args:
            path: Path from which to load the agent state
        """
        logger.info(f"Loading agent state from {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's performance.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the agent with the given parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.agent_config.update(config)
        
    @abstractmethod
    def export_model(self) -> Dict[str, Any]:
        """
        Export the agent's model for external use.
        
        Returns:
            Dictionary containing the model data
        """
        pass