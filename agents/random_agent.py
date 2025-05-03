"""
Random agent implementation for SAT solving.
"""

import numpy as np
import random
from typing import Dict, Any, Optional
import logging

from .base_agent import AgentBase

# Set up logging
logger = logging.getLogger(__name__)

class RandomAgent(AgentBase):
    """
    A simple agent that selects actions randomly.
    Useful as a baseline for comparison.
    """
    
    def __init__(
        self, 
        num_vars: int,
        action_space_size: int,
        observation_shape: Dict[str, Any],
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the random agent.
        
        Args:
            num_vars: Number of variables in the SAT problem
            action_space_size: Size of the action space
            observation_shape: Shape of the observation space
            agent_config: Configuration parameters for the agent
        """
        super().__init__(num_vars, action_space_size, observation_shape, agent_config)
        
        # Additional initialization for random agent
        self.rng = random.Random(agent_config.get('seed', None))
        
        # Track variables that have been set
        self.assigned_variables = set()
        self.use_smart_random = agent_config.get('use_smart_random', True)
    
    def act(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> int:
        """
        Choose a random action from the action space.
        If use_smart_random is True, prioritize unassigned variables.
        
        Args:
            observation: Current observation from the environment
            deterministic: Whether to act deterministically (ignored for random agent)
            
        Returns:
            Random action
        """
        if not self.use_smart_random:
            # Completely random action
            return self.rng.randint(0, self.action_space_size - 1)
        
        # Smart random: try to choose unassigned variables first
        if 'assignment' in observation:
            assignment = observation['assignment']
            
            if len(assignment.shape) == 1 and assignment.shape[0] > self.num_vars:
                # This is a simple assignment array with 1-indexed variables
                unassigned = [i for i in range(1, self.num_vars + 1) if assignment[i] == 0]
            else:
                # This might be a binary encoding - we need to check both positive and negative slots
                unassigned = []
                for i in range(1, self.num_vars + 1):
                    var_idx = i - 1
                    neg_idx = var_idx + self.num_vars
                    
                    if var_idx < len(assignment) and neg_idx < len(assignment):
                        if assignment[var_idx] == 0 and assignment[neg_idx] == 0:
                            unassigned.append(i)
            
            if unassigned:
                # Choose a random unassigned variable
                var_id = self.rng.choice(unassigned)
                value = self.rng.choice([0, 1])  # Random True/False value
                return var_id - 1 + value * self.num_vars  # Convert to action index
                
        # Fall back to completely random action
        return self.rng.randint(0, self.action_space_size - 1)
    
    def train(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Random agents don't learn, so this is a no-op.
        
        Args:
            experience: Experience data
                
        Returns:
            Empty metrics dictionary
        """
        # Random agents don't train, so just return empty metrics
        return {"loss": 0.0}
    
    def reset(self) -> None:
        """
        Reset the agent's state.
        """
        self.assigned_variables = set()
    
    def export_model(self) -> Dict[str, Any]:
        """
        Export the agent's model (empty for random agent).
        
        Returns:
            Empty dictionary
        """
        return {
            "agent_type": "random",
            "config": self.agent_config
        }