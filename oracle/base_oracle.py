"""
Base oracle interface for providing guidance in SAT solving.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)

class OracleBase(ABC):
    """
    Abstract base class for symbolic oracles that provide guidance to RL agents.
    
    Oracles encapsulate symbolic knowledge about SAT problems and can be queried
    by agents for guidance on variable assignments, clause prioritization, etc.
    """
    
    def __init__(
        self,
        clauses: List[List[int]],
        num_vars: int,
        oracle_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the oracle.
        
        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the problem
            oracle_config: Configuration parameters for the oracle
        """
        self.clauses = clauses
        self.num_vars = num_vars
        self.oracle_config = oracle_config or {}
        
        # Initialize statistics
        self.stats = {
            'queries': 0,
            'oracle_type': self.__class__.__name__,
            'avg_query_time': 0.0,
        }
    
    @abstractmethod
    def query(self, 
             state: Dict[str, np.ndarray], 
             available_actions: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query the oracle for guidance based on the current state.
        
        Args:
            state: Current state observation (typically including variable assignment)
            available_actions: Optional list of available actions
            
        Returns:
            Dictionary containing oracle guidance:
                - recommended_action: Recommended action to take
                - action_values: Optional values for each action
                - confidence: Confidence in the recommendation (0-1)
                - explanation: Optional explanation for the recommendation
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the oracle's internal state between episodes.
        Default implementation resets only statistics.
        """
        self.stats['queries'] = 0
        self.stats['avg_query_time'] = 0.0
    
    def update(self, state: Dict[str, np.ndarray], action: int, 
              reward: float, next_state: Dict[str, np.ndarray],
              done: bool, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the oracle with new information from the environment.
        
        Useful for oracles that adapt their guidance based on agent behavior.
        Default implementation is a no-op.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode is complete
            info: Additional information
        """
        pass
    
    def _track_query_time(self, start_time: float) -> float:
        """
        Helper method to track query time statistics.
        
        Args:
            start_time: Time when query started
            
        Returns:
            Duration of query in seconds
        """
        query_time = time.time() - start_time
        self.stats['avg_query_time'] = (
            (self.stats['avg_query_time'] * (self.stats['queries'] - 1) + query_time) / 
            self.stats['queries'] if self.stats['queries'] > 0 else query_time
        )
        return query_time