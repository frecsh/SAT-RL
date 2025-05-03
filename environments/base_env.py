"""
Base environment interface for SAT problems.
Defines the standard environment API for reinforcement learning on SAT instances.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import gym
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SATEnv(gym.Env, ABC):
    """
    Abstract base class for SAT environments.
    Follows the OpenAI Gym interface for RL environments.
    """
    
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    
    def __init__(
        self, 
        clauses: Optional[List[List[int]]] = None,
        num_vars: int = 0,
        max_steps: int = 1000,
        reward_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the SAT environment.
        
        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the problem
            max_steps: Maximum number of steps per episode
            reward_config: Configuration for reward shaping
        """
        super().__init__()
        
        self.clauses = clauses or []
        self.num_vars = num_vars or (max([abs(lit) for clause in self.clauses for lit in clause]) if self.clauses else 0)
        self.max_steps = max_steps
        self.reward_config = reward_config or {
            'sat_reward': 1.0,            # Reward for satisfying the formula
            'unsat_penalty': -1.0,        # Penalty for unsatisfiable formula
            'step_penalty': -0.01,        # Small penalty for each step to encourage efficiency
            'clause_reward': 0.1,         # Reward per satisfied clause
            'repeated_action_penalty': -0.05  # Penalty for repeating the same action
        }
        
        # Current state
        self.assignment = np.zeros(self.num_vars + 1, dtype=np.int8)  # 1-indexed variables
        self.steps_taken = 0
        self.done = False
        self.info = {}
        self.last_action = None
        
        # Define action and observation spaces
        # Action: Choose a variable (1 to num_vars) and a value (0 or 1)
        # We represent this as a single integer: var_id + value*num_vars
        self.action_space = gym.spaces.Discrete(self.num_vars * 2)
        
        # Observation: Current assignment state + clause status
        # This is a simplified default - subclasses will likely override this
        self.observation_space = gym.spaces.Dict({
            'assignment': gym.spaces.Box(low=-1, high=1, shape=(self.num_vars + 1,), dtype=np.int8),
            'clause_status': gym.spaces.Box(low=0, high=1, shape=(len(self.clauses),), dtype=np.int8),
            'steps': gym.spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.int32)
        })
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation.
        
        Returns:
            Observation dictionary
        """
        pass
    
    @abstractmethod
    def _calculate_reward(self, prev_satisfied: int, current_satisfied: int) -> float:
        """
        Calculate the reward based on the change in satisfied clauses.
        
        Args:
            prev_satisfied: Number of satisfied clauses before action
            current_satisfied: Number of satisfied clauses after action
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def _count_satisfied_clauses(self, assignment: Optional[np.ndarray] = None) -> int:
        """
        Count the number of satisfied clauses for the current assignment.
        
        Args:
            assignment: Optional specific assignment to evaluate instead of current state
            
        Returns:
            Number of satisfied clauses
        """
        pass
    
    @abstractmethod
    def _is_satisfied(self) -> bool:
        """
        Check if all clauses are satisfied with the current assignment.
        
        Returns:
            True if all clauses are satisfied, False otherwise
        """
        pass
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        self.assignment = np.zeros(self.num_vars + 1, dtype=np.int8)
        self.steps_taken = 0
        self.done = False
        self.info = {}
        self.last_action = None
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by setting a variable.
        
        Args:
            action: Integer representing the action to take
                   (var_id + value*num_vars) where var_id is 1-indexed
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            logger.warning("Episode already done, call reset() before taking more steps")
            return self._get_observation(), 0.0, True, self.info
        
        # Increment step counter
        self.steps_taken += 1
        
        # Decode the action
        # action = var_id + value*num_vars
        var_id = (action % self.num_vars) + 1  # 1-indexed
        value = action // self.num_vars         # 0 or 1
        literal = var_id if value == 1 else -var_id
        
        # Get number of satisfied clauses before action
        prev_satisfied = self._count_satisfied_clauses()
        
        # Apply the action (set the variable)
        self.assignment[var_id] = 1 if value == 1 else -1
        
        # Get number of satisfied clauses after action
        current_satisfied = self._count_satisfied_clauses()
        
        # Calculate reward
        reward = self._calculate_reward(prev_satisfied, current_satisfied)
        
        # Check if repeated action
        if self.last_action == action:
            reward += self.reward_config.get('repeated_action_penalty', 0.0)
        
        self.last_action = action
        
        # Check if done
        if self._is_satisfied():
            self.done = True
            reward += self.reward_config.get('sat_reward', 1.0)
        elif self.steps_taken >= self.max_steps:
            self.done = True
        
        # Update info
        self.info = {
            'satisfied_clauses': current_satisfied,
            'total_clauses': len(self.clauses),
            'satisfaction_ratio': current_satisfied / len(self.clauses) if self.clauses else 0.0,
            'steps_taken': self.steps_taken,
            'is_satisfied': self._is_satisfied(),
            'last_action': {'var_id': var_id, 'value': value, 'literal': literal}
        }
        
        return self._get_observation(), reward, self.done, self.info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'ansi', 'rgb_array')
            
        Returns:
            Optional rendering output depending on the mode
        """
        if mode == 'human' or mode == 'ansi':
            assignment_str = ' '.join([f"{i}:{'+' if v > 0 else '-' if v < 0 else '?'}" 
                                     for i, v in enumerate(self.assignment) if i > 0])
            satisfied = self._count_satisfied_clauses()
            total = len(self.clauses)
            output = (f"Step: {self.steps_taken}/{self.max_steps} | "
                     f"Satisfied: {satisfied}/{total} ({satisfied/total:.2%})\n"
                     f"Assignment: {assignment_str}")
            
            if mode == 'human':
                print(output)
                return None
            else:
                return output
        elif mode == 'rgb_array':
            # Default implementation returns a simple visualization
            # Subclasses should override this for more sophisticated visualizations
            img = np.zeros((100, self.num_vars * 10 + 20, 3), dtype=np.uint8)
            
            # Draw variables
            for i in range(1, self.num_vars + 1):
                color = [0, 255, 0] if self.assignment[i] > 0 else [255, 0, 0] if self.assignment[i] < 0 else [100, 100, 100]
                img[40:60, i*10:i*10+8] = color
            
            return img
            
        else:
            raise ValueError(f"Invalid render mode: {mode}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
            
        Returns:
            List containing the seed
        """
        np.random.seed(seed)
        return [seed]
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        pass
    
    def add_clause(self, clause: List[int]) -> None:
        """
        Add a clause to the problem.
        
        Args:
            clause: List of literals representing a clause
        """
        self.clauses.append(clause)
        # Update observation space if needed
        if hasattr(self.observation_space.spaces, 'clause_status'):
            self.observation_space.spaces['clause_status'] = gym.spaces.Box(
                low=0, high=1, shape=(len(self.clauses),), dtype=np.int8
            )
        
    def add_clauses(self, clauses: List[List[int]]) -> None:
        """
        Add multiple clauses to the problem.
        
        Args:
            clauses: List of clauses
        """
        self.clauses.extend(clauses)
        # Update observation space if needed
        if hasattr(self.observation_space.spaces, 'clause_status'):
            self.observation_space.spaces['clause_status'] = gym.spaces.Box(
                low=0, high=1, shape=(len(self.clauses),), dtype=np.int8
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state.
        
        Returns:
            Dictionary of statistics
        """
        satisfied = self._count_satisfied_clauses()
        return {
            'satisfied_clauses': satisfied,
            'total_clauses': len(self.clauses),
            'satisfaction_ratio': satisfied / len(self.clauses) if self.clauses else 0.0,
            'steps_taken': self.steps_taken,
            'max_steps': self.max_steps,
            'is_satisfied': self._is_satisfied(),
            'num_vars': self.num_vars
        }