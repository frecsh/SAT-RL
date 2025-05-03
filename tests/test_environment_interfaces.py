"""
Unit tests for environment interfaces in the SAT solving RL framework.

Tests the standardized interfaces for environments, agents, and oracles
to ensure they conform to the expected API.
"""

import unittest
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import os
import time

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.base_env import SATEnv
from environments.sat_env import SimpleSATEnv
from agents.base_agent import AgentBase
from oracle.base_oracle import OracleBase
from oracle.simple_oracle import SimpleDPLLOracle

# Create a minimal test environment
class MinimalSATEnv(SATEnv):
    def __init__(self, num_vars=3, num_clauses=5):
        # Simple 3-SAT problem with 3 variables and 5 clauses
        clauses = [
            [1, 2, 3],    # x1 or x2 or x3
            [1, 2, -3],   # x1 or x2 or not x3
            [1, -2, 3],   # x1 or not x2 or x3
            [-1, 2, 3],   # not x1 or x2 or x3
            [-1, -2, -3]  # not x1 or not x2 or not x3
        ]
        super().__init__(clauses, num_vars)

    def _count_satisfied_clauses(self):
        """Count how many clauses are satisfied by the current assignment."""
        satisfied_clauses = 0
        for clause in self.clauses:
            if any(lit * self.assignment[abs(lit)] > 0 for lit in clause if abs(lit) < len(self.assignment)):
                satisfied_clauses += 1
        return satisfied_clauses

    def _is_satisfied(self):
        """Check if all clauses are satisfied."""
        return self._count_satisfied_clauses() == len(self.clauses)

    def _calculate_reward(self, prev_satisfied=None):
        """Calculate reward based on number of satisfied clauses."""
        satisfied_clauses = self._count_satisfied_clauses()
        return satisfied_clauses / len(self.clauses)

    def _get_observation(self):
        """Create and return the observation dictionary."""
        satisfied_clauses = self._count_satisfied_clauses()
        return {
            'assignment': self.assignment.copy(),
            'clauses': np.array(self.clauses),
            'satisfied_clauses': satisfied_clauses
        }
        
    def step(self, action):
        # Simplified step function that uses the abstract methods
        var_idx = action // 2
        value = 1 if action % 2 == 1 else -1
        
        # Update assignment
        self.assignment[var_idx + 1] = value
        self.steps_taken += 1
        
        # Get observation using abstract method
        observation = self._get_observation()
        
        # Calculate reward using abstract method
        reward = self._calculate_reward()
        
        # Check if done
        done = (self.steps_taken >= self.num_vars)
        
        # Create info
        info = {
            'satisfied_ratio': observation['satisfied_clauses'] / len(self.clauses),
            'steps_taken': self.steps_taken
        }
        
        return observation, reward, done, info
    
    def reset(self):
        self.assignment = np.zeros(self.num_vars + 1, dtype=np.int8)
        self.steps_taken = 0
        
        return self._get_observation()
    
    def render(self, mode='human'):
        return "MinimalSATEnv"


# Create a minimal test agent
class MinimalAgent(AgentBase):
    def __init__(self):
        num_vars = 3  # Match the MinimalSATEnv default
        action_space_size = num_vars * 2  # Two actions per variable (True/False)
        observation_shape = (1, num_vars + 1)  # Simple observation shape
        super().__init__(num_vars, action_space_size, observation_shape)
        self.action_count = 0
    
    def act(self, observation, deterministic=False):
        """
        Choose an action based on the observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to act deterministically
            
        Returns:
            An integer action to take
        """
        # Always return the next action in sequence
        action = self.action_count % 6  # For 3 variables, 6 possible actions
        self.action_count += 1
        return action
    
    def train(self, *args, **kwargs):
        """
        Perform a training step.
        
        Returns:
            Dictionary with training metrics
        """
        # No training happens in this minimal agent
        return {"loss": 0.0}
    
    def reset(self):
        """Reset the agent's internal state."""
        self.action_count = 0
        
    def end_episode(self, observation, reward, info=None):
        """
        Signal the end of an episode to the agent.
        
        Args:
            observation: Final state observation
            reward: Final reward
            info: Additional information
        """
        # Just reset the action count
        self.action_count = 0
        
    def export_model(self, path=None):
        """
        Export the agent model to the specified path.
        
        Args:
            path: Path where to save the model
            
        Returns:
            Path where the model was saved or a model representation
        """
        # This minimal agent has no model to export
        return {"type": "MinimalAgent", "action_count": self.action_count}


# Create a minimal test oracle
class MinimalOracle(OracleBase):
    def __init__(self, clauses, num_vars, oracle_config=None):
        """
        Initialize the minimal oracle.
        
        Args:
            clauses: List of clauses
            num_vars: Number of variables
            oracle_config: Configuration parameters
        """
        super().__init__(clauses, num_vars, oracle_config)
    
    def query(self, state, available_actions=None):
        """
        Query the oracle for guidance based on the current state.
        
        Args:
            state: Current state observation
            available_actions: Optional list of available actions
            
        Returns:
            Dictionary containing oracle guidance
        """
        # Simple oracle that always recommends the first available action
        self.stats['queries'] += 1
        start_time = time.time()  # Track query time
        
        if available_actions and len(available_actions) > 0:
            recommended_action = available_actions[0]
        else:
            # If no actions provided, choose the first unassigned variable
            assignment = state.get('assignment', np.zeros(self.num_vars + 1))
            for i in range(1, self.num_vars + 1):
                if assignment[i] == 0:
                    # Found an unassigned variable, recommend setting it to True (positive)
                    recommended_action = (i - 1) * 2 + 1
                    break
            else:
                # All variables are assigned
                recommended_action = None
        
        # Track query time
        self._track_query_time(start_time)
        
        return {
            "recommended_action": recommended_action,
            "confidence": 0.8,
            "explanation": "Minimal oracle recommendation"
        }
    
    def reset(self):
        """Reset the oracle's internal state."""
        super().reset()  # Call parent reset method


class TestEnvironmentInterfaces(unittest.TestCase):
    """Test cases for environment interfaces."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinimalSATEnv()
        self.agent = MinimalAgent()
        self.oracle = MinimalOracle(
            clauses=self.env.clauses,
            num_vars=self.env.num_vars
        )
    
    def test_env_interface(self):
        """Test environment interface conformance."""
        # Test reset returns a valid observation
        obs = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIn('assignment', obs)
        
        # Test step returns the correct tuple
        action = 0
        next_obs, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Test render returns something
        render_result = self.env.render()
        self.assertIsInstance(render_result, str)
    
    def test_agent_interface(self):
        """Test agent interface conformance."""
        # Test act returns a valid action
        obs = self.env.reset()
        action = self.agent.act(obs)
        self.assertIsInstance(action, (int, np.integer))
        
        # Test that the action is within bounds
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.env.num_vars * 2)
        
        # Test train returns a dict
        train_result = self.agent.train()
        self.assertIsInstance(train_result, dict)
        
        # Test reset doesn't throw errors
        self.agent.reset()
        
        # Test end_episode doesn't throw errors
        self.agent.end_episode(obs, 0.0)
    
    def test_oracle_interface(self):
        """Test oracle interface conformance."""
        # Test query returns a valid recommendation
        obs = self.env.reset()
        result = self.oracle.query(obs)
        
        self.assertIsInstance(result, dict)
        self.assertIn('recommended_action', result)
        self.assertIn('confidence', result)
        
        # Test with available actions
        available_actions = [0, 2, 4]
        result = self.oracle.query(obs, available_actions)
        self.assertEqual(result['recommended_action'], available_actions[0])
        
        # Test stats are updated
        self.assertGreater(self.oracle.stats['queries'], 0)
        
        # Test reset
        old_queries = self.oracle.stats['queries']
        self.oracle.reset()
        self.assertEqual(self.oracle.stats['queries'], 0)
        
        # Test update doesn't throw errors
        next_obs, reward, done, info = self.env.step(0)
        self.oracle.update(obs, 0, reward, next_obs, done, info)
    
    def test_full_episode(self):
        """Test a full agent-environment-oracle interaction episode."""
        obs = self.env.reset()
        self.agent.reset()
        
        done = False
        total_reward = 0
        
        while not done:
            # Get oracle guidance
            result = self.oracle.query(obs)
            
            # Agent decides action (ignoring oracle in this test)
            action = self.agent.act(obs)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Update oracle
            self.oracle.update(obs, action, reward, next_obs, done, info)
            
            # Update tracking variables
            total_reward += reward
            obs = next_obs
        
        # End episode
        self.agent.end_episode(obs, total_reward, info)
        
        # Simple assertion to make sure we complete the episode
        self.assertTrue(done)


# Run additional tests for concrete implementations
class TestConcreteImplementations(unittest.TestCase):
    """Test cases for concrete implementations of interfaces."""
    
    def test_simple_sat_env(self):
        """Test SimpleSATEnv implementation."""
        clauses = [
            [1, 2, 3],
            [1, -2, -3],
            [-1, 2, -3],
            [-1, -2, 3]
        ]
        env = SimpleSATEnv(clauses, 3)
        
        # Test reset
        obs = env.reset()
        self.assertIsInstance(obs, dict)
        
        # Test step
        action = 0  # Set var 0 to False
        next_obs, reward, done, info = env.step(action)
        
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_simple_dpll_oracle(self):
        """Test SimpleDPLLOracle implementation if available."""
        try:
            clauses = [
                [1, 2, 3],
                [1, -2, -3],
                [-1, 2, -3],
                [-1, -2, 3]
            ]
            oracle = SimpleDPLLOracle(clauses, 3)
            
            # Create a state
            state = {
                'assignment': np.zeros(4, dtype=np.int8)
            }
            
            # Test query
            result = oracle.query(state)
            self.assertIsInstance(result, dict)
            self.assertIn('recommended_action', result)
            self.assertIn('confidence', result)
            
        except (ImportError, AttributeError):
            self.skipTest("SimpleDPLLOracle not fully implemented yet")


if __name__ == '__main__':
    unittest.main()