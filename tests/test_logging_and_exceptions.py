"""
Unit tests for logging and error handling components.

Tests the StructuredLogger and exception classes to ensure they work as expected.
"""

import unittest
import os
import tempfile
import json
import shutil
import csv
from io import StringIO
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import StructuredLogger, create_logger, NumpyJSONEncoder
from utils.exceptions import (
    SATBaseException, UnsatisfiableError, SolverTimeoutError, 
    InconsistentAssignmentError, InvalidClauseError
)


class TestExceptions(unittest.TestCase):
    """Test cases for custom exception classes."""
    
    def test_unsatisfiable_error(self):
        """Test UnsatisfiableError class."""
        error = UnsatisfiableError()
        self.assertEqual(str(error), "Problem is unsatisfiable")
        
        # Test with custom message and clause analysis
        clause_analysis = {"core_size": 3, "core_clauses": [1, 2, 5]}
        error = UnsatisfiableError("Custom message", clause_analysis)
        self.assertEqual(str(error), "Custom message")
        self.assertEqual(error.clause_analysis, clause_analysis)
    
    def test_solver_timeout_error(self):
        """Test SolverTimeoutError class."""
        error = SolverTimeoutError()
        self.assertEqual(str(error), "Solver exceeded time limit")
        
        # Test with custom attributes
        error = SolverTimeoutError(
            "Custom timeout message",
            time_spent=10.5,
            partial_assignment=[1, -2, 3],
            satisfied_clauses=42
        )
        self.assertEqual(error.time_spent, 10.5)
        self.assertEqual(error.partial_assignment, [1, -2, 3])
        self.assertEqual(error.satisfied_clauses, 42)
        
        # Check string representation includes details
        error_str = str(error)
        self.assertIn("Custom timeout message", error_str)
        self.assertIn("time_spent=10.50s", error_str)
        self.assertIn("satisfied_clauses=42", error_str)
    
    def test_inconsistent_assignment_error(self):
        """Test InconsistentAssignmentError class."""
        error = InconsistentAssignmentError()
        self.assertEqual(str(error), "Inconsistent variable assignment detected")
        
        # Test with variable specified
        error = InconsistentAssignmentError(variable=5)
        self.assertIn("variable 5", str(error))
        self.assertEqual(error.variable, 5)
    
    def test_invalid_clause_error(self):
        """Test InvalidClauseError class."""
        error = InvalidClauseError()
        self.assertEqual(str(error), "Invalid clause detected")
        
        # Test with clause specified
        error = InvalidClauseError(clause=[0, 1, 2])
        self.assertIn("[0, 1, 2]", str(error))
        self.assertEqual(error.clause, [0, 1, 2])


class TestStructuredLogger(unittest.TestCase):
    """Test cases for StructuredLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger(output_dir=self.test_dir, experiment_name="test_exp")
        self.assertEqual(logger.experiment_name, "test_exp")
        self.assertEqual(logger.output_dir, self.test_dir)
        self.assertEqual(logger.format_type, "json")
        logger.close()
    
    def test_json_logging(self):
        """Test JSON format logging."""
        logger = StructuredLogger(
            output_dir=self.test_dir, 
            experiment_name="json_test",
            format_type=StructuredLogger.FORMAT_JSON
        )
        
        # Log some events
        logger.log_reward(1, 2, 0.5, {"action": 1})
        logger.log_clause_stats(1, 2, 5, 10)
        logger.close()
        
        # Check that the files were created
        reward_file = os.path.join(self.test_dir, "json_test_reward.jsonl")
        clause_file = os.path.join(self.test_dir, "json_test_clause_stats.jsonl")
        
        self.assertTrue(os.path.exists(reward_file))
        self.assertTrue(os.path.exists(clause_file))
        
        # Check the content of the reward file
        with open(reward_file, 'r') as f:
            reward_data = json.loads(f.readline())
        
        self.assertEqual(reward_data["episode"], 1)
        self.assertEqual(reward_data["step"], 2)
        self.assertEqual(reward_data["reward"], 0.5)
        self.assertEqual(reward_data["info"]["action"], 1)
        
        # Check the content of the clause stats file
        with open(clause_file, 'r') as f:
            clause_data = json.loads(f.readline())
        
        self.assertEqual(clause_data["episode"], 1)
        self.assertEqual(clause_data["step"], 2)
        self.assertEqual(clause_data["satisfied_count"], 5)
        self.assertEqual(clause_data["total_count"], 10)
        self.assertEqual(clause_data["satisfaction_ratio"], 0.5)
    
    def test_csv_logging(self):
        """Test CSV format logging."""
        logger = StructuredLogger(
            output_dir=self.test_dir, 
            experiment_name="csv_test",
            format_type=StructuredLogger.FORMAT_CSV
        )
        
        # Log some events
        logger.log_reward(1, 2, 0.5, {"action": 1})
        logger.log_reward(1, 3, 0.7, {"action": 2})
        logger.close()
        
        # Check that the file was created
        reward_file = os.path.join(self.test_dir, "csv_test_reward.csv")
        self.assertTrue(os.path.exists(reward_file))
        
        # Check the content of the reward file
        with open(reward_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], "1")
        self.assertEqual(rows[0]["step"], "2")
        self.assertEqual(rows[0]["reward"], "0.5")
        
        self.assertEqual(rows[1]["episode"], "1")
        self.assertEqual(rows[1]["step"], "3")
        self.assertEqual(rows[1]["reward"], "0.7")
    
    def test_log_agent_decision(self):
        """Test logging agent decisions."""
        logger = StructuredLogger(output_dir=self.test_dir, experiment_name="agent_test")
        
        # Log an agent decision
        observation = {"assignment": [0, 1, -1, 0], "clauses": [[1, 2, 3], [-1, -2, 3]]}
        logger.log_agent_decision(1, 2, "agent1", 5, observation, {5: 0.8, 6: 0.2})
        logger.close()
        
        # Check the file content
        decision_file = os.path.join(self.test_dir, "agent_test_agent_decision.jsonl")
        with open(decision_file, 'r') as f:
            data = json.loads(f.readline())
        
        self.assertEqual(data["episode"], 1)
        self.assertEqual(data["step"], 2)
        self.assertEqual(data["agent_id"], "agent1")
        self.assertEqual(data["action"], 5)
        self.assertEqual(data["action_probs"]["5"], 0.8)
        self.assertEqual(data["action_probs"]["6"], 0.2)
        self.assertEqual(data["observation"]["assignment"], [0, 1, -1, 0])
    
    def test_log_exception(self):
        """Test logging exceptions."""
        logger = StructuredLogger(output_dir=self.test_dir, experiment_name="exception_test")
        
        # Log an exception
        logger.log_exception(1, 2, "UnsatisfiableError", "Problem is unsatisfiable", "Stack trace")
        logger.close()
        
        # Check the file content
        exception_file = os.path.join(self.test_dir, "exception_test_exception.jsonl")
        with open(exception_file, 'r') as f:
            data = json.loads(f.readline())
        
        self.assertEqual(data["episode"], 1)
        self.assertEqual(data["step"], 2)
        self.assertEqual(data["exception_type"], "UnsatisfiableError")
        self.assertEqual(data["exception_message"], "Problem is unsatisfiable")
        self.assertEqual(data["stack_trace"], "Stack trace")


if __name__ == '__main__':
    unittest.main()