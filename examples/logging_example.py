"""
Example script demonstrating the use of structured logging and error handling.

This script shows how to use the StructuredLogger and custom exception classes
to properly track and report events during SAT solving.
"""

import os
import sys
import time
import traceback
import numpy as np
import random
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import StructuredLogger, create_logger
from utils.exceptions import (
    SATBaseException, UnsatisfiableError, SolverTimeoutError, 
    InconsistentAssignmentError, InvalidClauseError
)


def simulate_simple_sat_solving(logger, num_episodes=5, num_steps=10):
    """
    Simulate a simple SAT solving process with logging.
    
    Args:
        logger: StructuredLogger instance
        num_episodes: Number of episodes to simulate
        num_steps: Maximum steps per episode
    """
    for episode in range(num_episodes):
        print(f"\nStarting episode {episode+1}/{num_episodes}")
        
        # Create a random SAT problem
        num_vars = random.randint(10, 20)
        num_clauses = random.randint(num_vars, num_vars * 4)
        
        # Track episode start time
        start_time = time.time()
        
        # Simulate solving process
        try:
            # Randomly decide if this episode will have an error
            if random.random() < 0.3:  # 30% chance of error
                error_type = random.choice([
                    "timeout", "unsatisfiable", "inconsistent", "invalid_clause"
                ])
                
                # Simulate steps before error
                error_step = random.randint(1, num_steps-1)
                
                for step in range(error_step):
                    simulate_step(logger, episode, step, num_vars, num_clauses)
                
                # Simulate different error types
                if error_type == "timeout":
                    raise SolverTimeoutError(
                        time_spent=random.uniform(1.0, 10.0),
                        satisfied_clauses=random.randint(int(num_clauses * 0.5), num_clauses - 1)
                    )
                elif error_type == "unsatisfiable":
                    raise UnsatisfiableError(
                        clause_analysis={"core_size": random.randint(1, 5)}
                    )
                elif error_type == "inconsistent":
                    raise InconsistentAssignmentError(
                        variable=random.randint(1, num_vars)
                    )
                else:  # invalid_clause
                    raise InvalidClauseError(
                        clause=[0] if random.random() < 0.5 else [num_vars + 10]
                    )
            
            # Normal solving process (no errors)
            for step in range(num_steps):
                simulate_step(logger, episode, step, num_vars, num_clauses)
                
                # Check if problem is solved
                if random.random() < 0.1:  # 10% chance to solve at each step
                    success = True
                    break
            else:
                success = False
        
        except UnsatisfiableError as e:
            print(f"Problem is unsatisfiable: {e}")
            logger.log_exception(
                episode, step, "UnsatisfiableError", str(e), 
                traceback.format_exc()
            )
            success = False
            
        except SolverTimeoutError as e:
            print(f"Solver timed out: {e}")
            logger.log_exception(
                episode, step, "SolverTimeoutError", str(e),
                traceback.format_exc()
            )
            success = False
            
        except SATBaseException as e:
            print(f"SAT solving error: {e}")
            logger.log_exception(
                episode, step, e.__class__.__name__, str(e),
                traceback.format_exc()
            )
            success = False
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            logger.log_exception(
                episode, step, e.__class__.__name__, str(e),
                traceback.format_exc()
            )
            success = False
            
        # Log performance
        solve_time = time.time() - start_time
        logger.log_performance(
            episode, 
            solve_time, 
            success,
            {
                "num_vars": num_vars,
                "num_clauses": num_clauses,
                "steps_taken": step + 1
            }
        )
        
        print(f"Episode {episode+1} {'succeeded' if success else 'failed'} "
              f"in {solve_time:.2f}s")
            

def simulate_step(logger, episode, step, num_vars, num_clauses):
    """
    Simulate a single step in the solving process.
    
    Args:
        logger: StructuredLogger instance
        episode: Current episode number
        step: Current step number
        num_vars: Number of variables in the problem
        num_clauses: Number of clauses in the problem
    """
    # Simulate agent's observation (simplified)
    observation = {
        "assignment": np.zeros(num_vars + 1, dtype=np.int8),
        "clauses": np.random.randint(-num_vars, num_vars + 1, size=(num_clauses, 3))
    }
    
    # Fill in some random assignments
    for i in range(1, min(step + 2, num_vars + 1)):
        observation["assignment"][i] = random.choice([-1, 1])
    
    # Simulate agent decision
    action = random.randint(0, num_vars * 2 - 1)
    agent_id = "agent_1"
    
    # Simulate oracle guidance
    oracle_action = random.randint(0, num_vars * 2 - 1)
    confidence = random.random()
    
    # Simulate reward
    satisfied = random.randint(0, num_clauses)
    reward = satisfied / num_clauses
    
    # Log everything
    logger.log_agent_decision(
        episode, step, agent_id, action, observation,
        {action: 0.8, (action + 1) % (num_vars * 2): 0.2}
    )
    
    logger.log_oracle_guidance(
        episode, step, oracle_action, confidence,
        "Heuristic recommendation based on VSIDS"
    )
    
    logger.log_reward(
        episode, step, reward,
        {"action_taken": action, "oracle_action": oracle_action}
    )
    
    logger.log_clause_stats(
        episode, step, satisfied, num_clauses
    )
    
    # Sleep a bit to simulate computation time
    time.sleep(0.01)
    
    print(f"  Step {step}: action={action}, reward={reward:.2f}, "
          f"clauses={satisfied}/{num_clauses}")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description='Logging example for SAT solving')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                        help='Log format (default: json)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to simulate')
    parser.add_argument('--steps', type=int, default=10,
                        help='Maximum steps per episode')
    parser.add_argument('--output', default='logs',
                        help='Output directory for logs')
    args = parser.parse_args()
    
    # Create logger
    logger = create_logger(
        experiment_name="sat_logging_example",
        format_type=args.format,
        output_dir=args.output
    )
    
    try:
        print(f"Starting SAT solving simulation with {args.episodes} episodes")
        print(f"Logs will be written to {os.path.abspath(args.output)}")
        simulate_simple_sat_solving(logger, args.episodes, args.steps)
    finally:
        # Make sure to close the logger
        logger.close()
    
    print(f"\nSimulation complete. Log files can be found in {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()