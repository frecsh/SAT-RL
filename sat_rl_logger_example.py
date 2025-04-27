"""
Example showing how to use the SAT RL Logger with an existing agent
"""

import numpy as np
import time
from sat_rl_logger import SATRLLogger, wrap_agent_step


# Mock environment and agent for demonstration purposes
class MockSATEnvironment:
    """Simple mock environment for demonstration"""
    
    def __init__(self, num_vars=10, num_clauses=40):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses_remaining = num_clauses
        self.conflict_detected = False
        self.reset()
    
    def reset(self):
        """Reset the environment state"""
        self.state = np.ones(self.num_vars) * -1  # -1 represents unassigned
        self.clauses_remaining = self.num_clauses
        self.conflict_detected = False
        return self.state.copy()
    
    def step(self, action):
        """Take a step in the environment"""
        var, val = action
        
        # Update state
        self.state[var] = val
        
        # Simulate environment behavior - this would be your actual SAT logic
        self.clauses_remaining = max(0, self.clauses_remaining - np.random.randint(0, 3))
        
        # Randomly generate conflicts (for demonstration)
        if np.random.random() < 0.1:
            self.conflict_detected = True
            reward = -0.5
            done = True
        elif self.clauses_remaining == 0:
            # All clauses satisfied
            reward = 1.0
            done = True
        else:
            # Still solving
            reward = 0.1
            done = False
        
        return self.state.copy(), reward, done, {}


class MockSATAgent:
    """Simple mock agent for demonstration"""
    
    def __init__(self, num_vars):
        self.num_vars = num_vars
    
    def step(self, env, state):
        """Original step function - choose a variable and assign it"""
        # Find unassigned variables (where value is -1)
        unassigned = np.where(state == -1)[0]
        
        if len(unassigned) == 0:
            # All variables are assigned - demonstrate handling None action
            print("All variables assigned, returning None action")
            return None, state, 0.5, True
        
        # Select a random unassigned variable
        var = np.random.choice(unassigned)
        
        # Select a random value (0 or 1)
        val = np.random.randint(0, 2)
        
        # Take the action in the environment
        next_state, reward, done, _ = env.step((var, val))
        
        return (var, val), next_state, reward, done


def run_example():
    """Run the example with a mock agent and environment"""
    # Create environment and agent
    num_vars = 5  # Reduced to make it faster to assign all variables
    env = MockSATEnvironment(num_vars=num_vars, num_clauses=20)
    agent = MockSATAgent(num_vars=num_vars)
    
    # Create logger and wrap the agent's step function
    logger = SATRLLogger(max_entries=1000, log_to_file=True)
    _, logger = wrap_agent_step(agent, env, logger)
    
    # Run a few episodes
    for episode in range(3):
        state = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\nStarting Episode {episode + 1}")
        
        step_count = 0
        # Make sure we run until all variables are assigned at least once
        while not done and step_count < num_vars + 5:  
            step_count += 1
            
            # Use the wrapped step function
            action, state, reward, done = agent.step(env, state, episode=episode)
            
            # Print out information about the action
            if action is None:
                print(f"  Step {step_count}: No action taken")
            else:
                var, val = action
                print(f"  Step {step_count}: Assigned var {var} = {val}, reward = {reward:.2f}")
            
            episode_reward += reward
            
        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
    
    # Get statistics
    stats = logger.get_statistics()
    print("\nLogging Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export to CSV
    csv_path = logger.export_traces_to_csv()
    print(f"\nTraces exported to: {csv_path}")


if __name__ == "__main__":
    run_example()