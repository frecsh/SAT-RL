import random
import numpy as np
import gym
from gym import spaces

# --- Simplified SAT Environment ---
class SATEnv(gym.Env):
    def __init__(self, problem=None, num_variables=3):
        super(SATEnv, self).__init__()
        
        if problem:
            self.num_variables = problem["num_vars"]
            self.clauses = problem["clauses"]
            self.problem_name = problem["name"]
        else:
            # Use default problem
            self.num_variables = num_variables
            self.clauses = [[1, -2, 3], [-1, 2], [2, -3]]  # Default problem
            self.problem_name = "default"
            
        self.action_space = spaces.MultiBinary(self.num_variables)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_variables,), dtype=np.int32)

    def reset(self):
        self.state = np.zeros(self.num_variables, dtype=np.int32)
        return self.state

    def step(self, action):
        assignment = {i + 1: 1 if bit else 0 for i, bit in enumerate(action)}
        reward = self._evaluate_sat(assignment)
        done = True  # One-shot task
        return action, reward, done, {}
    
    def _evaluate_sat(self, assignment):
        total_clauses = len(self.clauses)
        satisfied_clauses = 0
        
        for clause in self.clauses:
            if any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause):
                satisfied_clauses += 1
        
        # Return normalized reward (percentage of satisfied clauses)
        return satisfied_clauses / total_clauses

# --- Random Policy Agent ---
def run_random_agent(env, episodes=10):
    print("Running random agent...")
    for episode in range(episodes):
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(f"Episode {episode + 1}: Action={action}, Reward={reward}")

if __name__ == "__main__":
    env = SATEnv(num_variables=3)
    run_random_agent(env)
