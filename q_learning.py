import numpy as np
import random
from collections import defaultdict

# SAT environment (same as before)
class SATEnvironment:
    def __init__(self, formula):
        self.formula = formula
        self.n_vars = len({abs(lit) for clause in formula for lit in clause})
        self.reset()

    def reset(self):
        self.state = [None] * self.n_vars
        self.index = 0
        return tuple(self.state)

    def step(self, action):
        self.state[self.index] = bool(action)
        self.index += 1
        done = self.index == self.n_vars
        reward = 0
        if done:
            reward = self.evaluate_formula()
        return tuple(self.state), reward, done

    def evaluate_formula(self):
        for clause in self.formula:
            if not any((lit > 0 and self.state[abs(lit)-1]) or (lit < 0 and not self.state[abs(lit)-1]) for lit in clause):
                return -1  # failed to satisfy
        return 1  # satisfied!

# Simple Q-learning agent
class QLearningAgent:
    def __init__(self, n_vars, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 min_exploration_rate=0.01, exploration_decay=0.995):
        self.q_table = defaultdict(lambda: [0.0, 0.0])  # Q(state)[0] = False, Q(state)[1] = True
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay
        self.n_vars = n_vars

        # Add metrics tracking
        self.updates = 0
        self.q_value_history = []
        self.action_history = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return int(self.q_table[state][1] > self.q_table[state][0])

    def update(self, state, action, reward, next_state):
        next_max = max(self.q_table[next_state]) if next_state else 0.0
        current = self.q_table[state][action]
        new_value = current + self.alpha * (reward + self.gamma * next_max - current)
        self.q_table[state][action] = new_value

        # Track metrics
        self.updates += 1
        if self.updates % 100 == 0:
            # Sample average Q-value for reporting
            avg_q = sum(sum(values) for values in self.q_table.values()) / max(1, len(self.q_table))
            self.q_value_history.append(avg_q)

# Run the Q-learning simulation
def train_q_learning_agent(formula, episodes=1000):
    env = SATEnvironment(formula)
    agent = QLearningAgent(n_vars=env.n_vars)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state if not done else None)
            state = next_state

        if ep % 100 == 0:
            print(f"Episode {ep}, reward: {reward}")

    return agent

# Example formula: (x1 ∨ ¬x2) ∧ (¬x1 ∨ x2 ∨ x3)
example_formula = [[1, -2], [-1, 2, 3]]
trained_agent = train_q_learning_agent(example_formula)
