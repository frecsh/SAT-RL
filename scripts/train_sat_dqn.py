"""Train a DQN agent on the SAT domain using SymbolicGym."""

import gymnasium as gym

from symbolicgym.agents.dqn_agent import DQNAgent
from symbolicgym.utils.reproducibility import set_seed

set_seed(42)
env = gym.make("SymbolicGym-v0", cnf_file="benchmarks/uf50-01.cnf", reward_mode="dense")
agent = DQNAgent(env)
agent.train(num_episodes=500)
