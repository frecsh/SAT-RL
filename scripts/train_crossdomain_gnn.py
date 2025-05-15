"""
Train a GNN agent across SAT, SymPy, and Z3 domains using SymbolicGym.
"""
import gymnasium as gym

import symbolicgym
from symbolicgym.agents.gnn_agent import GNNCrossDomainAgent
from symbolicgym.utils.reproducibility import set_seed

set_seed(123)
domains = ["SAT", "SymPy", "Z3"]
envs = [gym.make(f"SymbolicGym-{d}-v0") for d in domains]
agent = GNNCrossDomainAgent(envs)
agent.train(num_episodes=300)
