"""
Script for oracle imitation pre-training.
"""
import numpy as np

from symbolicgym.agents.imitation_agent import ImitationAgent


def oracle_policy(state):
    # Placeholder: load or define oracle policy
    return np.argmax(state)


if __name__ == "__main__":
    agent = ImitationAgent(policy=oracle_policy)
    # Load demonstration data (stub)
    # for state, action, reward, next_state, done in demo_data:
    #     agent.observe(state, action, reward, next_state, done)
    agent.update()
    print("Oracle imitation pre-training complete.")
