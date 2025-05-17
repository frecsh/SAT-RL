"""
Advanced Usage Examples: Feedback-Driven Reward Integration in SatEnv
"""
# Feedback vector index mapping for SatEnv (order matches SATFeedback.get_feedback):
# 0: clause_satisfaction
# 1: variable_decisiveness
# 2: search_diversity
# 3: constraint_tension
# 4: proof_progress
# 5: clause_centrality
# 6: assignment_entropy
# 7: clause_length_var

import numpy as np

from symbolicgym.envs.sat_env import SatEnv
from symbolicgym.utils.feedback_metrics import (
    FEEDBACK_VECTOR_INDEX_TO_NAME,
    FEEDBACK_VECTOR_METRICS,
    FEEDBACK_VECTOR_NAME_TO_INDEX,
)

# 1. Single-Agent: Custom Feedback-Weighted Reward
print("\n--- Single-Agent Custom Feedback-Weighted Reward ---")
env = SatEnv(
    feedback_in_reward=True,
    feedback_reward_scale=0.05,
    feedback_metrics_to_use=[
        "clause_satisfaction",
        "proof_progress",
        "assignment_entropy",
    ],
    feedback_metric_weights=[0.5, 0.3, 0.2],
)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, *_ = env.step(action)
print("Reward with feedback:", reward)
print("Feedback vector:", obs["feedback_vector"])

# 2. Multi-Agent: Feedback-Weighted Reward
print("\n--- Multi-Agent Feedback-Weighted Reward ---")
env = SatEnv(
    multi_agent_mode=True,
    n_agents=2,
    feedback_in_reward=True,
    feedback_reward_scale=0.1,
    feedback_metrics_to_use=["clause_satisfaction", "clause_centrality"],
    feedback_metric_weights=[0.7, 0.3],
)
obs, info = env.reset()
actions = {aid: env.action_space[aid].sample() for aid in obs}
obs, rewards, *_ = env.step(actions)
for aid in obs:
    print(
        f"Agent {aid} reward: {rewards[aid]}, feedback: {obs[aid]['feedback_vector']}"
    )

# 3. Advanced: Multi-Agent Loop with Full Feedback Vector
print("\n--- Multi-Agent Loop with Full Feedback Vector ---")
env = SatEnv(
    multi_agent_mode=True,
    n_agents=2,
    feedback_in_reward=True,
    feedback_reward_scale=0.1,
    feedback_metrics_to_use=[
        "clause_satisfaction",
        "proof_progress",
        "assignment_entropy",
        "clause_centrality",
    ],
    feedback_metric_weights=[0.4, 0.3, 0.2, 0.1],
)
obs, info = env.reset()
done = {aid: False for aid in obs}
step = 0
while not all(done.values()):
    for aid, agent_obs in obs.items():
        print(f"Step {step} | Agent {aid} | Feedback: {agent_obs['feedback_vector']}")
    actions = {aid: env.action_space[aid].sample() for aid in obs}
    obs, rewards, terminated, truncated, info, _ = env.step(actions)
    print(f"Step {step} | Rewards: {rewards}")
    for aid in done:
        done[aid] = terminated[aid] or truncated[aid]
    step += 1
print("Episode finished.")

# Example: interpret feedback vector programmatically
print("\n--- Feedback Vector Metric Names ---")
for i, val in enumerate(obs["feedback_vector"]):
    print(f"{i}: {FEEDBACK_VECTOR_INDEX_TO_NAME[i]} = {val}")
