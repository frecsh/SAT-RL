import sys

sys.path.append("../../src")
import matplotlib.pyplot as plt
import torch

from symbolicgym.agents.comm_agent import CommAgent
from symbolicgym.agents.marl.communication import CommunicationChannel
from symbolicgym.envs.sat_env import SatEnv

if __name__ == "__main__":
    n_agents = 2
    env = SatEnv(n_agents=n_agents, multi_agent_mode=True)
    agents = [
        CommAgent(state_dim=env.num_vars, action_dim=env.num_vars)
        for _ in range(n_agents)
    ]
    channel = CommunicationChannel()
    obs, info = env.reset()
    done = {aid: False for aid in env.agent_ids}
    step = 0
    max_steps = 20
    rewards_history = {aid: [] for aid in env.agent_ids}
    feedback_history = {
        aid: {
            k: []
            for k in [
                "clause_satisfaction",
                "variable_decisiveness",
                "search_diversity",
                "constraint_tension",
                "proof_progress",
            ]
        }
        for aid in env.agent_ids
    }
    while not all(done.values()) and step < max_steps:
        messages = {}
        peer_messages = {}
        for i, aid in enumerate(env.agent_ids):
            msg_vec = agents[i].get_message(obs[aid]["variables"])
            messages[aid] = msg_vec
        for i, aid in enumerate(env.agent_ids):
            peer_msgs = [m for j, (k, m) in enumerate(messages.items()) if k != aid]
            peer_messages[aid] = peer_msgs
        actions = {
            aid: agents[i].act(obs[aid]["variables"], peer_messages[aid])
            for i, aid in enumerate(env.agent_ids)
        }
        next_obs, reward, terminated, truncated, info, feedback = env.step(
            actions, messages=messages
        )
        for i, aid in enumerate(env.agent_ids):
            agents[i].observe(
                obs[aid]["variables"],
                actions[aid],
                reward[aid],
                next_obs[aid]["variables"],
                terminated[aid],
                peer_messages[aid],
                feedback[aid],
            )
            rewards_history[aid].append(reward[aid])
            for k in feedback_history[aid]:
                feedback_history[aid][k].append(feedback[aid][k])
        obs = next_obs
        done = terminated
        step += 1
    # Visualization
    for aid in env.agent_ids:
        plt.plot(rewards_history[aid], label=f"Reward {aid}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Agent Rewards Over Time")
    plt.legend()
    plt.show()
    for k in [
        "clause_satisfaction",
        "variable_decisiveness",
        "search_diversity",
        "constraint_tension",
        "proof_progress",
    ]:
        for aid in env.agent_ids:
            plt.plot(feedback_history[aid][k], label=f"{k} {aid}")
        plt.xlabel("Step")
        plt.ylabel(k)
        plt.title(f"{k} Over Time")
        plt.legend()
        plt.show()
