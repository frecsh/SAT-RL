import sys

sys.path.append("../../src")

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
    while not all(done.values()) and step < 10:
        # Each agent sends a message
        messages = {}
        peer_messages = {}
        for i, aid in enumerate(env.agent_ids):
            msg_vec = agents[i].get_message(obs[aid]["variables"])
            messages[aid] = msg_vec
        # Each agent receives peer messages (excluding own)
        for i, aid in enumerate(env.agent_ids):
            peer_msgs = [m for j, (k, m) in enumerate(messages.items()) if k != aid]
            peer_messages[aid] = peer_msgs
        # Each agent acts
        actions = {
            aid: agents[i].act(obs[aid]["variables"], peer_messages[aid])
            for i, aid in enumerate(env.agent_ids)
        }
        obs, reward, terminated, truncated, info, feedback = env.step(
            actions, messages=messages
        )
        print(f"Step {step}: actions={actions}, reward={reward}, feedback={feedback}")
        done = terminated
        step += 1
    print("Final info:", info)
