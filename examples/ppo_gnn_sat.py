"""
Ready-to-run PPO pipeline using SharedEncoder (GNN) for SAT environment.
Requires: torch, torch_geometric, gymnasium, symbolicgym
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from symbolicgym.envs.sat_env import SatEnv
from symbolicgym.models.shared_encoder import SharedEncoder
from symbolicgym.wrappers.observation import ObservationWrapper

# PPO hyperparameters
GAMMA = 0.99
LR = 3e-4
EPOCHS = 5
STEPS_PER_EPOCH = 128
EMBED_DIM = 32


class PPOPolicy(nn.Module):
    def __init__(self, node_dim, embed_dim, n_actions):
        super().__init__()
        self.encoder = SharedEncoder(
            input_dim=node_dim, embed_dim=embed_dim, encoder_type="deeper_gnn"
        )
        self.policy_head = nn.Linear(embed_dim, n_actions)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, graph_data):
        emb = self.encoder(graph_data)
        logits = self.policy_head(emb)
        value = self.value_head(emb)
        return logits, value


class MultiAgentPPOPolicy(nn.Module):
    def __init__(self, n_agents, node_dim, embed_dim, n_actions):
        super().__init__()
        self.n_agents = n_agents
        self.encoders = nn.ModuleList(
            [
                SharedEncoder(
                    input_dim=node_dim, embed_dim=embed_dim, encoder_type="deeper_gnn"
                )
                for _ in range(n_agents)
            ]
        )
        self.policy_heads = nn.ModuleList(
            [nn.Linear(embed_dim, n_actions) for _ in range(n_agents)]
        )
        self.value_heads = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(n_agents)]
        )

    def forward(self, graph_data_list):
        logits, values = [], []
        for i, graph_data in enumerate(graph_data_list):
            emb = self.encoders[i](graph_data)
            logits.append(self.policy_heads[i](emb))
            values.append(self.value_heads[i](emb))
        return logits, values


def select_action(policy, graph_data):
    logits, value = policy(graph_data)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, value


def select_multiagent_action(policy, graph_data_list):
    logits, values = policy(graph_data_list)
    actions, log_probs, values_out = [], [], []
    for i in range(len(logits)):
        probs = torch.softmax(logits[i], dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        actions.append(action.item())
        log_probs.append(log_prob)
        values_out.append(values[i])
    return actions, log_probs, values_out


def process_obs(obs):
    obs_wrapper = ObservationWrapper(mode="graph")
    graph_obs = obs_wrapper.convert(obs)
    num_vars = len(obs["variables"])
    num_clauses = len(obs["clauses"])
    nodes = torch.tensor(
        [[v] for v in obs["variables"]] + [[c] for c in obs["clauses"]],
        dtype=torch.float32,
    )
    edges = torch.tensor(graph_obs["edges"], dtype=torch.long)
    return {"nodes": nodes, "edges": edges}


def process_multiagent_obs(obs, agent_ids):
    obs_wrapper = ObservationWrapper(mode="graph")
    graph_data_list = []
    for aid in agent_ids:
        agent_obs = obs[aid]
        graph_obs = obs_wrapper.convert(agent_obs)
        num_vars = len(agent_obs["variables"])
        num_clauses = len(agent_obs["clauses"])
        nodes = torch.tensor(
            [[v] for v in agent_obs["variables"]] + [[c] for c in agent_obs["clauses"]],
            dtype=torch.float32,
        )
        edges = torch.tensor(graph_obs["edges"], dtype=torch.long)
        graph_data_list.append({"nodes": nodes, "edges": edges})
    return graph_data_list


def compute_gae(rewards, values, masks, gamma=GAMMA, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def main():
    n_agents = 2
    env = SatEnv(n_agents=n_agents, multi_agent_mode=True, observation_mode="graph")
    n_actions = env.action_space[env.agent_ids[0]].n
    policy = MultiAgentPPOPolicy(
        n_agents=n_agents, node_dim=1, embed_dim=EMBED_DIM, n_actions=n_actions
    )
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        obs, _ = env.reset()
        log_probs, values, rewards, masks, actions, states = (
            [[] for _ in range(n_agents)],
            [[] for _ in range(n_agents)],
            [[] for _ in range(n_agents)],
            [[] for _ in range(n_agents)],
            [[] for _ in range(n_agents)],
            [[] for _ in range(n_agents)],
        )
        done = {aid: False for aid in env.agent_ids}
        for step in range(STEPS_PER_EPOCH):
            graph_data_list = process_multiagent_obs(obs, env.agent_ids)
            logits, value = policy(graph_data_list)
            actions_list, log_probs_list, values_list = [], [], []
            for i in range(n_agents):
                probs = torch.softmax(logits[i], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions_list.append(action.item())
                log_probs_list.append(log_prob)
                values_list.append(value[i].squeeze(0))
            actions_dict = {aid: actions_list[i] for i, aid in enumerate(env.agent_ids)}
            next_obs, reward, terminated, truncated, info, _ = env.step(actions_dict)
            for i, aid in enumerate(env.agent_ids):
                log_probs[i].append(log_probs_list[i])
                values[i].append(values_list[i])
                rewards[i].append(torch.tensor([reward[aid]], dtype=torch.float32))
                masks[i].append(
                    torch.tensor([1.0 - float(terminated[aid] or truncated[aid])])
                )
                actions[i].append(torch.tensor(actions_list[i]))
                states[i].append(graph_data_list[i])
            obs = next_obs
            done = terminated
            if all(done.values()):
                obs, _ = env.reset()
        # Compute returns and advantages for each agent
        for i in range(n_agents):
            with torch.no_grad():
                next_value = policy([states[i][-1]])[1][0].squeeze(0)
            values_i = [v.detach() for v in values[i]]
            returns = compute_gae(rewards[i], values_i + [next_value], masks[i])
            returns = torch.stack(returns)
            values_i = torch.stack(values[i])
            log_probs_i = torch.stack(log_probs[i])
            actions_i = torch.stack(actions[i])
            # PPO update
            for _ in range(4):
                for idx in range(len(states[i])):
                    logits, value = policy([states[i][idx]])
                    probs = torch.softmax(logits[0], dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(actions_i[idx])
                    ratio = torch.exp(new_log_prob - log_probs_i[idx].detach())
                    advantage = returns[idx] - value[0].squeeze(0)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = (returns[idx] - value[0].squeeze(0)).pow(2)
                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        mean_rewards = [torch.stack(rewards[i]).mean().item() for i in range(n_agents)]
        print(
            f"Epoch {epoch+1} complete. Mean rewards: {[f'{r:.3f}' for r in mean_rewards]}"
        )


if __name__ == "__main__":
    main()
