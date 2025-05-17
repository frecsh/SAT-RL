import torch
import torch.nn as nn
import torch.optim as optim

from symbolicgym.agents.base_agent import BaseAgent


class CentralizedCritic(nn.Module):
    def __init__(self, n_agents, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_agents * (state_dim + action_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states, actions):
        x = torch.cat([states.flatten(), actions.flatten()]).unsqueeze(0)
        return self.fc(x)


class AttentionCentralizedCritic(nn.Module):
    def __init__(self, n_agents, state_dim, action_dim, embed_dim=64, n_heads=2):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(n_agents * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states, actions):
        # states, actions: (n_agents, state_dim/action_dim)
        s = self.state_embed(states)
        a = self.action_embed(actions)
        x = s + a
        attn_out, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        flat = attn_out.flatten(start_dim=1)
        return self.fc(flat)


class AgentActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.fc(state)


class CTDEAgent(BaseAgent):
    """Centralized Training, Decentralized Execution (CTDE) multi-agent setup."""

    def __init__(self, config=None, debug=False, log_path=None):
        super().__init__(config=config, debug=debug, log_path=log_path)
        self.n_agents = config.get("n_agents", 1)
        self.state_dim = config.get("state_dim", 1)
        self.action_dim = config.get("action_dim", 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_attention_critic = config.get("use_attention_critic", True)
        if use_attention_critic:
            self.critic = AttentionCentralizedCritic(
                self.n_agents, self.state_dim, self.action_dim
            ).to(self.device)
        else:
            self.critic = CentralizedCritic(
                self.n_agents, self.state_dim, self.action_dim
            ).to(self.device)
        self.actors = nn.ModuleList(
            [
                AgentActor(self.state_dim, self.action_dim).to(self.device)
                for _ in range(self.n_agents)
            ]
        )
        lr = config.get("lr", 1e-3)
        self.optim = optim.Adam(
            list(self.critic.parameters()) + list(self.actors.parameters()), lr=lr
        )
        self.memory = []

    def act(self, states, message_channel=None):
        # states: list or tensor of per-agent states
        actions = []
        for i, actor in enumerate(self.actors):
            state = torch.tensor(states[i], dtype=torch.float32, device=self.device)
            logits = actor(state)
            action = torch.argmax(logits).item()
            actions.append(action)
        # Optionally send/receive messages
        if message_channel is not None:
            for i in range(self.n_agents):
                message_channel.send(
                    {"agent": i, "msg": f"step_{len(message_channel.messages)}"}
                )
        if self.debug:
            self.log_debug_info(states)
        return actions

    def log_debug_info(self, obs):
        # Advanced logging: feedback vectors, actions, etc.
        log_entry = {
            "step": getattr(self, "step_count", None),
            "action": getattr(self, "last_action", None),
            "feedback_vector": obs.get("feedback", None)
            if isinstance(obs, dict)
            else None,
        }
        print(f"[DEBUG][CTDEAgent] {log_entry}")
        if self.log_path:
            self._log_buffer.append(log_entry)
            if len(self._log_buffer) >= 100:
                self._flush_log()

    def observe(self, states, actions, rewards, next_states, dones, messages=None):
        # Optionally observe messages
        self.memory.append((states, actions, rewards, next_states, dones, messages))

    def update(self):
        # Parallel agent optimization (stub)
        # In practice, batch update for all agents and critic
        if not self.memory:
            return
        # Example: clear memory (replace with real update logic)
        self.memory = []

    def save(self, path):
        torch.save(self.critic.state_dict(), path)

    def load(self, path):
        self.critic.load_state_dict(torch.load(path))
