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


class CTDEAgent(BaseAgent):
    """Centralized Training, Decentralized Execution (CTDE) multi-agent setup."""

    def __init__(self, n_agents, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = CentralizedCritic(n_agents, state_dim, action_dim).to(self.device)
        self.optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []

    def act(self, states):
        # For demo, random actions for all agents
        return [
            torch.randint(0, self.action_dim, (1,)).item() for _ in range(self.n_agents)
        ]

    def observe(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def update(self):
        # Minimal stub: clear memory (real CTDE would do more)
        self.memory = []

    def save(self, path):
        torch.save(self.critic.state_dict(), path)

    def load(self, path):
        self.critic.load_state_dict(torch.load(path))
