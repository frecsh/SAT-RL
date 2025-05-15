import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x)


class PPOAgent(BaseAgent):
    """PPO agent with domain-specific policy/value heads."""

    def __init__(self, state_dim, action_dim, domain, lr=3e-4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.domain = domain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(state_dim, action_dim).to(self.device)
        self.value_net = ValueNet(state_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        return action

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # Minimal stub: clear memory (real PPO would do GAE, advantage, etc.)
        self.memory = []

    def save(self, path):
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "value": self.value_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        d = torch.load(path)
        self.policy_net.load_state_dict(d["policy"])
        self.value_net.load_state_dict(d["value"])
