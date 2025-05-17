import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class SimpleGNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


class GNNPolicyAgent(BaseAgent):
    """GNN-based policy agent with message passing (stub)."""

    def __init__(self, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = SimpleGNN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=lr)
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.gnn(state)
        action = int(torch.argmax(logits, dim=1).item())
        return action

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # Minimal stub: clear memory (real GNN RL would do more)
        self.memory = []

    def save(self, path):
        torch.save(self.gnn.state_dict(), path)

    def load(self, path):
        self.gnn.load_state_dict(torch.load(path))


class GNNCrossDomainAgent:
    """Stub for a cross-domain GNN agent that can train on multiple environments."""

    def __init__(self, envs):
        self.envs = envs
        # For demonstration, just use a list to store environments

    def train(self, num_episodes=1):
        # Minimal stub: run random actions in each env
        results = []
        for env in self.envs:
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(num_episodes):
                action = env.action_space.sample()
                result = env.step(action)
                obs, reward, terminated, truncated, *_ = result
                total_reward += reward
                if terminated or truncated:
                    obs, _ = env.reset()
            results.append(total_reward)
        return results
