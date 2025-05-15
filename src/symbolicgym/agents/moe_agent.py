import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent


class MoEAgent(BaseAgent):
    """Mixture of Experts agent: routes state to multiple expert policies via a gating network."""

    def __init__(self, state_dim, action_dim, num_experts=3, expert_cls=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        # Use provided expert class or default to simple MLP
        if expert_cls is None:

            class Expert(nn.Module):
                def __init__(self, state_dim, action_dim):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
                    )

                def forward(self, x):
                    return self.fc(x)

            expert_cls = Expert
        self.experts = nn.ModuleList(
            [expert_cls(state_dim, action_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, num_experts)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for expert in self.experts:
            expert.to(self.device)
        self.gate.to(self.device)
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        gate_logits = self.gate(state)
        gate_weights = F.softmax(gate_logits, dim=1)  # (1, num_experts)
        expert_outputs = torch.stack(
            [expert(state) for expert in self.experts], dim=1
        )  # (1, num_experts, action_dim)
        mixed_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(
            dim=1
        )  # (1, action_dim)
        probs = F.softmax(mixed_output, dim=1)
        action = torch.multinomial(probs, 1).item()
        return action

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # Minimal stub: clear memory (real MoE would update experts and gate)
        self.memory = []

    def save(self, path):
        torch.save(
            {
                "experts": [expert.state_dict() for expert in self.experts],
                "gate": self.gate.state_dict(),
            },
            path,
        )

    def load(self, path):
        d = torch.load(path)
        for expert, state in zip(self.experts, d["experts"]):
            expert.load_state_dict(state)
        self.gate.load_state_dict(d["gate"])
