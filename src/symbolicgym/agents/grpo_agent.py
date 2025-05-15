import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class GRPOAgent(BaseAgent):
    """Generalized Recurrent Policy Optimization (GRPO) agent with LSTM policy/value networks."""

    def __init__(self, state_dim, action_dim, domain, lr=3e-4, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.domain = domain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = nn.LSTM(
            input_size=state_dim, hidden_size=hidden_size, batch_first=True
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_net = nn.LSTM(
            input_size=state_dim, hidden_size=hidden_size, batch_first=True
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_net.parameters())
            + list(self.value_head.parameters()),
            lr=lr,
        )
        self.memory = []
        self.hidden_policy = None
        self.hidden_value = None

    def act(self, state_seq):
        # state_seq: shape (seq_len, state_dim)
        state_seq = (
            torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        )  # (1, seq_len, state_dim)
        out, self.hidden_policy = self.policy_net(state_seq, self.hidden_policy)
        logits = self.policy_head(out[:, -1, :])
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        return action

    def observe(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def update(self):
        # Minimal stub: clear memory (real GRPO would do recurrent advantage estimation, etc.)
        self.memory = []
        self.hidden_policy = None
        self.hidden_value = None

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_net": self.value_net.state_dict(),
                "value_head": self.value_head.state_dict(),
            },
            path,
        )

    def load(self, path):
        d = torch.load(path)
        self.policy_net.load_state_dict(d["policy_net"])
        self.policy_head.load_state_dict(d["policy_head"])
        self.value_net.load_state_dict(d["value_net"])
        self.value_head.load_state_dict(d["value_head"])
