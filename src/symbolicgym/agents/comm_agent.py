import torch
import torch.nn as nn

from symbolicgym.agents.base_agent import BaseAgent


class CommAgent(BaseAgent):
    """Multi-agent with learnable message vector."""

    def __init__(
        self,
        state_dim,
        action_dim,
        message_dim=8,
        config=None,
        debug=False,
        log_path=None,
    ):
        super().__init__(config=config, debug=debug, log_path=log_path)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.message_net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, message_dim)
        ).to(self.device)
        self.actor = nn.Sequential(
            nn.Linear(state_dim + message_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        ).to(self.device)

    def get_message(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.message_net(obs_tensor)

    def receive_messages(self, messages):
        # Aggregate peer messages (mean)
        if not messages:
            return torch.zeros(self.message_dim, device=self.device)
        msg_tensor = torch.stack(messages, dim=0)
        return msg_tensor.mean(dim=0)

    def act(self, obs, peer_messages=None):
        # Accept dict with 'feedback' key or tensor/list
        if isinstance(obs, dict) and "feedback" in obs:
            obs = obs["feedback"]
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        msg_tensor = (
            self.receive_messages(peer_messages)
            if peer_messages is not None
            else torch.zeros(self.message_dim, device=self.device)
        )
        input_tensor = torch.cat([obs_tensor, msg_tensor], dim=-1)
        logits = self.actor(input_tensor)
        action = torch.argmax(logits).item()
        if self.debug:
            self.log_debug_info(obs)
        return action

    def log_debug_info(self, obs):
        # Advanced logging: feedback vectors, actions, messages, etc.
        log_entry = {
            "step": getattr(self, "step_count", None),
            "action": getattr(self, "last_action", None),
            "feedback_vector": obs.get("feedback", None)
            if isinstance(obs, dict)
            else None,
            "message": getattr(self, "last_message", None),
            # ...add more as needed...
        }
        print(f"[DEBUG][CommAgent] {log_entry}")
        if self.log_path:
            self._log_buffer.append(log_entry)
            if len(self._log_buffer) >= 100:
                self._flush_log()

    def observe(
        self, obs, action, reward, next_obs, done, peer_messages=None, feedback=None
    ):
        # Store experience for learning (simple buffer)
        if not hasattr(self, "memory"):
            self.memory = []
        self.memory.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "peer_messages": peer_messages,
                "feedback": feedback,
            }
        )

    def update(self, optimizer=None, gamma=0.99):
        # Simple policy update stub (replace with RL algorithm as needed)
        if not hasattr(self, "memory") or not self.memory:
            return
        # For demo: clear memory (no learning)
        self.memory = []

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "message_net": self.message_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.message_net.load_state_dict(state["message_net"])
