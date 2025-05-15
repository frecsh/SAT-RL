from .base_agent import BaseAgent


class ImitationAgent(BaseAgent):
    """Agent that learns by imitating oracle demonstrations."""

    def __init__(self, policy):
        self.policy = policy
        self.memory = []

    def act(self, state):
        return self.policy(state)

    def observe(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # Fit policy to memory (stub)
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
