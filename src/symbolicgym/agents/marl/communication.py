"""Inter-agent communication: 1-bit gating mask per message (Gumbel-Sigmoid)."""

import torch


def gumbel_sigmoid(logits, tau=1.0, hard=True):
    """Sample a 1-bit gating mask using Gumbel-Sigmoid.

    Args:
        logits: torch.Tensor
        tau: temperature
        hard: if True, returns hard (0/1) mask
    Returns:
        mask: torch.Tensor (same shape as logits)
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = torch.sigmoid((logits + gumbel_noise) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y


gumbel_sigmoid.__module__ = "symbolicgym.agents.marl.communication"


class Message:
    """Basic message structure for agent communication."""

    def __init__(self, sender, content, hint=False, blackboard_entry=None):
        self.sender = sender
        self.content = content  # e.g., vector, dict, etc.
        self.hint = hint
        self.blackboard_entry = blackboard_entry

    def __repr__(self):
        return (
            f"Message(sender={self.sender}, hint={self.hint}, content={self.content})"
        )


class CommunicationChannel:
    """Simple message passing channel for agents."""

    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def receive(self, agent_id=None):
        if agent_id is None:
            return self.messages
        return [m for m in self.messages if m.sender != agent_id]

    def clear(self):
        self.messages = []
