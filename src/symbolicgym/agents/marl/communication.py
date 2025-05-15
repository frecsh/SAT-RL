"""
Inter-agent communication: 1-bit gating mask per message (Gumbel-Sigmoid).
"""
import torch
import torch.nn.functional as F


def gumbel_sigmoid(logits, tau=1.0, hard=True):
    """
    Sample a 1-bit gating mask using Gumbel-Sigmoid.
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
