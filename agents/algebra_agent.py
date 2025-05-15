# Forward/Inverse/Policy agent architecture for Symbolic Algebra MVP
import torch
import torch.nn as nn


class AlgebraAgent(nn.Module):
    def __init__(self, token_dim, feedback_dim, latent_dim, num_actions):
        super().__init__()
        self.token_encoder = nn.Sequential(nn.Linear(token_dim, latent_dim), nn.ReLU())
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, latent_dim), nn.ReLU()
        )
        self.latent_dim = latent_dim
        # Heads
        self.forward_head = nn.Linear(latent_dim, latent_dim)
        self.inverse_head = nn.Linear(latent_dim * 2, num_actions)
        self.policy_head = nn.Linear(latent_dim, num_actions)

    def forward(self, tokens, feedback, next_tokens=None):
        z_state = self.token_encoder(tokens) + self.feedback_encoder(feedback)
        logits = self.policy_head(z_state)
        outputs = {"logits": logits}
        if next_tokens is not None:
            z_next = self.token_encoder(next_tokens)
            outputs["forward_pred"] = self.forward_head(z_state)
            outputs["inverse_pred"] = self.inverse_head(
                torch.cat([z_state, z_next], dim=-1)
            )
        return outputs
