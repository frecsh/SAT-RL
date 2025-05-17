import torch.nn as nn


class SharedEncoder(nn.Module):
    """Shared representation encoder for cross-domain agents."""

    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)
