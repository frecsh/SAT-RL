"""
Logging integration for SymbolicGym (wandb or stdout fallback).
"""
import logging

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """Unified logging interface for SymbolicGym (wandb or stdout fallback)."""

    def __init__(self, project="SymbolicGym", use_wandb=False):
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=project)
        else:
            logging.basicConfig(level=logging.INFO)

    def log(self, data: dict):
        if self.use_wandb:
            wandb.log(data)
        else:
            logging.info(data)

    def close(self):
        if self.use_wandb:
            wandb.finish()
