"""Fixed seed management for SymbolicGym."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility across numpy, random, torch, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def set_all_seeds(seed_list=[0, 1, 2, 3, 4]):
    """Set up fixed seed management for seeds 0-4."""
    for seed in seed_list:
        set_seed(seed)
