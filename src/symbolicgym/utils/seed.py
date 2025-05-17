"""Global seeding utility for reproducibility in SymbolicGym experiments."""

import os
import random

import numpy as np


def set_global_seed(seed):
    """Set global seed for numpy, random, and PYTHONHASHSEED."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
