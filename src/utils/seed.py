"""Reproducibility utilities for setting random seeds."""

import random
import os
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[ok] Global seed set to {seed}")
