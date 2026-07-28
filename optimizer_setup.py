"""Optimizer factory for IndustrialSAM.

Freezes encoder parameters and builds an AdamW optimizer over the trainable
adapter + classifier parameters only.
"""

import torch
from model import IndustrialSAM
from model_setup import _ADAPTER_PREFIX


def build_optimizer(model, lr=1e-4, weight_decay=0.01):
    """Create an AdamW optimizer for all trainable parameters.

    Parameters
    ----------
    model : IndustrialSAM
    lr : float  Learning rate (default 1e-4).
    weight_decay : float  Weight decay (default 0.01).

    Returns
    -------
    torch.optim.AdamW
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(trainable) > 0, "No trainable parameters found in model"
    assert (
        len(trainable) == expected
    ), f"Trainable param count mismatch: {len(trainable)} vs expected {expected}"
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
