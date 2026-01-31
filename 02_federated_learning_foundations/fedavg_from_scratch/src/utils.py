"""
Utility functions for FedAvg implementation.

Provides weight serialization, reproducibility, and model state management.
"""

import random
import numpy as np
import torch
from typing import Dict, Any
from torch.nn import Module
from torch.optim import Optimizer


StateDict = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def serialize_weights(model: Module) -> StateDict:
    """
    Serialize model weights to a state dictionary.

    Args:
        model: PyTorch model to serialize

    Returns:
        StateDict: Dictionary mapping parameter names to tensors
    """
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def deserialize_weights(model: Module, weights: StateDict) -> None:
    """
    Load weights into model from state dictionary.

    Args:
        model: PyTorch model to load weights into
        weights: StateDict to load
    """
    model.load_state_dict(weights)


def compute_weight_delta(
    local_weights: StateDict,
    global_weights: StateDict
) -> StateDict:
    """
    Compute weight delta (local - global) for aggregation.

    Args:
        local_weights: Local model weights after training
        global_weights: Global model weights before training

    Returns:
        StateDict: Weight deltas for each parameter
    """
    delta = {}
    for key in local_weights.keys():
        delta[key] = local_weights[key] - global_weights[key]
    return delta


def apply_weight_update(
    model: Module,
    weight_update: StateDict
) -> None:
    """
    Apply weight updates to model in-place.

    Args:
        model: PyTorch model to update
        weight_update: Weight deltas or new weights to apply
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weight_update:
                param.data += weight_update[name].to(param.device)


def count_parameters(model: Module) -> int:
    """
    Count total number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(
    model: Module,
    optimizer_type: str,
    learning_rate: float,
    **kwargs
) -> Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('sgd', 'adam')
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer: Configured optimizer instance
    """
    if optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
