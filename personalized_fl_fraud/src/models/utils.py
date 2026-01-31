"""
Model Utilities for Personalized Federated Learning

Provides utilities for:
- Parameter manipulation (get/set specific layer parameters)
- Parameter counting (for compute budget tracking)
- Gradient management (for Per-FedAvg, Ditto)
- Model state dict operations
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def get_parameters_by_layer_type(
    model: nn.Module,
    layer_type: str = "all"
) -> List[np.ndarray]:
    """
    Extract model parameters as numpy arrays by layer type.

    Args:
        model: PyTorch model
        layer_type: One of 'all', 'feature_extractor', 'classifier'

    Returns:
        List of numpy arrays containing model parameters
    """
    state_dict = model.get_state_dict_by_layer_type(layer_type)
    return [v.cpu().numpy() for v in state_dict.values()]


def set_parameters_by_layer_type(
    model: nn.Module,
    parameters: List[np.ndarray],
    layer_type: str = "all"
) -> None:
    """
    Set model parameters from numpy arrays by layer type.

    Args:
        model: PyTorch model
        parameters: List of numpy arrays containing model parameters
        layer_type: One of 'all', 'feature_extractor', 'classifier'
    """
    # Get current state dict filtered by layer type
    if layer_type == "all":
        current_state = model.state_dict()
    else:
        current_state = model.get_state_dict_by_layer_type(layer_type)

    # Create parameter dict
    param_dict = zip(current_state.keys(), parameters)
    state_dict = OrderedDict([(k, torch.from_numpy(v)) for k, v in param_dict])

    # Load into model
    if layer_type == "all":
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict_by_layer_type(state_dict, layer_type, strict=False)


def split_model_parameters(
    model: nn.Module
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split model parameters into shared and personalizable.

    For FedPer:
    - Shared: Feature extractor parameters
    - Personal: Classifier parameters

    Args:
        model: FraudDetectionModel with feature_extractor and classifier

    Returns:
        Tuple of (shared_params, personal_params) as lists of nn.Parameter
    """
    if hasattr(model, 'feature_extractor') and hasattr(model, 'classifier'):
        shared_params = list(model.feature_extractor.parameters())
        personal_params = list(model.classifier.parameters())
    else:
        # Fallback: split by last layer
        all_params = list(model.parameters())
        personal_params = [all_params[-1]]  # Last layer
        shared_params = all_params[:-1]     # All other layers

    return shared_params, personal_params


def count_parameters(
    model: nn.Module,
    count_only_trainable: bool = True
) -> int:
    """
    Count total number of parameters in model.

    Args:
        model: PyTorch model
        count_only_trainable: If True, only count trainable parameters

    Returns:
        Total number of parameters
    """
    if count_only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_parameters_per_layer(
    model: nn.Module,
    layer_type: Optional[str] = None
) -> Dict[str, int]:
    """
    Count parameters per layer, optionally filtered by layer type.

    Args:
        model: PyTorch model
        layer_type: If specified, filter by prefix ('feature_extractor', 'classifier')

    Returns:
        Dictionary mapping layer names to parameter counts
    """
    param_counts = {}

    for name, param in model.named_parameters():
        if layer_type is None or name.startswith(layer_type):
            if param.requires_grad:
                param_counts[name] = param.numel()

    return param_counts


def compute_param_distance(
    model1: nn.Module,
    model2: nn.Module,
    layer_type: str = "all"
) -> float:
    """
    Compute L2 distance between two models' parameters.

    Used in Ditto for regularization term and FedProx for proximal term.

    Args:
        model1: First model
        model2: Second model
        layer_type: Layer type to compare ('all', 'feature_extractor', 'classifier')

    Returns:
        L2 norm of parameter differences
    """
    # Get parameters from both models
    params1 = model1.get_state_dict_by_layer_type(layer_type)
    params2 = model2.get_state_dict_by_layer_type(layer_type)

    # Compute L2 distance
    distance = 0.0
    for key in params1.keys():
        if key in params2:
            distance += torch.sum((params1[key] - params2[key]) ** 2).item()

    return distance


def compute_moreau_envelope(
    model: nn.Module,
    global_params: Dict[str, torch.Tensor],
    beta: float,
    layer_type: str = "all"
) -> torch.Tensor:
    """
    Compute Moreau envelope regularization term for Per-FedAvg.

    Moreau envelope: L_beta(w) = L(w) + (beta/2) * ||w - w_global||^2

    Args:
        model: Current model
        global_params: Global model parameters (state dict)
        beta: Regularization strength
        layer_type: Layer type to regularize

    Returns:
        Scalar tensor containing Moreau envelope term
    """
    current_params = model.get_state_dict_by_layer_type(layer_type)

    moreau_term = 0.0
    for key in current_params.keys():
        if key in global_params:
            moreau_term += torch.sum(
                (current_params[key] - global_params[key]) ** 2
            )

    return (beta / 2) * moreau_term


def aggregate_parameters(
    parameters_list: List[List[np.ndarray]],
    weights: List[float]
) -> List[np.ndarray]:
    """
    Aggregate parameters from multiple clients using weighted average.

    Args:
        parameters_list: List of parameter lists, one per client
        weights: Weight for each client (typically based on num_samples)

    Returns:
        Aggregated parameters as list of numpy arrays
    """
    if len(parameters_list) == 0:
        raise ValueError("Cannot aggregate empty parameter list")

    if len(parameters_list) != len(weights):
        raise ValueError(
            f"Number of parameter lists ({len(parameters_list)}) "
            f"must match number of weights ({len(weights)})"
        )

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero")
    normalized_weights = [w / total_weight for w in weights]

    # Aggregate each parameter
    aggregated = []
    for params in zip(*parameters_list):
        # Stack and compute weighted average
        stacked = np.stack(params)
        weighted_avg = np.average(stacked, axis=0, weights=normalized_weights)
        aggregated.append(weighted_avg)

    return aggregated


def compute_flops_per_forward(
    input_dim: int,
    hidden_dims: List[int],
    batch_size: int = 1
) -> int:
    """
    Estimate FLOPs for a forward pass through the MLP.

    Approximate FLOPs: 2 * m * n for (m, n) linear layer
    (multiply-add counts as 2 operations)

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        batch_size: Batch size

    Returns:
        Estimated FLOPs per forward pass
    """
    dims = [input_dim] + hidden_dims + [1]  # Add output dimension

    flops = 0
    for i in range(len(dims) - 1):
        m, n = dims[i], dims[i + 1]
        flops += 2 * m * n * batch_size

    # Add activation function FLOPs (negligible but included)
    flops += sum(hidden_dims + [1]) * batch_size

    return flops


def compute_flops_per_backward(
    input_dim: int,
    hidden_dims: List[int],
    batch_size: int = 1
) -> int:
    """
    Estimate FLOPs for a backward pass.

    Backward pass typically ~2x forward pass FLOPs.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        batch_size: Batch size

    Returns:
        Estimated FLOPs per backward pass
    """
    forward_flops = compute_flops_per_forward(input_dim, hidden_dims, batch_size)
    return 2 * forward_flops


def compute_communication_cost(
    parameters: List[np.ndarray]
) -> int:
    """
    Compute total bytes for transmitting parameters.

    Args:
        parameters: List of numpy arrays

    Returns:
        Total bytes (assuming float32 = 4 bytes per parameter)
    """
    total_params = sum(p.size for p in parameters)
    bytes_per_param = 4  # float32
    return total_params * bytes_per_param


def get_layer_names(model: nn.Module) -> List[str]:
    """
    Get list of all layer names in the model.

    Args:
        model: PyTorch model

    Returns:
        List of layer names (e.g., ['feature_extractor.0', 'classifier.0'])
    """
    return list(model.state_dict().keys())


def freeze_model_layers(
    model: nn.Module,
    layer_names: List[str]
) -> None:
    """
    Freeze specific layers by name.

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layer_names):
            param.requires_grad = False


def unfreeze_model_layers(
    model: nn.Module,
    layer_names: List[str]
) -> None:
    """
    Unfreeze specific layers by name.

    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layer_names):
            param.requires_grad = True
