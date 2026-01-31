"""
Base Strategy Utilities for Flower

Provides common functions used across custom strategies.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from src.utils import weighted_average


def aggregate_parameters(
    results: List[Tuple[ClientProxy, FitRes]],
) -> Parameters:
    """
    Aggregate model parameters using weighted average.

    Args:
        results: List of (client, fit_result) tuples from clients

    Returns:
        Aggregated parameters as Flower Parameters object
    """
    # Extract parameters and weights
    weights_primed = []
    parameters = [
        (np.asarray(fit_res.parameters.arrays, dtype=np.float32), fit_res.num_examples)
        for _, fit_res in results
    ]

    # Calculate total number of examples
    total_examples = sum(num_examples for _, num_examples in parameters)

    # Compute weighted average
    aggregated_parameters = [
        np.sum(
            [
                layer * num_examples / total_examples
                for layer, num_examples in [
                    (params[i], num_examples) for params, num_examples in parameters
                ]
            ],
            axis=0,
        )
        for i in range(len(parameters[0][0]))
    ]

    # Convert back to Flower Parameters
    from flwr.common import ndarray_to_bytes

    aggregated_parameters_bytes = [ndarray_to_bytes(layer) for layer in aggregated_parameters]

    return Parameters(
        tensors=aggregated_parameters_bytes,
        tensor_type="",
    )


def aggregate_fit_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """
    Aggregate training metrics from clients using weighted average.

    Args:
        metrics: List of (num_samples, metrics) tuples

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}

    _, aggregated = weighted_average(metrics)
    return aggregated


def aggregate_evaluate_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """
    Aggregate evaluation metrics from clients using weighted average.

    Args:
        metrics: List of (num_samples, metrics) tuples

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}

    _, aggregated = weighted_average(metrics)
    return aggregated
