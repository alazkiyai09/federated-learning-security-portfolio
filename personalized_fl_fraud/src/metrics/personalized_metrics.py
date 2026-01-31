"""
Per-Client Metrics for Personalized FL

Provides functions for computing and tracking per-client performance,
personalization benefit, and fairness metrics.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json

import numpy as np


def compute_per_client_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    client_id: int
) -> Dict[str, float]:
    """
    Compute metrics for a single client.

    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        client_id: Client identifier

    Returns:
        Metrics dictionary
    """
    from ..utils import compute_fraud_metrics
    return compute_fraud_metrics(predictions, targets)


def compute_personalization_benefit(
    global_metrics: Dict[int, Dict[str, float]],
    personalized_metrics: Dict[int, Dict[str, float]]
) -> Dict[int, Dict[str, float]]:
    """
    Compute personalization benefit (delta) for each client.

    Args:
        global_metrics: Metrics from global model per client
        personalized_metrics: Metrics from personalized model per client

    Returns:
        Dictionary mapping client_id to metric deltas
    """
    deltas = {}

    for client_id in global_metrics.keys():
        if client_id not in personalized_metrics:
            continue

        global_client = global_metrics[client_id]
        personal_client = personalized_metrics[client_id]

        delta = {}
        for metric_name in global_client.keys():
            if metric_name in personal_client:
                delta[metric_name] = (
                    personal_client[metric_name] - global_client[metric_name]
                )

        deltas[client_id] = delta

    return deltas


def aggregate_metrics(
    per_client_metrics: Dict[int, Dict[str, float]],
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Aggregate per-client metrics into summary statistics.

    Args:
        per_client_metrics: Dictionary mapping client_id to metrics
        aggregation: Aggregation method ('mean', 'median', 'std', 'min', 'max')

    Returns:
        Aggregated metrics dictionary
    """
    if not per_client_metrics:
        return {}

    aggregated = {}

    # Get all metric names
    metric_names = set()
    for client_metrics in per_client_metrics.values():
        metric_names.update(client_metrics.keys())

    # Aggregate each metric
    for metric_name in metric_names:
        values = [
            client_metrics.get(metric_name, 0)
            for client_metrics in per_client_metrics.values()
            if metric_name in client_metrics
        ]

        if not values:
            continue

        if aggregation == "mean":
            aggregated[metric_name] = float(np.mean(values))
        elif aggregation == "median":
            aggregated[metric_name] = float(np.median(values))
        elif aggregation == "std":
            aggregated[metric_name] = float(np.std(values))
        elif aggregation == "min":
            aggregated[metric_name] = float(np.min(values))
        elif aggregation == "max":
            aggregated[metric_name] = float(np.max(values))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return aggregated


def compute_worst_client_performance(
    per_client_metrics: Dict[int, Dict[str, float]],
    metric_name: str = "auc"
) -> Dict[str, float]:
    """
    Identify and return worst-performing client metrics.

    Args:
        per_client_metrics: Per-client metrics dictionary
        metric_name: Metric to use for ranking (default: auc)

    Returns:
        Dictionary with worst client info
    """
    if not per_client_metrics:
        return {
            'worst_client_id': -1,
            'worst_client_metric': 0.0,
            'worst_client_metrics': {}
        }

    # Find worst client
    worst_client_id = min(
        per_client_metrics.keys(),
        key=lambda cid: per_client_metrics[cid].get(metric_name, 0)
    )

    worst_metric = per_client_metrics[worst_client_id].get(metric_name, 0)

    return {
        'worst_client_id': worst_client_id,
        'worst_client_metric': worst_metric,
        'worst_client_metrics': per_client_metrics[worst_client_id]
    }


def compute_performance_variance(
    per_client_metrics: Dict[int, Dict[str, float]],
    metric_name: str = "auc"
) -> Dict[str, float]:
    """
    Compute variance in performance across clients.

    Args:
        per_client_metrics: Per-client metrics dictionary
        metric_name: Metric to analyze

    Returns:
        Variance statistics
    """
    if not per_client_metrics:
        return {'variance': 0, 'std': 0, 'range': 0, 'coeff_variation': 0}

    values = [
        client_metrics.get(metric_name, 0)
        for client_metrics in per_client_metrics.values()
    ]

    variance = float(np.var(values))
    std = float(np.std(values))
    value_range = float(np.max(values) - np.min(values))
    mean = float(np.mean(values))

    coeff_variation = std / mean if mean > 0 else 0

    return {
        'variance': variance,
        'std': std,
        'range': value_range,
        'mean': mean,
        'coeff_variation': coeff_variation
    }


def save_per_client_metrics(
    per_client_metrics: Dict[int, Dict[str, float]],
    output_path: str
) -> None:
    """
    Save per-client metrics to JSON file.

    Args:
        per_client_metrics: Per-client metrics dictionary
        output_path: Path to save metrics
    """
    # Convert int keys to strings for JSON serialization
    serializable = {
        str(client_id): metrics
        for client_id, metrics in per_client_metrics.items()
    }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_per_client_metrics(
    input_path: str
) -> Dict[int, Dict[str, float]]:
    """
    Load per-client metrics from JSON file.

    Args:
        input_path: Path to metrics file

    Returns:
        Per-client metrics dictionary
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Convert string keys back to int
    return {
        int(client_id): metrics
        for client_id, metrics in data.items()
    }


def compare_methods_per_client(
    results_by_method: Dict[str, Dict[int, Dict[str, float]]],
    metric_name: str = "auc"
) -> Dict[str, Dict[str, float]]:
    """
    Compare methods across all clients.

    Args:
        results_by_method: Dict mapping method_name to per-client metrics
        metric_name: Metric to compare

    Returns:
        Comparison statistics
    """
    comparison = {}

    for method_name, per_client_metrics in results_by_method.items():
        # Extract metric values
        values = [
            client_metrics.get(metric_name, 0)
            for client_metrics in per_client_metrics.values()
        ]

        comparison[method_name] = {
            'mean': float(np.mean(values)) if values else 0,
            'std': float(np.std(values)) if values else 0,
            'min': float(np.min(values)) if values else 0,
            'max': float(np.max(values)) if values else 0,
            'median': float(np.median(values)) if values else 0,
        }

    return comparison
