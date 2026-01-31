"""
Utility functions for data partitioning.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def validate_partitions(partitions: Dict[int, np.ndarray],
                       n_samples: int,
                       n_clients: int) -> None:
    """
    Validate that partitions meet basic requirements.

    Args:
        partitions: Dictionary mapping client_id to sample indices
        n_samples: Total number of samples in the dataset
        n_clients: Expected number of clients

    Raises:
        ValueError: If validation fails
    """
    # Check number of clients
    if len(partitions) != n_clients:
        raise ValueError(f"Expected {n_clients} clients, got {len(partitions)}")

    # Check client IDs
    expected_ids = set(range(n_clients))
    actual_ids = set(partitions.keys())
    if expected_ids != actual_ids:
        raise ValueError(f"Expected client IDs {expected_ids}, got {actual_ids}")

    # Collect all assigned indices
    all_indices = []
    for client_id, indices in partitions.items():
        if not isinstance(indices, np.ndarray):
            raise ValueError(f"Indices for client {client_id} must be numpy array")
        if len(indices) > 0:
            all_indices.extend(indices.tolist())

    # Check coverage
    all_indices = np.array(all_indices)
    unique_indices = np.unique(all_indices)

    if len(unique_indices) != n_samples:
        raise ValueError(
            f"Expected {n_samples} unique samples, got {len(unique_indices)}"
        )

    if not np.array_equal(np.sort(unique_indices), np.arange(n_samples)):
        raise ValueError("Sample indices must be [0, n_samples) without gaps")


def validate_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data dimensions and consistency.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)

    Raises:
        ValueError: If validation fails
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: "
            f"got {X.shape[0]} and {y.shape[0]}"
        )

    if X.shape[0] == 0:
        raise ValueError("Cannot partition empty dataset")


def set_random_state(random_state: int = None) -> np.random.RandomState:
    """
    Create a random state for reproducibility.

    Args:
        random_state: Seed for random number generation

    Returns:
        RandomState object
    """
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None")

    return np.random.RandomState(random_state)


def compute_label_distribution(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Compute the distribution of labels.

    Args:
        y: Label array
        n_classes: Number of classes

    Returns:
        Array of shape (n_classes,) with counts per class
    """
    distribution = np.zeros(n_classes, dtype=int)
    unique, counts = np.unique(y, return_counts=True)
    distribution[unique] = counts
    return distribution


def entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a distribution.

    Args:
        distribution: Probability distribution (sums to 1)

    Returns:
        Entropy value
    """
    distribution = distribution[distribution > 0]  # Remove zeros
    return -np.sum(distribution * np.log(distribution))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Args:
        values: Array of values (e.g., sample counts per client)

    Returns:
        Gini coefficient (0 = perfect equality, 1 = maximal inequality)
    """
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
