"""
Quantity skew partition strategy using power law distribution.

This simulates non-IID data where different clients have vastly different
amounts of data, following a heavy-tailed distribution.
"""

import numpy as np
from typing import Dict, Tuple
from ..utils import set_random_state


def power_law_allocation(n_samples: int,
                        n_clients: int,
                        exponent: float = 1.5,
                        min_samples_per_client: int = 1,
                        random_state: int = None) -> np.ndarray:
    """
    Generate power law distribution of sample counts across clients.

    The power law (Pareto) distribution generates heavy-tailed allocations
    where a few clients have many samples and most clients have few samples.

    Mathematical foundation:
        P(x) ~ x^(-exponent)
        - exponent = 1: Very heavy tail (maximal inequality)
        - exponent = 2: Standard Pareto distribution
        - exponent > 2: More moderate inequality
        - exponent → ∞: Approaches uniform distribution

    Args:
        n_samples: Total number of samples to distribute
        n_clients: Number of clients
        exponent: Power law exponent (higher = less skew, lower = more skew)
        min_samples_per_client: Minimum samples each client must receive
        random_state: Seed for reproducibility

    Returns:
        Array of shape (n_clients,) with sample counts per client

    Raises:
        ValueError: If parameters are invalid
    """
    if exponent <= 1:
        raise ValueError(f"exponent must be > 1, got {exponent}")

    if min_samples_per_client * n_clients > n_samples:
        raise ValueError(
            f"min_samples_per_client ({min_samples_per_client}) * n_clients "
            f"({n_clients}) cannot exceed n_samples ({n_samples})"
        )

    rng = set_random_state(random_state)

    # Generate power law distributed values
    # Use Pareto distribution: x ~ (1 + U) ^ (-1/exponent)
    # where U ~ Uniform(0, 1)
    u = rng.random(n_clients)
    raw_counts = (1 + u) ** (-1 / (exponent - 1))

    # Normalize and scale
    raw_counts = raw_counts / raw_counts.sum()

    # Allocate with minimum constraint
    available_samples = n_samples - (min_samples_per_client * n_clients)
    counts = min_samples_per_client + raw_counts * available_samples

    # Round to integers while preserving total
    counts_int = np.floor(counts).astype(int)
    remainder = n_samples - counts_int.sum()

    # Distribute remainder to clients with largest fractional parts
    if remainder > 0:
        fractional_parts = counts - counts_int
        remainder_clients = np.argsort(-fractional_parts)[:remainder]
        counts_int[remainder_clients] += 1

    return counts_int


def quantity_skew_partition(X: np.ndarray,
                           y: np.ndarray,
                           n_clients: int,
                           exponent: float = 1.5,
                           min_samples_per_client: int = 1,
                           random_state: int = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data with quantity skew using power law distribution.

    This strategy simulates realistic FL scenarios where clients have
    vastly different amounts of data (e.g., varying user activity levels).

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        exponent: Power law exponent controlling skew (higher = less skew)
        min_samples_per_client: Minimum samples each client must receive
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to (X_client, y_client) tuples
    """
    rng = set_random_state(random_state)

    # Get sample counts per client
    client_counts = power_law_allocation(
        n_samples=X.shape[0],
        n_clients=n_clients,
        exponent=exponent,
        min_samples_per_client=min_samples_per_client,
        random_state=random_state
    )

    # Shuffle all samples
    all_indices = rng.permutation(X.shape[0])

    # Allocate samples to clients
    partitions = {}
    start_idx = 0
    for client_id in range(n_clients):
        count = client_counts[client_id]
        end_idx = start_idx + count
        client_indices = all_indices[start_idx:end_idx]
        partitions[client_id] = (X[client_indices], y[client_indices])
        start_idx = end_idx

    return partitions


def quantity_skew_partition_indices(y: np.ndarray,
                                    n_clients: int,
                                    exponent: float = 1.5,
                                    min_samples_per_client: int = 1,
                                    random_state: int = None) -> Dict[int, np.ndarray]:
    """
    Partition indices with quantity skew using power law distribution.

    This is a convenience function that returns only the sample indices.

    Args:
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        exponent: Power law exponent controlling skew
        min_samples_per_client: Minimum samples each client must receive
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to array of sample indices
    """
    rng = set_random_state(random_state)

    n_samples = len(y)

    # Get sample counts per client
    client_counts = power_law_allocation(
        n_samples=n_samples,
        n_clients=n_clients,
        exponent=exponent,
        min_samples_per_client=min_samples_per_client,
        random_state=random_state
    )

    # Shuffle all samples
    all_indices = rng.permutation(n_samples)

    # Allocate samples to clients
    partitions = {}
    start_idx = 0
    for client_id in range(n_clients):
        count = client_counts[client_id]
        end_idx = start_idx + count
        client_indices = all_indices[start_idx:end_idx]
        partitions[client_id] = client_indices
        start_idx = end_idx

    return partitions
