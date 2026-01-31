"""
IID (Independent and Identically Distributed) partition strategy.

This is the baseline strategy where data is randomly and uniformly distributed
across all clients, approximating an idealized IID setting.
"""

import numpy as np
from typing import Dict, Tuple
from ..utils import set_random_state


def iid_partition(X: np.ndarray,
                 y: np.ndarray,
                 n_clients: int,
                 random_state: int = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data randomly and uniformly across clients (IID setting).

    Each client receives approximately the same number of samples, and the
    label distribution is similar across all clients.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to (X_client, y_client) tuples
    """
    rng = set_random_state(random_state)

    n_samples = X.shape[0]
    samples_per_client = n_samples // n_clients

    # Shuffle indices
    indices = rng.permutation(n_samples)

    partitions = {}
    for client_id in range(n_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < n_clients - 1 else n_samples

        client_indices = indices[start_idx:end_idx]
        partitions[client_id] = (X[client_indices], y[client_indices])

    return partitions
