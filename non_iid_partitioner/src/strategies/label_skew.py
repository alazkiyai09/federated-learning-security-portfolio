"""
Label skew partition strategy using Dirichlet distribution.

This simulates non-IID data where different clients have different
class distributions. The concentration parameter alpha controls the
degree of heterogeneity.
"""

import numpy as np
from typing import Dict, Tuple, List
from ..utils import set_random_state


def dirichlet_partition(X: np.ndarray,
                       y: np.ndarray,
                       n_clients: int,
                       alpha: float = 1.0,
                       min_samples_per_client: int = 1,
                       random_state: int = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data with label skew using Dirichlet distribution.

    The Dirichlet distribution controls how different class proportions
    are distributed across clients:

    - alpha → 0: Extreme non-IID, each client dominated by 1-2 classes
    - alpha = 1: Uniform Dirichlet (moderate heterogeneity)
    - alpha → ∞: Approaches IID (uniform class distribution)

    Mathematical foundation:
        For K classes and N clients, we sample proportions p_i ~ Dir(alpha)
        for each client, where p_i determines the fraction of each class
        allocated to that client.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        alpha: Dirichlet concentration parameter (lower = more heterogeneity)
        min_samples_per_client: Minimum samples each client must receive
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to (X_client, y_client) tuples

    Raises:
        ValueError: If alpha <= 0 or if data cannot be partitioned as specified
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    rng = set_random_state(random_state)

    n_samples = X.shape[0]
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Map class labels to 0..(K-1) for indexing
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[label] for label in y])

    # Group sample indices by class
    class_indices = {class_idx: np.where(y_mapped == class_idx)[0]
                    for class_idx in range(n_classes)}

    # Sample Dirichlet distribution for each client
    # Shape: (n_clients, n_classes)
    proportions = rng.dirichlet(alpha=np.ones(n_classes) * alpha, size=n_clients)

    # Allocate samples to clients based on proportions
    partitions = {client_id: [] for client_id in range(n_clients)}

    for class_idx in range(n_classes):
        class_samples = class_indices[class_idx]
        n_class_samples = len(class_samples)

        # Get proportion of this class for each client
        class_proportions = proportions[:, class_idx]

        # Scale to actual sample counts (float)
        client_counts = class_proportions * n_class_samples

        # Ensure minimum allocation while maintaining proportions
        # Clients with very small proportions get at least min_samples_per_client
        if min_samples_per_client > 0:
            # Identify clients below threshold
            below_threshold = client_counts < min_samples_per_client
            n_below = below_threshold.sum()

            if n_below > 0:
                # Allocate minimum to those clients
                min_allocation = n_below * min_samples_per_client
                if min_allocation >= n_class_samples:
                    raise ValueError(
                        f"Cannot allocate {min_samples_per_client} min samples per client "
                        f"for class {class_idx} with only {n_class_samples} samples"
                    )

                # Redistribute remaining samples proportionally
                remaining_samples = n_class_samples - min_allocation
                clients_above = ~below_threshold
                if clients_above.sum() > 0:
                    # Scale up proportions for clients above threshold
                    total_prop_above = client_counts[clients_above].sum()
                    client_counts[clients_above] = (
                        client_counts[clients_above] / total_prop_above * remaining_samples
                    )

                # Set minimum for clients below threshold
                client_counts[below_threshold] = min_samples_per_client

        # Round to integers (must sum to n_class_samples)
        client_counts_int = np.floor(client_counts).astype(int)
        remainder = n_class_samples - client_counts_int.sum()

        # Distribute remainder to clients with largest fractional parts
        if remainder > 0:
            fractional_parts = client_counts - client_counts_int
            remainder_clients = np.argsort(-fractional_parts)[:remainder]
            client_counts_int[remainder_clients] += 1

        # Shuffle samples within this class
        shuffled_indices = rng.permutation(class_samples)

        # Allocate samples to clients
        start = 0
        for client_id in range(n_clients):
            count = client_counts_int[client_id]
            if count > 0:
                end = start + count
                partitions[client_id].extend(shuffled_indices[start:end].tolist())
                start = end

    # Convert to numpy arrays and create final partition dict
    result = {}
    for client_id in range(n_clients):
        client_indices = np.array(partitions[client_id], dtype=int)
        result[client_id] = (X[client_indices], y[client_indices])

    return result


def dirichlet_partition_indices(y: np.ndarray,
                                n_clients: int,
                                alpha: float = 1.0,
                                min_samples_per_client: int = 1,
                                random_state: int = None) -> Dict[int, np.ndarray]:
    """
    Partition indices with label skew using Dirichlet distribution.

    This is a convenience function that returns only the sample indices
    rather than the full (X, y) tuples.

    Args:
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        alpha: Dirichlet concentration parameter
        min_samples_per_client: Minimum samples each client must receive
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to array of sample indices

    Raises:
        ValueError: If alpha <= 0 or if data cannot be partitioned as specified
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    rng = set_random_state(random_state)

    n_samples = len(y)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Map class labels to 0..(K-1) for indexing
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[label] for label in y])

    # Group sample indices by class
    class_indices = {class_idx: np.where(y_mapped == class_idx)[0]
                    for class_idx in range(n_classes)}

    # Sample Dirichlet distribution for each client
    proportions = rng.dirichlet(alpha=np.ones(n_classes) * alpha, size=n_clients)

    # Allocate samples to clients
    partitions = {client_id: [] for client_id in range(n_clients)}

    for class_idx in range(n_classes):
        class_samples = class_indices[class_idx]
        n_class_samples = len(class_samples)
        class_proportions = proportions[:, class_idx]
        client_counts = class_proportions * n_class_samples

        # Ensure minimum allocation
        if min_samples_per_client > 0:
            below_threshold = client_counts < min_samples_per_client
            n_below = below_threshold.sum()

            if n_below > 0:
                min_allocation = n_below * min_samples_per_client
                if min_allocation >= n_class_samples:
                    raise ValueError(
                        f"Cannot allocate {min_samples_per_client} min samples per client "
                        f"for class {class_idx} with only {n_class_samples} samples"
                    )

                remaining_samples = n_class_samples - min_allocation
                clients_above = ~below_threshold
                if clients_above.sum() > 0:
                    total_prop_above = client_counts[clients_above].sum()
                    client_counts[clients_above] = (
                        client_counts[clients_above] / total_prop_above * remaining_samples
                    )

                client_counts[below_threshold] = min_samples_per_client

        # Round to integers
        client_counts_int = np.floor(client_counts).astype(int)
        remainder = n_class_samples - client_counts_int.sum()

        if remainder > 0:
            fractional_parts = client_counts - client_counts_int
            remainder_clients = np.argsort(-fractional_parts)[:remainder]
            client_counts_int[remainder_clients] += 1

        # Shuffle and allocate
        shuffled_indices = rng.permutation(class_samples)
        start = 0
        for client_id in range(n_clients):
            count = client_counts_int[client_id]
            if count > 0:
                end = start + count
                partitions[client_id].extend(shuffled_indices[start:end].tolist())
                start = end

    # Convert to numpy arrays
    result = {}
    for client_id in range(n_clients):
        result[client_id] = np.array(partitions[client_id], dtype=int)

    return result
