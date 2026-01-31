"""
Feature skew partition strategy using clustering.

This simulates non-IID data where clients have data from different regions
of the feature space, even if label distributions are similar.
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Dict, Tuple
from ..utils import set_random_state


def feature_based_partition(X: np.ndarray,
                           y: np.ndarray,
                           n_clients: int,
                           n_clusters: int = None,
                           use_minibatch: bool = False,
                           random_state: int = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data based on feature space clustering.

    This strategy groups similar samples (in feature space) together and
    assigns clusters to clients, simulating scenarios where clients have
    data from different subpopulations or regions.

    Use cases:
    - Geographic distribution (e.g., banks in different regions)
    - Demographic differences (e.g., age groups, income levels)
    - Temporal shifts (e.g., seasonal variations)

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        n_clients: Number of clients to partition data across
        n_clusters: Number of clusters (default: n_clients)
        use_minibatch: Use MiniBatchKMeans for large datasets
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to (X_client, y_client) tuples
    """
    if n_clusters is None:
        n_clusters = n_clients

    if n_clusters > X.shape[0]:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed n_samples ({X.shape[0]})"
        )

    rng = set_random_state(random_state)

    # Perform clustering on feature space
    if use_minibatch or X.shape[0] > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=1000
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    cluster_labels = kmeans.fit_predict(X)

    # Assign clusters to clients
    # If n_clusters == n_clients: 1 cluster per client
    # If n_clusters < n_clients: distribute clusters evenly
    # If n_clusters > n_clients: merge clusters
    partitions = {}

    if n_clusters == n_clients:
        # Direct assignment
        for client_id in range(n_clients):
            mask = cluster_labels == client_id
            partitions[client_id] = (X[mask], y[mask])

    elif n_clusters < n_clients:
        # Distribute clusters across clients
        clusters_per_client = n_clusters // n_clients
        remainder = n_clusters % n_clients

        cluster_idx = 0
        for client_id in range(n_clients):
            # Determine how many clusters this client gets
            n_client_clusters = clusters_per_client + (1 if client_id < remainder else 0)

            # Collect samples from assigned clusters
            client_mask = np.zeros(X.shape[0], dtype=bool)
            for _ in range(n_client_clusters):
                if cluster_idx < n_clusters:
                    client_mask |= (cluster_labels == cluster_idx)
                    cluster_idx += 1

            partitions[client_id] = (X[client_mask], y[client_mask])

    else:  # n_clusters > n_clients
        # Merge clusters: assign multiple clusters per client
        clusters_per_client = n_clusters // n_clients
        remainder = n_clusters % n_clients

        cluster_assignments = {}
        cluster_idx = 0
        for client_id in range(n_clients):
            n_client_clusters = clusters_per_client + (1 if client_id < remainder else 0)
            client_clusters = list(range(cluster_idx, cluster_idx + n_client_clusters))
            cluster_assignments[client_id] = client_clusters
            cluster_idx += n_client_clusters

        # Create client masks
        for client_id in range(n_clients):
            client_mask = np.zeros(X.shape[0], dtype=bool)
            for cluster_id in cluster_assignments[client_id]:
                client_mask |= (cluster_labels == cluster_id)
            partitions[client_id] = (X[client_mask], y[client_mask])

    return partitions


def feature_based_partition_indices(y: np.ndarray,
                                    X: np.ndarray,
                                    n_clients: int,
                                    n_clusters: int = None,
                                    use_minibatch: bool = False,
                                    random_state: int = None) -> Dict[int, np.ndarray]:
    """
    Partition indices based on feature space clustering.

    This is a convenience function that returns only the sample indices.

    Args:
        y: Label array of shape (n_samples,)
        X: Feature array of shape (n_samples, n_features)
        n_clients: Number of clients to partition data across
        n_clusters: Number of clusters (default: n_clients)
        use_minibatch: Use MiniBatchKMeans for large datasets
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to array of sample indices
    """
    if n_clusters is None:
        n_clusters = n_clients

    if n_clusters > X.shape[0]:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed n_samples ({X.shape[0]})"
        )

    rng = set_random_state(random_state)

    # Perform clustering
    if use_minibatch or X.shape[0] > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=1000
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    cluster_labels = kmeans.fit_predict(X)

    # Get sample indices for each client
    partitions = {}

    if n_clusters == n_clients:
        for client_id in range(n_clients):
            mask = cluster_labels == client_id
            partitions[client_id] = np.where(mask)[0]

    elif n_clusters < n_clients:
        clusters_per_client = n_clusters // n_clients
        remainder = n_clusters % n_clients

        cluster_idx = 0
        for client_id in range(n_clients):
            n_client_clusters = clusters_per_client + (1 if client_id < remainder else 0)

            client_indices = []
            for _ in range(n_client_clusters):
                if cluster_idx < n_clusters:
                    cluster_mask = cluster_labels == cluster_idx
                    client_indices.extend(np.where(cluster_mask)[0].tolist())
                    cluster_idx += 1

            partitions[client_id] = np.array(client_indices, dtype=int)

    else:  # n_clusters > n_clients
        clusters_per_client = n_clusters // n_clients
        remainder = n_clusters % n_clients

        cluster_assignments = {}
        cluster_idx = 0
        for client_id in range(n_clients):
            n_client_clusters = clusters_per_client + (1 if client_id < remainder else 0)
            client_clusters = list(range(cluster_idx, cluster_idx + n_client_clusters))
            cluster_assignments[client_id] = client_clusters
            cluster_idx += n_client_clusters

        for client_id in range(n_clients):
            client_indices = []
            for cluster_id in cluster_assignments[client_id]:
                cluster_mask = cluster_labels == cluster_id
                client_indices.extend(np.where(cluster_mask)[0].tolist())

            partitions[client_id] = np.array(client_indices, dtype=int)

    return partitions
