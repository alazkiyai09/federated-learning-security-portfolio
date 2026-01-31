"""
Main partitioner class orchestrating all partition strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from .strategies import (
    iid_partition,
    dirichlet_partition,
    quantity_skew_partition,
    feature_based_partition,
    realistic_bank_partition
)
from .utils import validate_data


class NonIIDPartitioner:
    """
    Main interface for partitioning data in federated learning scenarios.

    This class provides a unified API for multiple partition strategies that
    simulate different types of non-IID data distributions encountered in
    real-world federated learning deployments.

    Supported strategies:
    - IID: Random uniform distribution (baseline)
    - Label skew: Dirichlet-based class distribution
    - Quantity skew: Power law sample allocation
    - Feature skew: Clustering-based feature distribution
    - Realistic bank: Geography + demographic simulation

    Example usage:
        >>> partitioner = NonIIDPartitioner(n_clients=10, random_state=42)
        >>> partitions = partitioner.partition_label_skew(X, y, alpha=0.5)
        >>> for client_id, (X_client, y_client) in partitions.items():
        ...     print(f"Client {client_id}: {len(y_client)} samples")
    """

    def __init__(self, n_clients: int, random_state: Optional[int] = None):
        """
        Initialize the partitioner.

        Args:
            n_clients: Number of clients to partition data across
            random_state: Seed for reproducibility
        """
        if n_clients <= 0:
            raise ValueError(f"n_clients must be positive, got {n_clients}")

        self.n_clients = n_clients
        self.random_state = random_state
        self._partitions = None

    def partition_iid(self,
                     X: np.ndarray,
                     y: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data randomly and uniformly (IID baseline).

        This is the idealized scenario where data is independently and
        identically distributed across all clients.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)

        Returns:
            Dictionary mapping client_id to (X_client, y_client) tuples
        """
        validate_data(X, y)
        self._partitions = iid_partition(
            X, y, self.n_clients, self.random_state
        )
        return self._partitions

    def partition_label_skew(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            alpha: float = 1.0,
                            min_samples_per_client: int = 1) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with label skew using Dirichlet distribution.

        Lower alpha values create more extreme non-IID distributions:
        - alpha → 0: Each client has 1-2 dominant classes
        - alpha = 1: Moderate heterogeneity
        - alpha → ∞: Approaches IID

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            alpha: Dirichlet concentration parameter (lower = more skew)
            min_samples_per_client: Minimum samples each client receives

        Returns:
            Dictionary mapping client_id to (X_client, y_client) tuples
        """
        validate_data(X, y)
        self._partitions = dirichlet_partition(
            X, y, self.n_clients, alpha, min_samples_per_client, self.random_state
        )
        return self._partitions

    def partition_quantity_skew(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               exponent: float = 1.5,
                               min_samples_per_client: int = 1) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with quantity skew using power law distribution.

        This simulates scenarios where clients have vastly different
        amounts of data (e.g., varying user activity).

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            exponent: Power law exponent (1.1 = heavy skew, 3+ = mild skew)
            min_samples_per_client: Minimum samples each client receives

        Returns:
            Dictionary mapping client_id to (X_client, y_client) tuples
        """
        validate_data(X, y)
        self._partitions = quantity_skew_partition(
            X, y, self.n_clients, exponent, min_samples_per_client, self.random_state
        )
        return self._partitions

    def partition_feature_skew(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              n_clusters: Optional[int] = None,
                              use_minibatch: bool = False) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data based on feature space clustering.

        Clients receive data from different regions of the feature space,
        simulating geographic or demographic differences.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            n_clusters: Number of feature clusters (default: n_clients)
            use_minibatch: Use MiniBatchKMeans for large datasets

        Returns:
            Dictionary mapping client_id to (X_client, y_client) tuples
        """
        validate_data(X, y)
        self._partitions = feature_based_partition(
            X, y, self.n_clients, n_clusters, use_minibatch, self.random_state
        )
        return self._partitions

    def partition_realistic_bank(self,
                                df: pd.DataFrame,
                                region_col: Optional[str] = None,
                                label_col: str = 'label',
                                feature_cols: Optional[list] = None,
                                balance_within_regions: bool = True) -> Dict[int, pd.DataFrame]:
        """
        Partition bank transaction data with realistic geographic/demographic skew.

        This simulates real-world FL scenarios where banks in different regions
        have different customer demographics and fraud patterns.

        Args:
            df: DataFrame with transaction data
            region_col: Column name for region (if None, infers from features)
            label_col: Column name for fraud labels
            feature_cols: List of feature columns
            balance_within_regions: Balance fraud classes within each region

        Returns:
            Dictionary mapping client_id to DataFrame with client's data
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")

        self._partitions = realistic_bank_partition(
            df, self.n_clients, region_col, label_col,
            feature_cols, balance_within_regions, self.random_state
        )
        return self._partitions

    def partition_realistic_bank_arrays(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       region_labels: Optional[np.ndarray] = None,
                                       balance_within_regions: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition array data using realistic bank simulation.

        Convenience method for working with numpy arrays.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            region_labels: Optional region labels for each sample
            balance_within_regions: Balance classes within each region

        Returns:
            Dictionary mapping client_id to (X_client, y_client) tuples
        """
        from .strategies.realistic_bank import realistic_bank_partition_from_arrays

        validate_data(X, y)
        self._partitions = realistic_bank_partition_from_arrays(
            X, y, self.n_clients, region_labels, balance_within_regions, self.random_state
        )
        return self._partitions

    def get_partition_statistics(self) -> Dict[int, Dict[str, any]]:
        """
        Get statistics about the current partition.

        Returns:
            Dictionary with statistics for each client
            {'n_samples': int, 'label_distribution': dict}
        """
        if self._partitions is None:
            raise ValueError("No partition created yet. Call a partition_* method first.")

        stats = {}
        for client_id, data in self._partitions.items():
            if isinstance(data, tuple):
                # (X, y) format
                _, y = data
                unique, counts = np.unique(y, return_counts=True)
                label_dist = dict(zip(unique.tolist(), counts.tolist()))
                stats[client_id] = {
                    'n_samples': len(y),
                    'label_distribution': label_dist
                }
            else:
                # DataFrame format
                label_col = None
                for col in ['label', 'target', 'class', 'fraud']:
                    if col in data.columns:
                        label_col = col
                        break

                if label_col is None:
                    # Assume last column is label
                    label_col = data.columns[-1]

                unique = data[label_col].unique()
                label_dist = {label: (data[label_col] == label).sum()
                            for label in unique}
                stats[client_id] = {
                    'n_samples': len(data),
                    'label_distribution': label_dist
                }

        return stats

    def get_client_sizes(self) -> Dict[int, int]:
        """
        Get the number of samples for each client.

        Returns:
            Dictionary mapping client_id to sample count
        """
        if self._partitions is None:
            raise ValueError("No partition created yet. Call a partition_* method first.")

        sizes = {}
        for client_id, data in self._partitions.items():
            if isinstance(data, tuple):
                sizes[client_id] = len(data[1])
            else:
                sizes[client_id] = len(data)

        return sizes
