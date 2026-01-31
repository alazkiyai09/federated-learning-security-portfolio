"""
Data Partitioning Wrapper for Personalized FL Experiments

Provides wrapper around Day 9 NonIIDPartitioner for creating
non-IID data partitions across multiple alpha levels.
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
import json

import numpy as np
from sklearn.model_selection import train_test_split


# Add parent project to path to import Day 9 partitioner
import sys
import importlib.util

_non_iid_src = Path(__file__).parent.parent.parent.parent / "non_iid_partitioner" / "src"

# Try to import NonIIDPartitioner from Day 9 project
NonIIDPartitioner = None
_partitioner_import_error = None

try:
    # Add the non_iid_partitioner parent directory to path
    _non_iid_parent = str(_non_iid_src.parent)
    if _non_iid_parent not in sys.path:
        sys.path.insert(0, _non_iid_parent)

    # Import as module
    from non_iid_partitioner.src.partitioner import NonIIDPartitioner as _NonIIDPartitioner
    NonIIDPartitioner = _NonIIDPartitioner
except (ImportError, ModuleNotFoundError) as e:
    _partitioner_import_error = str(e)
    # Create a placeholder that will raise a helpful error when used
    class _NonIIDPartitionerPlaceholder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"Could not import NonIIDPartitioner from Day 9 project.\n"
                f"Error: {_partitioner_import_error}\n\n"
                f"Please ensure the non_iid_partitioner project is at:\n"
                f"  {_non_iid_src.parent}\n\n"
                f"Or install it: pip install -e {_non_iid_src.parent}"
            )
    NonIIDPartitioner = _NonIIDPartitionerPlaceholder


class DataPartitioner:
    """
    Wrapper for NonIIDPartitioner with caching and train/val/test split support.

    This class handles:
    1. Partitioning data with various non-IID strategies (label skew, etc.)
    2. Splitting partitions into train/val/test sets
    3. Caching partitions to disk for reproducibility
    4. Loading pre-computed partitions

    Example usage:
        >>> partitioner = DataPartitioner(
        ...     n_clients=10,
        ...     test_size=0.2,
        ...     val_size=0.1,
        ...     cache_dir="./data/processed"
        ... )
        >>> partitions = partitioner.create_partitions(
        ...     X, y, alpha=0.5, strategy="label_skew"
        ... )
    """

    def __init__(
        self,
        n_clients: int,
        test_size: float = 0.2,
        val_size: float = 0.1,
        cache_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize data partitioner.

        Args:
            n_clients: Number of clients/banks
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation
            cache_dir: Directory to cache partitions (None = no caching)
            random_state: Random seed for reproducibility
        """
        self.n_clients = n_clients
        self.test_size = test_size
        self.val_size = val_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.random_state = random_state

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Day 9 partitioner
        self.base_partitioner = NonIIDPartitioner(
            n_clients=n_clients,
            random_state=random_state
        )

    def create_partitions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        strategy: str = "label_skew",
        use_cache: bool = True,
        **kwargs
    ) -> Dict[int, Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Create train/val/test partitions for all clients.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            alpha: Dirichlet concentration parameter (for label skew)
            strategy: Partition strategy ('label_skew', 'iid', etc.)
            use_cache: Whether to use cached partitions if available
            **kwargs: Additional arguments passed to partition strategy

        Returns:
            Dictionary mapping client_id to {
                'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
            }
        """
        # Check cache first
        cache_key = self._get_cache_key(X.shape, alpha, strategy)
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # First split off test set (global test set)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Partition training+val data among clients
        if strategy == "label_skew":
            client_partitions = self.base_partitioner.partition_label_skew(
                X_train_val, y_train_val,
                alpha=alpha,
                min_samples_per_client=kwargs.get('min_samples_per_client', 50)
            )
        elif strategy == "iid":
            client_partitions = self.base_partitioner.partition_iid(
                X_train_val, y_train_val
            )
        elif strategy == "quantity_skew":
            client_partitions = self.base_partitioner.partition_quantity_skew(
                X_train_val, y_train_val,
                exponent=kwargs.get('exponent', 1.5),
                min_samples_per_client=kwargs.get('min_samples_per_client', 50)
            )
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

        # Split each client's data into train/val
        final_partitions = {}
        for client_id, (X_client, y_client) in client_partitions.items():
            # Skip clients with too few samples
            if len(y_client) < 10:
                continue

            # Split client data into train/val
            # Adjust val_size to account for already-split test set
            adjusted_val_size = self.val_size / (1 - self.test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_client, y_client,
                test_size=adjusted_val_size,
                random_state=self.random_state + client_id,
                stratify=y_client
            )

            final_partitions[client_id] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,  # Global test set for all clients
                'y_test': y_test
            }

        # Cache results
        if self.cache_dir:
            self._save_to_cache(cache_key, final_partitions)

        return final_partitions

    def create_partitions_at_multiple_alphas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha_values: list,
        strategy: str = "label_skew",
        use_cache: bool = True,
        **kwargs
    ) -> Dict[float, Dict[int, Dict[str, np.ndarray]]]:
        """
        Create partitions for multiple alpha values.

        Args:
            X: Feature array
            y: Label array
            alpha_values: List of alpha values to try
            strategy: Partition strategy
            use_cache: Whether to use cache
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping alpha -> partitions dict
        """
        all_partitions = {}

        for alpha in alpha_values:
            partitions = self.create_partitions(
                X, y, alpha=alpha, strategy=strategy,
                use_cache=use_cache, **kwargs
            )
            all_partitions[alpha] = partitions

        return all_partitions

    def get_partition_statistics(
        self,
        partitions: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, any]]:
        """
        Compute statistics for each client's partition.

        Args:
            partitions: Partition dictionary from create_partitions()

        Returns:
            Dictionary mapping client_id to statistics
        """
        stats = {}

        for client_id, data in partitions.items():
            y_train = data['y_train']
            y_val = data['y_val']

            # Count fraud cases
            n_train_fraud = (y_train == 1).sum()
            n_train_legit = (y_train == 0).sum()
            n_val_fraud = (y_val == 1).sum()
            n_val_legit = (y_val == 0).sum()

            stats[client_id] = {
                'n_train_samples': len(y_train),
                'n_val_samples': len(y_val),
                'n_test_samples': len(data['y_test']),
                'train_fraud_ratio': n_train_fraud / len(y_train) if len(y_train) > 0 else 0,
                'val_fraud_ratio': n_val_fraud / len(y_val) if len(y_val) > 0 else 0,
                'n_train_fraud': n_train_fraud,
                'n_train_legit': n_train_legit,
                'n_val_fraud': n_val_fraud,
                'n_val_legit': n_val_legit,
            }

        return stats

    def _get_cache_key(
        self,
        data_shape: Tuple[int, int],
        alpha: float,
        strategy: str
    ) -> str:
        """Generate cache key for partition."""
        return f"partitions_n{self.n_clients}_a{alpha}_{strategy}_{data_shape}.pkl"

    def _load_from_cache(
        self,
        cache_key: str
    ) -> Optional[Dict]:
        """Load partitions from cache if available."""
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def _save_to_cache(
        self,
        cache_key: str,
        partitions: Dict
    ) -> None:
        """Save partitions to cache."""
        cache_path = self.cache_dir / cache_key

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(partitions, f)
        except Exception as e:
            print(f"Warning: Failed to cache partitions: {e}")

    def save_statistics(
        self,
        partitions: Dict[int, Dict[str, np.ndarray]],
        output_path: str
    ) -> None:
        """
        Save partition statistics to JSON file.

        Args:
            partitions: Partition dictionary
            output_path: Path to save statistics
        """
        stats = self.get_partition_statistics(partitions)

        # Convert numpy types to Python types for JSON serialization
        stats_serializable = {}
        for client_id, client_stats in stats.items():
            stats_serializable[str(client_id)] = {
                k: int(v) if isinstance(v, (np.integer, int)) else float(v)
                for k, v in client_stats.items()
            }

        with open(output_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)


def create_synthetic_fraud_data(
    n_samples: int = 10000,
    n_features: int = 20,
    fraud_ratio: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection data for testing.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        fraud_ratio: Fraction of fraudulent transactions
        random_state: Random seed

    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    # Generate features
    X = rng.randn(n_samples, n_features)

    # Generate labels with some structure
    # Use first 5 features to determine fraud probability
    fraud_score = (
        0.3 * X[:, 0] +
        0.2 * X[:, 1] +
        0.2 * X[:, 2] +
        0.15 * X[:, 3] +
        0.15 * X[:, 4]
    )

    # Convert to probability using sigmoid
    fraud_prob = 1 / (1 + np.exp(-fraud_score))

    # Adjust to match target fraud ratio
    threshold = np.percentile(fraud_prob, 100 * (1 - fraud_ratio))
    y = (fraud_prob > threshold).astype(int)

    return X, y
