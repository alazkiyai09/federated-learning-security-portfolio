"""
Data Loading and Partitioning for Fraud Detection

Integrates with the Non-IID partitioner from Day 9 to create
client-specific data loaders for federated learning.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_synthetic_fraud_data(
    n_samples: int = 10000,
    n_features: int = 30,
    fraud_ratio: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection dataset.

    Creates a tabular dataset with imbalanced binary labels.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        fraud_ratio: Proportion of positive (fraud) samples
        seed: Random seed

    Returns:
        (X, y) tuple where X is (n_samples, n_features) and y is (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Generate features
    X = rng.randn(n_samples, n_features)

    # Generate imbalanced labels
    n_fraud = int(n_samples * fraud_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    fraud_indices = rng.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1

    # Make fraud samples have different feature distribution
    X[fraud_indices] += rng.randn(n_fraud, n_features) * 0.5

    return X, y


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X: Feature array
        y: Label array
        batch_size: Batch size for DataLoader
        val_split: Fraction of data to use for validation
        seed: Random seed for splitting

    Returns:
        (train_loader, val_loader) tuple
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def partition_data_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    val_split: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    Partition data IID across clients.

    Args:
        X: Feature array
        y: Label array
        num_clients: Number of clients
        val_split: Fraction of data to use for validation
        batch_size: Batch size for DataLoaders
        seed: Random seed

    Returns:
        Dictionary mapping client_id to (train_loader, val_loader)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    client_indices = np.array_split(indices, num_clients)

    client_loaders = {}
    for client_id, idx in enumerate(client_indices):
        X_client, y_client = X[idx], y[idx]
        train_loader, val_loader = create_data_loaders(
            X_client, y_client, batch_size, val_split, seed
        )
        client_loaders[client_id] = (train_loader, val_loader)

    return client_loaders


def partition_data_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    val_split: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    Partition data Non-IID using Dirichlet distribution.

    This creates label distribution skew where each client has
    a different proportion of positive (fraud) samples.

    Args:
        X: Feature array
        y: Label array
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more skewed)
        val_split: Fraction of data to use for validation
        batch_size: Batch size for DataLoaders
        seed: Random seed

    Returns:
        Dictionary mapping client_id to (train_loader, val_loader)
    """
    from scipy.stats import dirichlet

    rng = np.random.RandomState(seed)

    # Get indices for each class
    fraud_indices = np.where(y == 1)[0]
    legit_indices = np.where(y == 0)[0]

    # Sample Dirichlet distribution for class proportions
    # Each client gets a different proportion of fraud vs legit samples
    proportions = dirichlet.rvs(alpha=[alpha, alpha], size=num_clients, random_state=seed)

    client_loaders = {}
    for client_id in range(num_clients):
        # Determine how many fraud and legit samples this client gets
        n_fraud = int(len(fraud_indices) * proportions[client_id][0])
        n_legit = int(len(legit_indices) * proportions[client_id][1])

        # Sample without replacement
        fraud_idx = rng.choice(fraud_indices, n_fraud, replace=False)
        legit_idx = rng.choice(legit_indices, n_legit, replace=False)

        # Remove used indices from pool
        fraud_indices = np.setdiff1d(fraud_indices, fraud_idx)
        legit_indices = np.setdiff1d(legit_indices, legit_idx)

        # Combine indices
        client_idx = np.concatenate([fraud_idx, legit_idx])
        X_client, y_client = X[client_idx], y[client_idx]

        train_loader, val_loader = create_data_loaders(
            X_client, y_client, batch_size, val_split, seed
        )
        client_loaders[client_id] = (train_loader, val_loader)

    return client_loaders


def prepare_federated_data(
    cfg: DictConfig,
) -> Tuple[Dict[int, Tuple[DataLoader, DataLoader]], int]:
    """
    Prepare federated data based on configuration.

    Main entry point for data preparation in federated learning.

    Args:
        cfg: Hydra configuration object

    Returns:
        (client_loaders, input_dim) tuple where:
        - client_loaders: dict mapping client_id to (train_loader, val_loader)
        - input_dim: number of input features
    """
    # Load data
    X, y = load_synthetic_fraud_data(
        n_samples=10000,
        n_features=cfg.input_dim,
        fraud_ratio=cfg.data.fraud_ratio,
        seed=cfg.seed,
    )

    # Partition based on config
    partition_type = cfg.data.partition_type

    if partition_type == "iid":
        client_loaders = partition_data_iid(
            X=X,
            y=y,
            num_clients=cfg.num_clients,
            val_split=cfg.data.val_split,
            batch_size=cfg.client_batch_size,
            seed=cfg.seed,
        )
    elif partition_type == "non_iid":
        partition_strategy = cfg.data.get("partition_strategy", "dirichlet")
        alpha = cfg.data.get("alpha", 0.5)

        if partition_strategy == "dirichlet":
            client_loaders = partition_data_dirichlet(
                X=X,
                y=y,
                num_clients=cfg.num_clients,
                alpha=alpha,
                val_split=cfg.data.val_split,
                batch_size=cfg.client_batch_size,
                seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {partition_strategy}")
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    return client_loaders, X.shape[1]
