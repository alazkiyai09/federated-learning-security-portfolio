"""
Data loading and partitioning for federated learning.

Supports:
- MNIST for sanity checks
- Fraud detection data (Credit Card Fraud Detection)
- Non-IID and IID data partitioning
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
import numpy as np


def partition_data(
    dataset: Dataset,
    num_clients: int,
    batch_size: int = 32,
    distribution: str = 'iid',
    alpha: float = 0.5,
    seed: int = 42
) -> Tuple[List[DataLoader], List[int]]:
    """
    Partition dataset among clients for federated learning.

    Args:
        dataset: PyTorch Dataset to partition
        num_clients: Number of clients to split data across
        batch_size: Batch size for DataLoaders
        distribution: 'iid' for independent partitions, 'non-iid' for Dirichlet
        alpha: Concentration parameter for Dirichlet distribution (lower = more skewed)
        seed: Random seed for reproducibility

    Returns:
        Tuple[List[DataLoader], List[int]]:
            - List of DataLoader objects, one per client
            - List of sample counts per client
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(dataset)
    indices = np.arange(num_samples)

    if distribution == 'iid':
        # Random IID partition
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)

    elif distribution == 'non-iid':
        # Non-IID partition using Dirichlet distribution over labels
        client_indices = _non_iid_partition(dataset, num_clients, alpha, seed)

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    # Create DataLoaders
    client_loaders = []
    sample_counts = []

    for idx_list in client_indices:
        if len(idx_list) == 0:
            # Handle empty partition (rare edge case)
            subset = Subset(dataset, [0])  # Dummy single sample
            loader = DataLoader(subset, batch_size=1, shuffle=True)
            sample_counts.append(0)
        else:
            subset = Subset(dataset, idx_list)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            sample_counts.append(len(idx_list))

        client_loaders.append(loader)

    return client_loaders, sample_counts


def _non_iid_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int
) -> List[np.ndarray]:
    """
    Create non-IID partition using Dirichlet distribution over labels.

    Implementation based on: "Federated Learning with Matched Averaging"
    Uses Dirichlet distribution to sample label proportions for each client.

    Args:
        dataset: Dataset with targets attribute
        num_clients: Number of clients
        alpha: Dirichlet concentration (lower = more skewed)
        seed: Random seed

    Returns:
        List[np.ndarray]: List of index arrays for each client
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # If no labels attribute, need to extract from dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    num_samples = len(labels)

    # Initialize client index lists
    client_indices = [[] for _ in range(num_clients)]

    # Partition each class according to Dirichlet
    for k in range(num_classes):
        # Get indices of samples with this class
        idx_k = np.where(labels == k)[0]

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Calculate number of samples per client for this class
        num_samples_k = len(idx_k)
        counts = (proportions * num_samples_k).astype(int)

        # Distribute remaining samples (due to rounding)
        remainder = num_samples_k - counts.sum()
        if remainder > 0:
            # Assign remainder to clients with highest proportions
            remainder_clients = np.argsort(-proportions)[:remainder]
            counts[remainder_clients] += 1

        # Shuffle indices for this class
        np.random.shuffle(idx_k)

        # Assign to clients
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(idx_k[start:end].tolist())
            start = end

    # Convert to numpy arrays
    client_indices = [np.array(idx, dtype=np.int64) for idx in client_indices]

    # Shuffle each client's indices
    for idx in client_indices:
        np.random.shuffle(idx)

    return client_indices


def load_mnist(
    data_dir: str = './data',
    batch_size: int = 32
) -> Tuple[Dataset, Dataset]:
    """
    Load MNIST train and test datasets.

    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size (unused but kept for consistency)

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def get_client_sample_counts(loaders: List[DataLoader]) -> List[int]:
    """
    Get number of samples for each client.

    Args:
        loaders: List of client DataLoaders

    Returns:
        List[int]: Sample count per client
    """
    return [len(loader.dataset) for loader in loaders]


def create_test_loader(
    test_dataset: Dataset,
    batch_size: int = 100
) -> DataLoader:
    """
    Create DataLoader for test set.

    Args:
        test_dataset: Test dataset
        batch_size: Batch size

    Returns:
        DataLoader: Test data loader
    """
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FraudDataset(Dataset):
    """
    Dataset for credit card fraud detection.

    Expects CSV file with features and target column.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True
    ):
        """
        Args:
            features: Feature matrix of shape (num_samples, num_features)
            labels: Target labels (0 for legitimate, 1 for fraud)
            normalize: Whether to normalize features to zero mean, unit variance
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

        if normalize:
            mean = self.features.mean(dim=0)
            std = self.features.std(dim=0) + 1e-8  # Avoid division by zero
            self.features = (self.features - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_fraud_data(
    data_path: str,
    train_ratio: float = 0.8
) -> Tuple[FraudDataset, FraudDataset]:
    """
    Load and split fraud detection dataset.

    Args:
        data_path: Path to CSV file (must have 'Class' column as target)
        train_ratio: Ratio of training data

    Returns:
        Tuple[FraudDataset, FraudDataset]: (train_dataset, test_dataset)
    """
    import pandas as pd

    df = pd.read_csv(data_path)

    # Separate features and target (assuming 'Class' column is target)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must have 'Class' column as target")

    features = df.drop('Class', axis=1).values
    labels = df['Class'].values

    # Shuffle
    num_samples = len(features)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    features = features[indices]
    labels = labels[indices]

    # Split
    split_idx = int(num_samples * train_ratio)

    train_dataset = FraudDataset(features[:split_idx], labels[:split_idx])
    test_dataset = FraudDataset(features[split_idx:], labels[split_idx:], normalize=False)

    # Normalize test set using training statistics
    train_mean = train_dataset.features.mean(dim=0)
    train_std = train_dataset.features.std(dim=0) + 1e-8
    test_dataset.features = (test_dataset.features - train_mean) / train_std

    return train_dataset, test_dataset
