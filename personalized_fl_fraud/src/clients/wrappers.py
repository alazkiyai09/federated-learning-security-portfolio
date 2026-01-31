"""
Client Factory Wrappers

Factory functions for creating clients for each personalization method.
"""

from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig


def create_fedavg_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    device: str = "cpu"
):
    """
    Create standard FedAvg client.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration
        device: Device to use

    Returns:
        FedAvgClient instance
    """
    from .personalized_client import FedAvgClient
    return FedAvgClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )


def create_fedper_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    method_config: DictConfig,
    device: str = "cpu"
):
    """
    Create FedPer client with personalized layers.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Global configuration
        method_config: FedPer-specific configuration
        device: Device to use

    Returns:
        FedPerClient instance
    """
    from .personalized_client import FedPerClient
    return FedPerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        method_config=method_config,
        device=device
    )


def create_ditto_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    method_config: DictConfig,
    device: str = "cpu"
):
    """
    Create Ditto client with local + global models.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Global configuration
        method_config: Ditto-specific configuration
        device: Device to use

    Returns:
        DittoClient instance
    """
    from .personalized_client import DittoClient
    return DittoClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        method_config=method_config,
        device=device
    )


def create_per_fedavg_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    method_config: DictConfig,
    device: str = "cpu"
):
    """
    Create Per-FedAvg client with meta-learning.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Global configuration
        method_config: Per-FedAvg-specific configuration
        device: Device to use

    Returns:
        PerFedAvgClient instance
    """
    from .personalized_client import PerFedAvgClient
    return PerFedAvgClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        method_config=method_config,
        device=device
    )


# Client factory mapping
CLIENT_FACTORIES = {
    'fedavg': create_fedavg_client,
    'local_finetuning': create_fedavg_client,  # Same client, fine-tuning is post-hoc
    'fedper': create_fedper_client,
    'ditto': create_ditto_client,
    'per_fedavg': create_per_fedavg_client,
}


def create_client(
    method_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    device: str = "cpu"
):
    """
    Generic client factory.

    Args:
        method_name: Name of personalization method
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration
        device: Device to use

    Returns:
        Client instance
    """
    if method_name not in CLIENT_FACTORIES:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(CLIENT_FACTORIES.keys())}"
        )

    factory = CLIENT_FACTORIES[method_name]

    # Get method-specific config
    method_config = config.get('methods', {}).get(method_name, {})

    return factory(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        method_config=method_config,
        device=device
    )
