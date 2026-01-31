"""
Unit Tests for Flower Client

Tests client parameter handling, training, and evaluation logic.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client import FlClient
from src.model import FraudDetectionModel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create simple synthetic data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100, 1)).float()
    X_test = torch.randn(20, 10)
    y_test = torch.randint(0, 2, (20, 1)).float()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=10)
    test_loader = DataLoader(test_dataset, batch_size=10)

    return train_loader, test_loader


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return FraudDetectionModel(input_dim=10, hidden_dims=[8, 4])


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "local_epochs": 2,
        "learning_rate": 0.01,
        "optimizer": "adam",
    })


def test_client_initialization(sample_model, sample_data, sample_config):
    """Test client initialization."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    assert client.model is not None
    assert client.train_loader is train_loader
    assert client.test_loader is test_loader
    assert client.device == "cpu"


def test_get_parameters(sample_model, sample_data, sample_config):
    """Test getting model parameters."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    params = client.get_parameters(config={})

    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)


def test_set_parameters(sample_model, sample_data, sample_config):
    """Test setting model parameters."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    # Get original parameters
    original_params = client.get_parameters(config={})

    # Set modified parameters
    modified_params = [p + 0.1 for p in original_params]
    client.set_parameters(modified_params)

    # Get new parameters and verify they changed
    new_params = client.get_parameters(config={})

    for orig, new in zip(original_params, new_params):
        assert not np.allclose(orig, new)


def test_fit(sample_model, sample_data, sample_config):
    """Test client training (fit)."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    # Get initial parameters
    initial_params = client.get_parameters(config={})

    # Train for one round
    config = {"local_epochs": 1, "proximal_mu": 0.0}
    updated_params, num_samples, metrics = client.fit(initial_params, config)

    # Verify output
    assert isinstance(updated_params, list)
    assert all(isinstance(p, np.ndarray) for p in updated_params)
    assert num_samples > 0
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert metrics["loss"] >= 0


def test_fit_with_proximal(sample_model, sample_data, sample_config):
    """Test client training with proximal term (FedProx)."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    # Get initial parameters
    initial_params = client.get_parameters(config={})

    # Train with proximal term
    config = {"local_epochs": 1, "proximal_mu": 0.1}
    updated_params, num_samples, metrics = client.fit(initial_params, config)

    # Verify output
    assert isinstance(updated_params, list)
    assert num_samples > 0
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_evaluate(sample_model, sample_data, sample_config):
    """Test client evaluation."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    # Get initial parameters
    initial_params = client.get_parameters(config={})

    # Evaluate
    loss, num_samples, metrics = client.evaluate(initial_params, config={})

    # Verify output
    assert isinstance(loss, float)
    assert loss >= 0
    assert num_samples > 0
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_parameter_roundtrip(sample_model, sample_data, sample_config):
    """Test parameter serialization/deserialization roundtrip."""
    train_loader, test_loader = sample_data
    client = FlClient(
        model=sample_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=sample_config,
        device="cpu",
    )

    # Get original parameters
    original_params = client.get_parameters(config={})

    # Set parameters back
    client.set_parameters(original_params)

    # Get parameters again
    new_params = client.get_parameters(config={})

    # Verify they are the same
    for orig, new in zip(original_params, new_params):
        assert np.allclose(orig, new)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
