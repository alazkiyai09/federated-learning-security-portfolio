"""
Unit tests for FederatedClient.

Tests local training logic and weight update computation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.client import FederatedClient
from src.utils import serialize_weights
from src.models import SimpleCNN


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def create_dummy_data(num_samples=100, num_features=10):
    """Create dummy dataset for testing."""
    features = torch.randn(num_samples, num_features)
    targets = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(features, targets)
    return dataset


def test_client_initialization():
    """Test that client initializes correctly."""
    model = DummyModel()
    dataset = create_dummy_data()
    loader = DataLoader(dataset, batch_size=10)
    config = {
        'local_epochs': 3,
        'learning_rate': 0.01,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    client = FederatedClient(0, model, loader, config)

    assert client.client_id == 0
    assert client.local_epochs == 3
    assert client.num_samples == 100
    assert len(client.train_loader) == 10  # 100 samples / batch_size 10


def test_local_train_returns_weights():
    """Test that local_train returns updated weights."""
    model = DummyModel()
    dataset = create_dummy_data(num_samples=50)
    loader = DataLoader(dataset, batch_size=10)
    config = {
        'local_epochs': 2,
        'learning_rate': 0.1,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    client = FederatedClient(0, model, loader, config)

    # Get initial weights
    initial_weights = serialize_weights(model)

    # Perform local training
    updated_weights, num_samples = client.local_train(initial_weights)

    # Check return types
    assert isinstance(updated_weights, dict)
    assert len(updated_weights) == len(initial_weights)
    assert num_samples == 50

    # Check that weights changed (training should modify weights)
    weights_changed = False
    for key in initial_weights.keys():
        if not torch.allclose(initial_weights[key], updated_weights[key], rtol=1e-5):
            weights_changed = True
            break

    assert weights_changed, "Training should modify weights"


def test_local_train_uses_global_weights():
    """Test that local_train starts from provided global weights."""
    model1 = DummyModel()
    model2 = DummyModel()

    # Make models different
    with torch.no_grad():
        for param in model2.parameters():
            param.data += 1.0

    dataset = create_dummy_data(num_samples=50)
    loader = DataLoader(dataset, batch_size=10)
    config = {
        'local_epochs': 1,
        'learning_rate': 0.01,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    # Get weights from model1
    global_weights = serialize_weights(model1)

    # Train client with model2
    client = FederatedClient(0, model2, loader, config)
    local_weights, _ = client.local_train(global_weights)

    # Local weights should start from global weights and then change
    # They should NOT be equal to model2's original weights
    model2_original = serialize_weights(DummyModel())
    with torch.no_grad():
        for param in model2_original.values():
            param += 1.0

    starts_from_global = True
    for key in global_weights.keys():
        # Local weights should be closer to global than to original model2
        dist_to_global = (local_weights[key] - global_weights[key]).abs().sum()
        dist_to_original = (local_weights[key] - model2_original[key]).abs().sum()
        if dist_to_original < dist_to_global:
            starts_from_global = False
            break

    assert starts_from_global, "Local training should start from global weights"


def test_client_sample_count():
    """Test that client reports correct sample count."""
    model = DummyModel()

    for num_samples in [10, 50, 100, 250]:
        dataset = create_dummy_data(num_samples=num_samples)
        loader = DataLoader(dataset, batch_size=10)
        config = {
            'local_epochs': 1,
            'learning_rate': 0.01,
            'device': torch.device('cpu')
        }

        client = FederatedClient(0, model, loader, config)
        _, returned_samples = client.local_train(serialize_weights(model))

        assert client.num_samples == num_samples
        assert returned_samples == num_samples


def test_multiple_clients_different_data():
    """Test that multiple clients with different data produce different updates."""
    model1 = DummyModel()
    model2 = DummyModel()

    # Create different data for each client
    torch.manual_seed(42)
    dataset1 = create_dummy_data(num_samples=50)
    torch.manual_seed(43)
    dataset2 = create_dummy_data(num_samples=50)

    loader1 = DataLoader(dataset1, batch_size=10)
    loader2 = DataLoader(dataset2, batch_size=10)

    config = {
        'local_epochs': 3,
        'learning_rate': 0.1,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    client1 = FederatedClient(0, model1, loader1, config)
    client2 = FederatedClient(1, model2, loader2, config)

    # Shared global weights
    global_weights = serialize_weights(model1)

    # Train both clients
    weights1, _ = client1.local_train(global_weights)
    weights2, _ = client2.local_train(global_weights)

    # Weights should be different (different training data)
    weights_different = False
    for key in weights1.keys():
        if not torch.allclose(weights1[key], weights2[key], rtol=1e-4, atol=1e-5):
            weights_different = True
            break

    assert weights_different, "Different training data should produce different updates"


def test_local_epochs_parameter():
    """Test that local_epochs controls training duration."""
    model = DummyModel()
    dataset = create_dummy_data(num_samples=50)
    loader = DataLoader(dataset, batch_size=10)
    config = {
        'learning_rate': 0.1,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    global_weights = serialize_weights(model)

    for local_epochs in [1, 3, 5]:
        config['local_epochs'] = local_epochs
        client = FederatedClient(0, model, loader, config)

        # Reset model to global weights
        deserialize_weights(model, global_weights)

        # Train
        client.local_train(global_weights)

        # Check loss history length
        assert len(client.loss_history) == local_epochs


def test_evaluate():
    """Test client evaluation."""
    model = DummyModel()

    train_dataset = create_dummy_data(num_samples=100)
    test_dataset = create_dummy_data(num_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=10)
    test_loader = DataLoader(test_dataset, batch_size=10)

    config = {
        'local_epochs': 2,
        'learning_rate': 0.1,
        'device': torch.device('cpu')
    }

    client = FederatedClient(0, model, train_loader, config)

    # Train first
    global_weights = serialize_weights(model)
    client.local_train(global_weights)

    # Evaluate
    metrics = client.evaluate(test_loader)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'num_samples' in metrics
    assert metrics['num_samples'] == 50
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_client_with_cnn():
    """Test client with actual CNN model."""
    model = SimpleCNN(num_classes=10)

    # Create MNIST-like dummy data
    features = torch.randn(100, 1, 28, 28)
    targets = torch.randint(0, 10, (100,))
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=10)

    config = {
        'local_epochs': 2,
        'learning_rate': 0.01,
        'optimizer_type': 'sgd',
        'device': torch.device('cpu')
    }

    client = FederatedClient(0, model, loader, config)

    global_weights = serialize_weights(model)
    updated_weights, num_samples = client.local_train(global_weights)

    assert num_samples == 100
    assert len(updated_weights) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
