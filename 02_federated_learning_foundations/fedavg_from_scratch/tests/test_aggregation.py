"""
Unit tests for FedAvg weighted averaging.

Tests the mathematical correctness of aggregation.
"""

import pytest
import torch
import torch.nn as nn

from src.server import FederatedServer
from src.utils import serialize_weights
from src.models import SimpleCNN


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def create_model_with_weights(value_offset: float) -> nn.Module:
    """Create model with weights offset by given value."""
    model = DummyModel()
    with torch.no_grad():
        for param in model.parameters():
            param.data += value_offset
    return model


def test_aggregate_weights_basic():
    """Test basic weighted averaging with 2 clients."""
    model1 = create_model_with_weights(0.0)
    model2 = create_model_with_weights(1.0)

    weights1 = serialize_weights(model1)
    weights2 = serialize_weights(model2)

    # Both clients have same number of samples
    client_updates = [(weights1, 100), (weights2, 100)]

    config = {'num_rounds': 10}
    server = FederatedServer(model1, config)
    aggregated = server.aggregate_weights(client_updates)

    # With equal weights, should be average of the two
    for key in aggregated.keys():
        expected = (weights1[key] + weights2[key]) / 2
        assert torch.allclose(aggregated[key], expected, atol=1e-6), \
            f"Aggregation incorrect for {key}"


def test_aggregate_weights_unequal_samples():
    """Test weighted averaging with different sample counts."""
    model1 = create_model_with_weights(0.0)
    model2 = create_model_with_weights(10.0)

    weights1 = serialize_weights(model1)
    weights2 = serialize_weights(model2)

    # Client 1 has 3x more samples
    client_updates = [(weights1, 300), (weights2, 100)]

    config = {'num_rounds': 10}
    server = FederatedServer(model1, config)
    aggregated = server.aggregate_weights(client_updates)

    # Weighted average: (3*0 + 1*10) / 4 = 2.5
    # So aggregated should be closer to weights1
    for key in aggregated.keys():
        expected = (0.75 * weights1[key]) + (0.25 * weights2[key])
        assert torch.allclose(aggregated[key], expected, atol=1e-6), \
            f"Weighted aggregation incorrect for {key}"


def test_aggregate_single_client():
    """Test aggregation with single client returns client weights."""
    model = create_model_with_weights(5.0)
    weights = serialize_weights(model)

    client_updates = [(weights, 100)]

    config = {'num_rounds': 10}
    server = FederatedServer(model, config)
    aggregated = server.aggregate_weights(client_updates)

    # Should be exactly the client's weights
    for key in aggregated.keys():
        assert torch.equal(aggregated[key], weights[key]), \
            f"Single client aggregation failed for {key}"


def test_aggregate_multiple_clients():
    """Test aggregation with multiple clients (3-5)."""
    num_clients = 5
    models = [create_model_with_weights(float(i)) for i in range(num_clients)]

    # Create sample counts (varying sizes)
    sample_counts = [50, 100, 150, 200, 250]

    client_updates = [
        (serialize_weights(model), samples)
        for model, samples in zip(models, sample_counts)
    ]

    total_samples = sum(sample_counts)

    config = {'num_rounds': 10}
    server = FederatedServer(models[0], config)
    aggregated = server.aggregate_weights(client_updates)

    # Verify weighted average
    for key in aggregated.keys():
        expected = torch.zeros_like(aggregated[key])
        for i, (weights, samples) in enumerate(client_updates):
            expected += (samples / total_samples) * weights

        assert torch.allclose(aggregated[key], expected, atol=1e-6), \
            f"Multi-client aggregation incorrect for {key}"


def test_aggregate_preserves_parameter_shapes():
    """Test that aggregation preserves parameter shapes."""
    model = SimpleCNN(num_classes=10)

    # Create different weight sets
    models = [SimpleCNN(num_classes=10) for _ in range(3)]
    weights_list = [serialize_weights(m) for m in models]

    sample_counts = [100, 150, 200]
    client_updates = list(zip(weights_list, sample_counts))

    config = {'num_rounds': 10}
    server = FederatedServer(model, config)
    aggregated = server.aggregate_weights(client_updates)

    # Check shapes match
    original_shapes = {k: v.shape for k, v in weights_list[0].items()}
    aggregated_shapes = {k: v.shape for k, v in aggregated.items()}

    assert set(original_shapes.keys()) == set(aggregated_shapes.keys())

    for key in original_shapes.keys():
        assert original_shapes[key] == aggregated_shapes[key], \
            f"Shape mismatch for {key}"


def test_aggregate_empty_client_list():
    """Test that empty client list returns current weights."""
    model = create_model_with_weights(3.0)
    original_weights = serialize_weights(model)

    client_updates = []

    config = {'num_rounds': 10}
    server = FederatedServer(model, config)
    aggregated = server.aggregate_weights(client_updates)

    # Should return unchanged weights
    for key in aggregated.keys():
        assert torch.equal(aggregated[key], original_weights[key]), \
            f"Empty client list should return original weights for {key}"


def test_aggregate_numerical_stability():
    """Test that aggregation is numerically stable with many clients."""
    num_clients = 20
    models = [create_model_with_weights(float(i)) for i in range(num_clients)]

    # Large sample counts
    sample_counts = [10000 + i * 1000 for i in range(num_clients)]

    client_updates = [
        (serialize_weights(model), samples)
        for model, samples in zip(models, sample_counts)
    ]

    config = {'num_rounds': 10}
    server = FederatedServer(models[0], config)
    aggregated = server.aggregate_weights(client_updates)

    # Verify no NaN or Inf
    for key, tensor in aggregated.items():
        assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"


def test_aggregate_with_cnn():
    """Test aggregation works with CNN model."""
    models = [SimpleCNN(num_classes=10) for _ in range(3)]

    # Modify weights to make them different
    for i, model in enumerate(models):
        with torch.no_grad():
            for param in model.parameters():
                param.data += float(i)

    weights_list = [serialize_weights(m) for m in models]
    sample_counts = [100, 200, 150]
    client_updates = list(zip(weights_list, sample_counts))

    config = {'num_rounds': 10}
    server = FederatedServer(models[0], config)
    aggregated = server.aggregate_weights(client_updates)

    # Verify all parameters are aggregated
    assert len(aggregated) > 0

    # Check that aggregation changed values
    for key in aggregated.keys():
        # Should be different from any single client
        is_different_from_all = True
        for weights in weights_list:
            if torch.allclose(aggregated[key], weights[key], rtol=1e-2):
                is_different_from_all = False
                break
        assert is_different_from_all, \
            f"Aggregated weights should differ from individual clients for {key}"


def test_weighted_average_math():
    """Test the mathematical formula of weighted averaging."""
    # Create simple 1D tensors for easy verification
    weights1 = {'param': torch.tensor([1.0, 2.0, 3.0])}
    weights2 = {'param': torch.tensor([4.0, 5.0, 6.0])}
    weights3 = {'param': torch.tensor([7.0, 8.0, 9.0])}

    # Sample counts: 10, 20, 30 (total = 60)
    # Expected: (10*1 + 20*4 + 30*7) / 60 = (10 + 80 + 210) / 60 = 300/60 = 5
    #           (10*2 + 20*5 + 30*8) / 60 = (20 + 100 + 240) / 60 = 360/60 = 6
    #           (10*3 + 20*6 + 30*9) / 60 = (30 + 120 + 270) / 60 = 420/60 = 7

    client_updates = [
        (weights1, 10),
        (weights2, 20),
        (weights3, 30)
    ]

    model = DummyModel()
    config = {'num_rounds': 10}
    server = FederatedServer(model, config)
    aggregated = server.aggregate_weights(client_updates)

    expected = torch.tensor([5.0, 6.0, 7.0])
    assert torch.allclose(aggregated['param'], expected, atol=1e-6), \
        "Weighted average formula incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
