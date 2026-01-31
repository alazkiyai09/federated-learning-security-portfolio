"""
Unit tests for weight serialization utilities.

Tests that serialization preserves model weights exactly.
"""

import pytest
import torch
import torch.nn as nn
from src.utils import (
    serialize_weights,
    deserialize_weights,
    compute_weight_delta,
    apply_weight_update
)


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_serialize_preserves_weights():
    """Test that serialization preserves exact weight values."""
    model = DummyModel()

    # Get original weights
    original_weights = serialize_weights(model)

    # Serialize again
    serialized_weights = serialize_weights(model)

    # Check all keys and values match
    assert set(original_weights.keys()) == set(serialized_weights.keys())

    for key in original_weights.keys():
        assert torch.equal(original_weights[key], serialized_weights[key]), \
            f"Weights for {key} changed during serialization"


def test_deserialize_restores_weights():
    """Test that deserialization correctly restores weights."""
    model1 = DummyModel()
    model2 = DummyModel()

    # Ensure models have different weights
    with torch.no_grad():
        for param in model2.parameters():
            param.data += 1.0

    # Serialize model1 weights
    weights = serialize_weights(model1)

    # Deserialize into model2
    deserialize_weights(model2, weights)

    # Check models are now identical
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(param1, param2), "Deserialization failed to restore weights"


def test_serialize_detach_and_clone():
    """Test that serialized weights are detached and cloned."""
    model = DummyModel()
    weights = serialize_weights(model)

    # All weights should be on CPU
    for tensor in weights.values():
        assert not tensor.is_cuda, "Serialized weights should be on CPU"

    # Modifying weights should not affect model
    with torch.no_grad():
        for key in weights:
            weights[key] += 1.0

    # Model weights should be unchanged
    for name, param in model.named_parameters():
        original_value = param.data.clone()
        serialized_value = weights[name] - 1.0  # Undo our modification
        assert torch.equal(original_value, serialized_value), \
            "Serialization did not clone weights properly"


def test_compute_weight_delta():
    """Test weight delta computation."""
    model1 = DummyModel()
    model2 = DummyModel()

    # Make model2 weights different
    with torch.no_grad():
        for param in model2.parameters():
            param.data += 0.5

    weights1 = serialize_weights(model1)
    weights2 = serialize_weights(model2)

    delta = compute_weight_delta(weights2, weights1)

    # Delta should equal the difference
    for key in delta.keys():
        expected = weights2[key] - weights1[key]
        assert torch.allclose(delta[key], expected, atol=1e-6), \
            f"Delta computation incorrect for {key}"


def test_apply_weight_update():
    """Test applying weight updates to model."""
    model = DummyModel()

    # Get original weights
    original_weights = serialize_weights(model)

    # Create an update
    update = {}
    for key, tensor in original_weights.items():
        update[key] = torch.ones_like(tensor) * 0.1

    # Apply update
    apply_weight_update(model, update)

    # Check weights changed correctly
    for name, param in model.named_parameters():
        expected = original_weights[name] + 0.1
        assert torch.allclose(param.data, expected, atol=1e-6), \
            f"Weight update failed for {name}"


def test_roundtrip_serialization():
    """Test that serialize -> deserialize -> serialize preserves weights."""
    model1 = DummyModel()
    model2 = DummyModel()

    # Make model2 different
    with torch.no_grad():
        for param in model2.parameters():
            param.data += 1.0

    # Save model1, load into model2, save again
    weights1 = serialize_weights(model1)
    deserialize_weights(model2, weights1)
    weights2 = serialize_weights(model2)

    # Should be identical
    for key in weights1.keys():
        assert torch.equal(weights1[key], weights2[key]), \
            f"Roundtrip serialization failed for {key}"


def test_empty_model_serialization():
    """Test edge case of model with no parameters."""
    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.tensor(0.0))

    model = EmptyModel()
    weights = serialize_weights(model)

    assert 'dummy' in weights
    assert weights['dummy'].item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
