"""
Unit Tests for Custom Strategies

Tests strategy aggregation logic.
"""

import numpy as np
import pytest
from flwr.common import FitRes, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from src.strategy import FedAvgCustom, FedProxCustom, FedAdamCustom


class MockClientProxy(ClientProxy):
    """Mock client proxy for testing."""

    def __init__(self, cid: str):
        super().__init__(cid)
        self.cid = cid

    def get_properties(self):
        return {}

    def get_parameters(self):
        return Parameters([], "")

    def fit(self, ins):
        return FitRes(
            parameters=Parameters([], ""),
            num_examples=100,
            metrics={"loss": 0.5},
        )

    def evaluate(self, ins):
        return EvaluateRes(
            loss=0.3,
            num_examples=50,
            metrics={"accuracy": 0.9},
        )


@pytest.fixture
def sample_parameters():
    """Create sample parameters for testing."""
    # Create simple numpy arrays
    layer1 = np.random.randn(10, 5).astype(np.float32)
    layer2 = np.random.randn(5).astype(np.float32)

    # Convert to bytes (Flower format)
    from flwr.common import ndarray_to_bytes

    tensors = [ndarray_to_bytes(layer1), ndarray_to_bytes(layer2)]
    return Parameters(tensors=tensors, tensor_type="")


def test_fedavg_initialization():
    """Test FedAvg strategy initialization."""
    strategy = FedAvgCustom(
        fraction_fit=0.5,
        fraction_evaluate=0.3,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    assert strategy.fraction_fit == 0.5
    assert strategy.fraction_evaluate == 0.3
    assert strategy.min_fit_clients == 2
    assert strategy.min_evaluate_clients == 2


def test_fedprox_initialization():
    """Test FedProx strategy initialization."""
    strategy = FedProxCustom(
        proximal_mu=0.1,
        fraction_fit=0.5,
        fraction_evaluate=0.3,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    assert strategy.proximal_mu == 0.1
    assert strategy.fraction_fit == 0.5
    assert strategy.fraction_evaluate == 0.3


def test_fedadam_initialization():
    """Test FedAdam strategy initialization."""
    strategy = FedAdamCustom(
        tau=0.9,
        eta=0.01,
        fraction_fit=0.5,
        fraction_evaluate=0.3,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    assert strategy.tau == 0.9
    assert strategy.eta == 0.01
    assert strategy.fraction_fit == 0.5
    assert strategy.fraction_evaluate == 0.3


def test_weighted_average_aggregation():
    """Test weighted average aggregation logic."""
    from src.utils import weighted_average

    # Create sample metrics
    metrics = [
        (100, {"accuracy": 0.8, "loss": 0.5}),
        (200, {"accuracy": 0.9, "loss": 0.4}),
        (150, {"accuracy": 0.85, "loss": 0.45}),
    ]

    total_samples, aggregated = weighted_average(metrics)

    # Verify total samples
    assert total_samples == 450

    # Verify weighted averages
    expected_accuracy = (100 * 0.8 + 200 * 0.9 + 150 * 0.85) / 450
    expected_loss = (100 * 0.5 + 200 * 0.4 + 150 * 0.45) / 450

    assert abs(aggregated["accuracy"] - expected_accuracy) < 1e-6
    assert abs(aggregated["loss"] - expected_loss) < 1e-6


def test_aggregate_fit_empty_results():
    """Test aggregate_fit with empty results."""
    strategy = FedAvgCustom()

    aggregated_params, metrics = strategy.aggregate_fit(
        rnd=1,
        results=[],
        failures=[],
    )

    assert aggregated_params is None
    assert metrics == {}


def test_aggregate_evaluate_empty_results():
    """Test aggregate_evaluate with empty results."""
    strategy = FedAvgCustom()

    loss, metrics = strategy.aggregate_evaluate(
        rnd=1,
        results=[],
        failures=[],
    )

    assert loss is None
    assert metrics == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
