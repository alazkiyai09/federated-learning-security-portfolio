"""
Unit tests for personalization methods.

Tests core logic without requiring full FL simulation.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLocalFineTuning:
    """Unit tests for Local Fine-Tuning method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        n_features = 10

        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples, 1)).float()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        return loader, n_features

    @pytest.fixture
    def sample_model(self, sample_data):
        """Create sample model."""
        _, n_features = sample_data

        from src.models.base import FraudDetectionModel

        return FraudDetectionModel(
            input_dim=n_features,
            hidden_dims=[16, 8]
        )

    def test_model_forward(self, sample_model, sample_data):
        """Test model forward pass."""
        loader, _ = sample_data

        for features, _ in loader:
            output = sample_model(features)
            assert output.shape[0] == features.shape[0]
            assert output.shape[1] == 1
            assert (output >= 0).all() and (output <= 1).all()  # Sigmoid

    def test_model_parameter_count(self, sample_model):
        """Test parameter counting."""
        n_params = sample_model.get_num_parameters()
        assert n_params > 0

        params_by_layer = sample_model.get_num_parameters_per_layer()
        assert len(params_by_layer) > 0

    def test_model_clone(self, sample_model):
        """Test model cloning."""
        cloned = sample_model.clone()

        # Check that parameters are equal
        for p1, p2 in zip(sample_model.parameters(), cloned.parameters()):
            assert torch.equal(p1, p2)

        # Check that they are different objects
        assert id(sample_model) != id(cloned)

    def test_feature_extractor_split(self, sample_model):
        """Test feature extractor / classifier split."""
        # Test that model has required attributes
        assert hasattr(sample_model, 'feature_extractor')
        assert hasattr(sample_model, 'classifier')

        # Test forward through feature extractor only
        x = torch.randn(5, 10)
        features = sample_model.forward_features(x)

        # Features should have different dimension than input
        assert features.shape[0] == 5  # Batch size preserved
        assert features.shape[1] == 8  # Last hidden dim

    def test_freeze_unfreeze(self, sample_model):
        """Test freezing and unfreezing layers."""
        # Freeze feature extractor
        sample_model.freeze_feature_extractor()

        # Check that feature extractor params are frozen
        for param in sample_model.feature_extractor.parameters():
            assert not param.requires_grad

        # Unfreeze
        sample_model.unfreeze_feature_extractor()

        # Check that params are trainable again
        for param in sample_model.feature_extractor.parameters():
            assert param.requires_grad


class TestFedPer:
    """Unit tests for FedPer method."""

    @pytest.fixture
    def sample_model(self):
        """Create sample model for FedPer."""
        from src.models.base import FraudDetectionModel

        return FraudDetectionModel(
            input_dim=10,
            hidden_dims=[16, 8]
        )

    def test_parameter_split(self, sample_model):
        """Test splitting parameters into shared and personal."""
        from src.models.utils import split_model_parameters

        shared, personal = split_model_parameters(sample_model)

        # Both should be non-empty
        assert len(shared) > 0
        assert len(personal) > 0

        # Total should equal all parameters
        all_params = list(sample_model.parameters())
        assert len(shared) + len(personal) == len(all_params)

    def test_get_shared_parameters(self, sample_model):
        """Test getting shared parameters."""
        from src.models.utils import get_parameters_by_layer_type

        shared_params = get_parameters_by_layer_type(sample_model, 'feature_extractor')

        assert len(shared_params) > 0
        assert all(isinstance(p, np.ndarray) for p in shared_params)

    def test_get_personal_parameters(self, sample_model):
        """Test getting personal parameters."""
        from src.models.utils import get_parameters_by_layer_type

        personal_params = get_parameters_by_layer_type(sample_model, 'classifier')

        assert len(personal_params) > 0
        assert all(isinstance(p, np.ndarray) for p in personal_params)


class TestDitto:
    """Unit tests for Ditto method."""

    @pytest.fixture
    def sample_models(self):
        """Create two sample models for Ditto."""
        from src.models.base import FraudDetectionModel

        model1 = FraudDetectionModel(input_dim=10, hidden_dims=[16, 8])
        model2 = FraudDetectionModel(input_dim=10, hidden_dims=[16, 8])

        # Make them different
        model2.apply(lambda m: m.weight.data.normal_(0, 0.1) if hasattr(m, 'weight') else None)

        return model1, model2

    def test_param_distance(self, sample_models):
        """Test computing parameter distance."""
        from src.models.utils import compute_param_distance

        model1, model2 = sample_models

        distance = compute_param_distance(model1, model2)

        assert distance >= 0
        assert isinstance(distance, (int, float))

    def test_distance_zero_for_same_model(self):
        """Test that distance is zero for identical models."""
        from src.models.utils import compute_param_distance
        from src.models.base import FraudDetectionModel

        model = FraudDetectionModel(input_dim=10, hidden_dims=[16, 8])

        distance = compute_param_distance(model, model)

        assert distance == 0


class TestPerFedAvg:
    """Unit tests for Per-FedAvg method."""

    @pytest.fixture
    def sample_model(self):
        """Create sample model."""
        from src.models.base import FraudDetectionModel

        return FraudDetectionModel(
            input_dim=10,
            hidden_dims=[16, 8]
        )

    def test_moreau_envelope(self, sample_model):
        """Test Moreau envelope computation."""
        from src.models.utils import compute_moreau_envelope

        # Get current state as "global" parameters
        global_params = sample_model.get_state_dict_by_layer_type('all')

        # Compute Moreau term
        moreau = compute_moreau_envelope(
            sample_model,
            global_params,
            beta=1.0
        )

        # Should be zero (same parameters)
        assert moreau.item() == 0

    def test_moreau_envelope_with_perturbation(self, sample_model):
        """Test Moreau envelope with perturbed parameters."""
        from src.models.utils import compute_moreau_envelope

        # Get current state
        global_params = sample_model.get_state_dict_by_layer_type('all')

        # Perturb model parameters
        with torch.no_grad():
            for param in sample_model.parameters():
                param.data += 0.1 * torch.randn_like(param)

        # Compute Moreau term
        moreau = compute_moreau_envelope(
            sample_model,
            global_params,
            beta=1.0
        )

        # Should be positive (different parameters)
        assert moreau.item() > 0


class TestMetrics:
    """Unit tests for metrics computation."""

    def test_compute_fraud_metrics(self):
        """Test fraud detection metrics computation."""
        from src.utils import compute_fraud_metrics

        # Create predictions and targets
        predictions = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        targets = np.array([0, 0, 0, 1, 1])

        metrics = compute_fraud_metrics(predictions, targets)

        # Check required metrics
        assert 'auc' in metrics
        assert 'pr_auc' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'detection_rate' in metrics
        assert 'false_alarm_rate' in metrics

        # Check value ranges
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1

    def test_fairness_metrics(self):
        """Test fairness metrics computation."""
        from src.utils import compute_fairness_metrics

        # Create per-client metrics
        per_client_metrics = {
            0: {'auc': 0.8},
            1: {'auc': 0.85},
            2: {'auc': 0.75},
            3: {'auc': 0.9},
        }

        fairness = compute_fairness_metrics(per_client_metrics)

        # Check metrics
        assert 'performance_mean' in fairness
        assert 'performance_std' in fairness
        assert 'performance_variance' in fairness
        assert 'worst_client_performance' in fairness
        assert 'best_client_performance' in fairness
        assert 'gini_coefficient' in fairness

        # Check values
        assert fairness['worst_client_performance'] == 0.75
        assert fairness['best_client_performance'] == 0.9
        assert fairness['performance_range'] == 0.15


class TestComputeTracking:
    """Unit tests for compute budget tracking."""

    def test_flops_estimation(self):
        """Test FLOPs estimation."""
        from src.models.utils import (
            compute_flops_per_forward,
            compute_flops_per_backward
        )

        input_dim = 10
        hidden_dims = [16, 8]
        batch_size = 32

        forward_flops = compute_flops_per_forward(input_dim, hidden_dims, batch_size)
        backward_flops = compute_flops_per_backward(input_dim, hidden_dims, batch_size)

        assert forward_flops > 0
        assert backward_flops > 0
        assert backward_flops > forward_flops  # Backward ~2x forward

    def test_communication_cost(self):
        """Test communication cost estimation."""
        from src.models.utils import compute_communication_cost

        # Create sample parameters
        parameters = [
            np.random.randn(10, 16).astype(np.float32),
            np.random.randn(16, 8).astype(np.float32),
            np.random.randn(8, 1).astype(np.float32),
        ]

        bytes_cost = compute_communication_cost(parameters)

        # Should be total elements * 4 bytes (float32)
        total_elements = sum(p.size for p in parameters)
        expected_bytes = total_elements * 4

        assert bytes_cost == expected_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
