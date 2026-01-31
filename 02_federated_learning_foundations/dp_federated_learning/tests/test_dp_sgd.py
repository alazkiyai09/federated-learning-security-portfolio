"""
Unit Tests for DP-SGD Optimizer
================================

Tests for custom DP-SGD implementation and Opacus wrapper.
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.models.dp_sgd_custom import (
    DPSGDOptimizer,
    DPSGDMetrics,
    create_dp_sgd_optimizer
)


class TestDPSGDOptimizerInit:
    """Test DPSGDOptimizer initialization."""

    def test_initialization(self):
        """Verify optimizer initializes correctly."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        assert optimizer.noise_multiplier == 1.5
        assert optimizer.clipping_bound == 1.0
        assert optimizer.batch_size == 32
        assert optimizer.lr == 0.01
        assert optimizer.sampling_rate == 32/1000
        assert optimizer.steps == 0

    def test_invalid_parameters(self):
        """Verify invalid parameters raise errors."""
        model = nn.Linear(10, 2)

        with pytest.raises(ValueError, match="Noise multiplier must be positive"):
            DPSGDOptimizer(
                model=model,
                noise_multiplier=0,
                clipping_bound=1.0,
                batch_size=32,
                lr=0.01,
                dataset_size=1000
            )

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            DPSGDOptimizer(
                model=model,
                noise_multiplier=1.5,
                clipping_bound=1.0,
                batch_size=32,
                lr=0,
                dataset_size=1000
            )

    def test_missing_sampling_info(self):
        """Verify error when sampling info is missing."""
        model = nn.Linear(10, 2)

        with pytest.raises(ValueError, match="Must provide either sampling_rate or dataset_size"):
            DPSGDOptimizer(
                model=model,
                noise_multiplier=1.5,
                clipping_bound=1.0,
                batch_size=32,
                lr=0.01
            )


class TestDPSGDStep:
    """Test DP-SGD step execution."""

    def test_step_returns_metrics(self):
        """Verify step() returns DPSGDMetrics."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        metrics = optimizer.step(inputs, targets, loss_fn)

        assert isinstance(metrics, DPSGDMetrics)
        assert metrics.clip_norm_mean >= 0
        assert metrics.clip_norm_max >= 0
        assert 0 <= metrics.clip_fraction <= 1
        assert metrics.noise_std > 0
        assert metrics.epsilon_spent > 0

    def test_step_updates_parameters(self):
        """Verify step() updates model parameters."""
        model = nn.Linear(10, 2)
        original_params = [p.data.clone() for p in model.parameters()]

        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        optimizer.step(inputs, targets, loss_fn)

        # Parameters should have changed
        for p, orig_p in zip(model.parameters(), original_params):
            assert not torch.allclose(p.data, orig_p, rtol=1e-5)

    def test_step_increments_counter(self):
        """Verify step() increments the step counter."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        assert optimizer.steps == 0

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        optimizer.step(inputs, targets, loss_fn)
        assert optimizer.steps == 1

        optimizer.step(inputs, targets, loss_fn)
        assert optimizer.steps == 2

    def test_clipping_statistics(self):
        """Verify clipping statistics are reasonable."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        metrics = optimizer.step(inputs, targets, loss_fn)

        # Clip statistics should be valid
        assert metrics.clip_norm_mean >= 0
        assert metrics.clip_norm_max >= metrics.clip_norm_mean
        assert 0 <= metrics.clip_fraction <= 1


class TestPrivacyAccounting:
    """Test privacy accounting in DP-SGD."""

    def test_epsilon_increases(self):
        """Verify ε increases with more steps."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        eps1 = optimizer.get_epsilon()
        optimizer.step(inputs, targets, loss_fn)
        eps2 = optimizer.get_epsilon()

        assert eps2 > eps1, f"ε did not increase: {eps1:.6f} → {eps2:.6f}"

    def test_get_privacy_spent(self):
        """Verify get_privacy_spent() returns correct values."""
        model = nn.Linear(10, 2)
        delta = 1e-5
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000,
            delta=delta
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        optimizer.step(inputs, targets, loss_fn)

        epsilon, returned_delta = optimizer.get_privacy_spent()

        assert epsilon > 0
        assert returned_delta == delta

    def test_monotonic_privacy_consumption(self):
        """Verify privacy consumption is monotonic."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        previous_eps = 0.0
        for _ in range(10):
            optimizer.step(inputs, targets, loss_fn)
            current_eps = optimizer.get_epsilon()
            assert current_eps > previous_eps, \
                f"ε not monotonic: {previous_eps:.6f} → {current_eps:.6f}"
            previous_eps = current_eps


class TestConvenienceFunction:
    """Test create_dp_sgd_optimizer() convenience function."""

    def test_creates_optimizer(self):
        """Verify convenience function creates optimizer."""
        model = nn.Linear(10, 2)
        optimizer = create_dp_sgd_optimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        assert isinstance(optimizer, DPSGDOptimizer)
        assert optimizer.noise_multiplier == 1.5
        assert optimizer.clipping_bound == 1.0

    def test_optimizer_works(self):
        """Verify created optimizer actually works."""
        model = nn.Linear(10, 2)
        optimizer = create_dp_sgd_optimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        metrics = optimizer.step(inputs, targets, loss_fn)

        assert metrics.epsilon_spent > 0


class TestMomentum:
    """Test momentum functionality."""

    def test_momentum_initialized_zero(self):
        """Verify momentum is initialized to zero."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000,
            momentum=0.9
        )

        # All velocities should be zero
        for v in optimizer.velocity:
            assert (v == 0).all()

    def test_momentum_updates(self):
        """Verify momentum state updates after steps."""
        model = nn.Linear(10, 2)
        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=1.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000,
            momentum=0.9
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        optimizer.step(inputs, targets, loss_fn)

        # Velocity should now be non-zero
        for v in optimizer.velocity:
            assert v.abs().sum() > 0


class TestDifferentNoiseMultipliers:
    """Test behavior with different noise multipliers."""

    def test_small_noise(self):
        """Verify behavior with small noise multiplier."""
        model = nn.Linear(10, 2)

        optimizer = DPSGDOptimizer(
            model=model,
            noise_multiplier=0.1,  # Small but positive
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        metrics = optimizer.step(inputs, targets, loss_fn)

        # Should work with small noise
        assert metrics.noise_std > 0
        assert metrics.noise_std < 1.0  # Small noise

    def test_high_noise_reduces_epsilon(self):
        """Verify higher noise multiplier gives lower ε."""
        model1 = nn.Linear(10, 2)
        model2 = nn.Linear(10, 2)

        # Copy weights so models are identical
        model2.load_state_dict(model1.state_dict())

        opt_low_noise = DPSGDOptimizer(
            model=model1,
            noise_multiplier=0.5,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        opt_high_noise = DPSGDOptimizer(
            model=model2,
            noise_multiplier=2.0,
            clipping_bound=1.0,
            batch_size=32,
            lr=0.01,
            dataset_size=1000
        )

        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        opt_low_noise.step(inputs, targets, loss_fn)
        opt_high_noise.step(inputs, targets, loss_fn)

        eps_low = opt_low_noise.get_epsilon()
        eps_high = opt_high_noise.get_epsilon()

        assert eps_low > eps_high, \
            f"Lower noise should consume more privacy: {eps_low:.4f} > {eps_high:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
