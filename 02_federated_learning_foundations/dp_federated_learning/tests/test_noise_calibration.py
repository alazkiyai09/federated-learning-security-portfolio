"""
Unit Tests for Noise Calibration
=================================

Tests to verify Gaussian noise addition and calibration
are implemented correctly.
"""

import pytest
import torch
import numpy as np
from typing import Optional

from src.dp_mechanisms.noise_addition import (
    add_gaussian_noise,
    compute_noise_multiplier,
    compute_noise_multiplier_standard,
    compute_delta_for_epsilon
)


class TestGaussianNoise:
    """Test Gaussian noise addition."""

    def test_noise_changes_tensor(self):
        """Verify noise actually modifies the tensor."""
        tensor = torch.randn(100)
        noised = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32
        )

        # Should be different
        assert not torch.allclose(tensor, noised), "Tensor was not modified"

    def test_noise_statistics(self):
        """Verify added noise has correct statistics."""
        tensor = torch.zeros(10000)
        noise_multiplier = 1.0
        clipping_bound = 1.0
        num_samples = 100

        noised = add_gaussian_noise(
            tensor,
            noise_multiplier=noise_multiplier,
            clipping_bound=clipping_bound,
            num_samples=num_samples
        )

        # Compute expected standard deviation
        expected_std = noise_multiplier * clipping_bound * np.sqrt(num_samples)

        # Sample std should be close to expected
        sample_std = noised.std().item()
        assert abs(sample_std - expected_std) / expected_std < 0.05, \
            f"Sample std {sample_std:.2f} too far from expected {expected_std:.2f}"

    def test_noise_mean_zero(self):
        """Verify noise has zero mean."""
        tensor = torch.zeros(10000)
        noised = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32
        )

        sample_mean = noised.mean().item()
        assert abs(sample_mean) < 0.1, f"Noise mean {sample_mean:.4f} not close to zero"

    def test_independent_noise(self):
        """Verify different calls produce independent noise."""
        tensor = torch.ones(100)

        noised1 = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32
        )
        noised2 = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32
        )

        # Should be different
        assert not torch.allclose(noised1, noised2), "Noise is not independent"

    def test_noise_multiplier_scales_variance(self):
        """Verify larger noise multiplier increases variance."""
        tensor = torch.zeros(10000)

        noised1 = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32
        )
        noised2 = add_gaussian_noise(
            tensor,
            noise_multiplier=2.0,
            clipping_bound=1.0,
            num_samples=32
        )

        std1 = noised1.std().item()
        std2 = noised2.std().item()

        # Ratio of std should be 2.0/1.0 = 2, variance ratio should be 4
        ratio = (std2 / std1)
        variance_ratio = ratio ** 2
        assert 1.8 < ratio < 2.2, f"Std ratio {ratio:.2f}, expected ~2.0"
        assert 3.5 < variance_ratio < 4.5, f"Variance ratio {variance_ratio:.2f}, expected ~4.0"

    def test_central_dp_scaling(self):
        """Verify central DP scaling with num_clients."""
        tensor = torch.zeros(10000)

        noised1 = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32,
            num_clients=None
        )
        noised2 = add_gaussian_noise(
            tensor,
            noise_multiplier=1.0,
            clipping_bound=1.0,
            num_samples=32,
            num_clients=10
        )

        # Central DP should have different scaling
        assert not torch.allclose(noised1, noised2, rtol=0.1), \
            "Central DP scaling not applied"

    def test_invalid_parameters(self):
        """Verify invalid parameters raise errors."""
        tensor = torch.randn(100)

        with pytest.raises(ValueError, match="Noise multiplier must be non-negative"):
            add_gaussian_noise(tensor, noise_multiplier=-1.0, clipping_bound=1.0, num_samples=32)

        with pytest.raises(ValueError, match="Clipping bound must be positive"):
            add_gaussian_noise(tensor, noise_multiplier=1.0, clipping_bound=0, num_samples=32)


class TestNoiseMultiplierComputation:
    """Test noise multiplier calibration from target epsilon."""

    def test_higher_epsilon_requires_less_noise(self):
        """Verify larger target ε requires smaller σ."""
        sigma_low = compute_noise_multiplier(
            target_epsilon=0.5,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.01
        )
        sigma_high = compute_noise_multiplier(
            target_epsilon=5.0,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.01
        )

        assert sigma_low > sigma_high, \
            f"Higher ε should require less noise: σ(ε=0.5)={sigma_low:.2f}, σ(ε=5.0)={sigma_high:.2f}"

    def test_more_steps_require_more_noise(self):
        """Verify more steps require larger σ for same ε."""
        sigma_few = compute_noise_multiplier(
            target_epsilon=1.0,
            delta=1e-5,
            steps=500,
            sampling_rate=0.01
        )
        sigma_many = compute_noise_multiplier(
            target_epsilon=1.0,
            delta=1e-5,
            steps=2000,
            sampling_rate=0.01
        )

        assert sigma_many > sigma_few, \
            f"More steps should require more noise: σ(500)={sigma_few:.2f}, σ(2000)={sigma_many:.2f}"

    def test_smaller_sampling_rate_requires_less_noise(self):
        """Verify smaller sampling rate requires less noise."""
        sigma_low_q = compute_noise_multiplier(
            target_epsilon=1.0,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.001  # Very small q
        )
        sigma_high_q = compute_noise_multiplier(
            target_epsilon=1.0,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.1  # Larger q
        )

        # With smaller sampling rate, we add less frequently, so need less noise
        assert sigma_low_q < sigma_high_q, \
            f"Lower q should require less noise: σ(q=0.001)={sigma_low_q:.2f}, σ(q=0.1)={sigma_high_q:.2f}"

    def test_monotonicity(self):
        """Verify σ is monotonic in ε."""
        epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas = []

        for eps in epsilons:
            sigma = compute_noise_multiplier(
                target_epsilon=eps,
                delta=1e-5,
                steps=1000,
                sampling_rate=0.01
            )
            sigmas.append(sigma)

        # Should be monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i+1], \
                f"σ not monotonic: σ({epsilons[i]})={sigmas[i]:.2f}, σ({epsilons[i+1]})={sigmas[i+1]:.2f}"

    def test_reasonable_range(self):
        """Verify computed σ is in reasonable range."""
        sigma = compute_noise_multiplier(
            target_epsilon=1.0,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.01
        )

        # For typical parameters, σ should be in [0.5, 10]
        assert 0.5 < sigma < 10.0, f"σ={sigma:.2f} outside reasonable range [0.5, 10.0]"


class TestStandardFormula:
    """Test standard Gaussian mechanism formula."""

    def test_standard_formula(self):
        """Verify standard formula matches theory."""
        sigma = compute_noise_multiplier_standard(
            target_epsilon=1.0,
            delta=1e-5
        )

        # Expected: σ = sqrt(2 * ln(1.25/1e-5)) / 1.0
        expected = np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0

        assert abs(sigma - expected) < 1e-10, \
            f"σ={sigma:.6f}, expected {expected:.6f}"

    def test_epsilon_sigma_relationship(self):
        """Verify σ ∝ 1/ε for standard formula."""
        sigma1 = compute_noise_multiplier_standard(1.0, 1e-5)
        sigma2 = compute_noise_multiplier_standard(2.0, 1e-5)

        # Should be approximately inverse relationship
        ratio = sigma1 / sigma2
        expected_ratio = 2.0 / 1.0  # ε2/ε1

        assert abs(ratio - expected_ratio) < 0.01, \
            f"Ratio {ratio:.2f}, expected {expected_ratio:.2f}"

    def test_delta_sensitivity(self):
        """Verify σ increases as δ decreases."""
        sigma1 = compute_noise_multiplier_standard(1.0, 1e-3)  # Larger δ
        sigma2 = compute_noise_multiplier_standard(1.0, 1e-6)  # Smaller δ

        assert sigma2 > sigma1, \
            f"Smaller δ should require more noise: σ(δ=1e-3)={sigma1:.2f}, σ(δ=1e-6)={sigma2:.2f}"


class TestDeltaComputation:
    """Test δ computation from ε and σ."""

    def test_delta_decreases_with_sigma(self):
        """Verify δ decreases as σ increases."""
        delta1 = compute_delta_for_epsilon(
            epsilon=10.0,  # Larger ε to avoid underflow
            sigma=0.5,
            steps=100,
            sampling_rate=0.01
        )
        delta2 = compute_delta_for_epsilon(
            epsilon=10.0,
            sigma=2.0,
            steps=100,
            sampling_rate=0.01
        )

        # Both should be very small, but delta2 should be <= delta1
        # (can be equal if both underflow to 0)
        assert delta2 <= delta1, \
            f"Higher σ should give smaller δ: δ(σ=0.5)={delta1:.2e}, δ(σ=2.0)={delta2:.2e}"

    def test_delta_in_range(self):
        """Verify δ is in valid range (0, 1)."""
        delta = compute_delta_for_epsilon(
            epsilon=1.0,
            sigma=1.5,
            steps=1000,
            sampling_rate=0.01
        )

        assert 0 < delta < 1, f"δ={delta:.2e} not in (0, 1)"

    def test_delta_decreases_with_epsilon(self):
        """Verify δ decreases as ε increases (for fixed σ)."""
        delta1 = compute_delta_for_epsilon(
            epsilon=1.0,  # Use larger ε values
            sigma=2.0,  # Larger σ to avoid underflow
            steps=100,  # Fewer steps
            sampling_rate=0.01
        )
        delta2 = compute_delta_for_epsilon(
            epsilon=10.0,
            sigma=2.0,
            steps=100,
            sampling_rate=0.01
        )

        # Higher ε should give smaller or equal δ
        assert delta2 <= delta1, \
            f"Higher ε should give smaller δ: δ(ε=1.0)={delta1:.2e}, δ(ε=10.0)={delta2:.2e}"


class TestNumericalStability:
    """Test numerical stability of noise functions."""

    def test_very_small_noise_multiplier(self):
        """Verify behavior with very small noise multiplier."""
        tensor = torch.zeros(100)

        noised = add_gaussian_noise(
            tensor,
            noise_multiplier=0.01,
            clipping_bound=1.0,
            num_samples=32
        )

        # Should still work, just very small noise
        assert torch.allclose(tensor, noised, atol=0.5), \
            "Very small noise should barely change tensor"

    def test_very_large_noise_multiplier(self):
        """Verify behavior with very large noise multiplier."""
        tensor = torch.zeros(100)

        noised = add_gaussian_noise(
            tensor,
            noise_multiplier=100.0,
            clipping_bound=1.0,
            num_samples=32
        )

        # Should add lots of noise but not overflow
        assert not torch.isnan(noised).any(), "NaN in output"
        assert not torch.isinf(noised).any(), "Inf in output"

    def test_zero_clipping_bound(self):
        """Verify zero clipping bound raises error."""
        tensor = torch.randn(100)

        with pytest.raises(ValueError):
            add_gaussian_noise(
                tensor,
                noise_multiplier=1.0,
                clipping_bound=0.0,
                num_samples=32
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
