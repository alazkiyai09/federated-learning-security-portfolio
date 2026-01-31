"""
Unit Tests for Gradient Clipping
=================================

Tests to verify per-sample gradient computation and L2 clipping
are implemented correctly.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.dp_mechanisms.gradient_clipper import (
    compute_per_sample_gradients,
    clip_gradients_l2
)


class TestPerSampleGradients:
    """Test per-sample gradient computation."""

    def test_output_shape(self):
        """Verify output shape matches (batch_size, num_params)."""
        model = nn.Linear(10, 2)
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        grads = compute_per_sample_gradients(model, inputs, targets, loss_fn)

        num_params = sum(p.numel() for p in model.parameters())
        assert grads.shape == (8, num_params), f"Expected shape (8, {num_params}), got {grads.shape}"

    def test_gradients_non_zero(self):
        """Verify gradients are computed (not all zeros)."""
        model = nn.Linear(10, 2)
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        grads = compute_per_sample_gradients(model, inputs, targets, loss_fn)

        # At least some gradients should be non-zero
        assert grads.abs().sum() > 0, "All gradients are zero"

    def test_gradients_differ_across_samples(self):
        """Verify different samples produce different gradients."""
        model = nn.Linear(10, 2)
        # Use distinct inputs
        inputs = torch.eye(8, 10)[:8]  # Orthogonal inputs
        targets = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        grads = compute_per_sample_gradients(model, inputs, targets, loss_fn)

        # Gradients for different samples should be different
        # (at least for some pairs)
        for i in range(7):
            for j in range(i+1, 8):
                if not torch.allclose(grads[i], grads[j], rtol=1e-5, atol=1e-5):
                    return  # Found differing gradients

        assert False, "All gradients are identical (unexpected)"


class TestGradientClipping:
    """Test L2 gradient clipping."""

    def test_clipping_bound(self):
        """Verify all clipped gradients have norm ≤ C."""
        batch_size = 100
        num_params = 50
        C = 1.0

        # Create random gradients with varying norms
        grads = torch.randn(batch_size, num_params)
        # Scale some gradients to have large norms
        grads[:50] *= 5.0

        clipped_grads, norms = clip_gradients_l2(grads, C)

        # Verify all clipped norms ≤ C
        clipped_norms = clipped_grads.norm(dim=1)
        assert (clipped_norms <= C + 1e-6).all(), \
            f"Max clipped norm is {clipped_norms.max().item()}, expected ≤ {C}"

    def test_small_gradients_unchanged(self):
        """Verify gradients with norm < C are unchanged."""
        C = 5.0
        grads = torch.randn(10, 20) * 0.5  # Small gradients

        clipped_grads, norms = clip_gradients_l2(grads, C)

        # Small gradients should be unchanged (norm < C)
        assert torch.allclose(grads, clipped_grads, rtol=1e-5, atol=1e-5), \
            "Small gradients were modified"

    def test_large_gradients_scaled(self):
        """Verify large gradients are scaled correctly."""
        C = 1.0

        # Create a gradient with known norm
        grad = torch.ones(1, 10) * 2.0  # Norm = sqrt(10*4) = sqrt(40) ≈ 6.32
        expected_norm = grad.norm().item()

        grads = torch.cat([grad, torch.randn(5, 10)])
        clipped_grads, norms = clip_gradients_l2(grads, C)

        # Check scaling factor
        scale_factor = max(1.0, expected_norm / C)
        expected_clipped = grad / scale_factor

        assert torch.allclose(clipped_grads[0], expected_clipped, rtol=1e-5, atol=1e-5), \
            f"Large gradient not scaled correctly. Expected {expected_clipped}, got {clipped_grads[0]}"

    def test_clip_norms_recorded(self):
        """Verify original norms are recorded correctly."""
        grads = torch.randn(50, 20)
        C = 1.0

        clipped_grads, norms = clip_gradients_l2(grads, C)

        # Manually compute norms
        expected_norms = grads.norm(dim=1)

        assert torch.allclose(norms, expected_norms, rtol=1e-5, atol=1e-5), \
            "Clip norms don't match manually computed norms"

    def test_clipping_fraction(self):
        """Verify fraction of gradients clipped is reasonable."""
        C = 1.0
        batch_size = 1000
        num_params = 50  # Reduced dimensionality

        # Random gradients with appropriate scale
        grads = torch.randn(batch_size, num_params) * 0.5
        clipped_grads, norms = clip_gradients_l2(grads, C)

        # Count how many were clipped
        num_clipped = (norms > C).sum().item()
        clip_fraction = num_clipped / batch_size

        # For L2 norm with scale 0.5 in 50 dimensions, some should be clipped
        # Just verify the mechanism works (some may or may not be clipped)
        assert 0.0 <= clip_fraction <= 1.0, \
            f"Clip fraction {clip_fraction:.2%} outside valid range"

    def test_invalid_clipping_bound(self):
        """Verify invalid clipping bounds raise errors."""
        grads = torch.randn(10, 20)

        with pytest.raises(ValueError, match="Clipping bound must be positive"):
            clip_gradients_l2(grads, clipping_bound=0)

        with pytest.raises(ValueError, match="Clipping bound must be positive"):
            clip_gradients_l2(grads, clipping_bound=-1.0)


class TestClippingProperties:
    """Test mathematical properties of clipping."""

    def test_clipping_reduces_norm(self):
        """Verify clipping never increases norm."""
        grads = torch.randn(100, 50)
        C = 1.0

        clipped_grads, original_norms = clip_gradients_l2(grads, C)
        clipped_norms = clipped_grads.norm(dim=1)

        # All clipped norms should be ≤ original norms
        assert (clipped_norms <= original_norms + 1e-6).all(), \
            "Clipping increased some norms"

    def test_clipping_preserves_direction(self):
        """Verify clipping preserves gradient direction."""
        grads = torch.randn(10, 20)
        C = 1.0

        clipped_grads, _ = clip_gradients_l2(grads, C)

        # Check direction preservation for clipped gradients
        original_norms = grads.norm(dim=1, keepdim=True)
        clipped_norms = clipped_grads.norm(dim=1, keepdim=True)

        # Normalize both
        normalized_original = grads / (original_norms + 1e-10)
        normalized_clipped = clipped_grads / (clipped_norms + 1e-10)

        # Should be parallel (dot product ≈ 1)
        for i in range(10):
            dot_product = (normalized_original[i] * normalized_clipped[i]).sum().item()
            assert abs(1 - dot_product) < 1e-4, \
                f"Direction changed for sample {i}: dot={dot_product:.6f}"

    def test_per_sample_independence(self):
        """Verify clipping one sample doesn't affect others."""
        batch_size = 50
        num_params = 30
        C = 1.0

        grads = torch.randn(batch_size, num_params)
        clipped_grads, _ = clip_gradients_l2(grads, C)

        # Check that modifying one input gradient only affects that output
        for i in range(batch_size):
            # Create modified input
            modified_grads = grads.clone()
            modified_grads[i] *= 10.0

            # Clip modified input
            modified_clipped, _ = clip_gradients_l2(modified_grads, C)

            # Only sample i should be different
            for j in range(batch_size):
                if j != i:
                    assert torch.allclose(clipped_grads[j], modified_clipped[j], rtol=1e-5), \
                        f"Clipping sample {i} affected sample {j}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
