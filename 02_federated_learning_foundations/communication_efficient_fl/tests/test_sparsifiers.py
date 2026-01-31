"""
Unit tests for gradient sparsification techniques.

Tests verify correctness of sparsification methods, mask accuracy,
and compression ratio calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.sparsifiers import (
    top_k_sparsify,
    random_k_sparsify,
    threshold_sparsify,
    top_k_sparsify_percentage
)


class TestTopKSparsify:
    """Tests for Top-K sparsification."""

    def test_top_k_basic(self):
        """Test basic Top-K sparsification."""
        gradients = np.array([0.1, -0.5, 0.3, 0.05, -0.8, 0.7])
        k = 3

        sparse, mask, ratio = top_k_sparsify(gradients, k)

        # Check that exactly k elements are non-zero
        assert np.sum(mask) == k
        assert np.sum(sparse != 0) == k

        # Check that masked elements are zero
        assert np.all(sparse[~mask] == 0)

        # Check that non-masked elements preserve values
        np.testing.assert_array_equal(sparse[mask], gradients[mask])

    def test_top_k_selects_largest(self):
        """Test that Top-K selects largest magnitude elements."""
        gradients = np.array([0.1, -0.9, 0.3, 0.05, -0.8, 0.7])
        k = 3

        sparse, mask, ratio = top_k_sparsify(gradients, k)

        # Get indices of selected elements
        selected_indices = np.where(mask)[0]
        selected_values = gradients[selected_indices]

        # Get indices of top-k by magnitude (ground truth)
        magnitudes = np.abs(gradients)
        top_k_indices = np.argpartition(-magnitudes, k)[:k]
        top_k_values = gradients[top_k_indices]

        # Check that selected values match top-k values (order may differ)
        assert set(selected_values) == set(top_k_values)

    def test_top_k_k_equals_zero(self):
        """Test that k=0 raises error."""
        gradients = np.array([0.1, -0.5, 0.3])

        with pytest.raises(ValueError):
            top_k_sparsify(gradients, k=0)

    def test_top_k_k_exceeds_size(self):
        """Test that k >= size returns original array."""
        gradients = np.array([0.1, -0.5, 0.3])
        k = 10

        sparse, mask, ratio = top_k_sparsify(gradients, k)

        np.testing.assert_array_equal(sparse, gradients)
        np.testing.assert_array_equal(mask, np.ones_like(gradients, dtype=bool))
        assert ratio == 1.0

    def test_top_k_multidimensional(self):
        """Test Top-K with multi-dimensional arrays."""
        gradients = np.random.randn(10, 10)
        k = 50

        sparse, mask, ratio = top_k_sparsify(gradients, k)

        # Check that exactly k elements are non-zero
        assert np.sum(mask) == k
        assert np.sum(sparse != 0) == k

        # Check shape preserved
        assert sparse.shape == gradients.shape
        assert mask.shape == gradients.shape

    def test_top_k_compression_ratio_positive(self):
        """Test that compression ratio is positive."""
        gradients = np.random.randn(1000)
        k = 100

        sparse, mask, ratio = top_k_sparsify(gradients, k)

        assert ratio > 1.0  # Should have compression benefit


class TestRandomKSparsify:
    """Tests for Random-K sparsification."""

    def test_random_k_basic(self):
        """Test basic Random-K sparsification."""
        gradients = np.array([0.1, -0.5, 0.3, 0.05, -0.8, 0.7])
        k = 3
        random_state = 42

        sparse, mask, ratio = random_k_sparsify(gradients, k, random_state=random_state)

        # Check that exactly k elements are non-zero
        assert np.sum(mask) == k
        assert np.sum(sparse != 0) == k

    def test_random_k_reproducibility(self):
        """Test that same random_state produces same results."""
        gradients = np.random.randn(100)
        k = 20
        random_state = 42

        sparse1, mask1, _ = random_k_sparsify(gradients, k, random_state=random_state)
        sparse2, mask2, _ = random_k_sparsify(gradients, k, random_state=random_state)

        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(sparse1, sparse2)

    def test_random_k_different_seeds(self):
        """Test that different seeds produce different results."""
        gradients = np.random.randn(100)
        k = 20

        sparse1, mask1, _ = random_k_sparsify(gradients, k, random_state=42)
        sparse2, mask2, _ = random_k_sparsify(gradients, k, random_state=123)

        # Results should be different (with high probability)
        assert not np.array_equal(mask1, mask2)

    def test_random_k_k_equals_zero(self):
        """Test that k=0 raises error."""
        gradients = np.array([0.1, -0.5, 0.3])

        with pytest.raises(ValueError):
            random_k_sparsify(gradients, k=0)


class TestThresholdSparsify:
    """Tests for Threshold-based sparsification."""

    def test_threshold_basic(self):
        """Test basic threshold sparsification."""
        gradients = np.array([0.1, -0.5, 0.3, 0.05, -0.8, 0.7])
        threshold = 0.4

        sparse, mask, ratio = threshold_sparsify(gradients, threshold)

        # Check that only values >= threshold are kept
        for i, (orig, sp) in enumerate(zip(gradients, sparse)):
            if np.abs(orig) >= threshold:
                assert sp == orig
            else:
                assert sp == 0

    def test_threshold_all_zero(self):
        """Test threshold larger than all values."""
        gradients = np.array([0.1, 0.2, 0.3])
        threshold = 1.0

        sparse, mask, ratio = threshold_sparsify(gradients, threshold)

        assert np.all(sparse == 0)
        assert np.all(mask == False)

    def test_threshold_none_zero(self):
        """Test threshold smaller than all values."""
        gradients = np.array([0.1, 0.2, 0.3])
        threshold = 0.05

        sparse, mask, ratio = threshold_sparsify(gradients, threshold)

        np.testing.assert_array_equal(sparse, gradients)
        np.testing.assert_array_equal(mask, np.ones_like(gradients, dtype=bool))

    def test_threshold_negative_raises_error(self):
        """Test that negative threshold raises error."""
        gradients = np.array([0.1, 0.2, 0.3])

        with pytest.raises(ValueError):
            threshold_sparsify(gradients, threshold=-0.1)

    def test_threshold_zero(self):
        """Test threshold=0 keeps all non-zero values."""
        gradients = np.array([0.1, 0.0, -0.2, 0.0, 0.3])
        threshold = 0.0

        sparse, mask, ratio = threshold_sparsify(gradients, threshold)

        # Zero values should be masked
        assert sparse[1] == 0
        assert sparse[3] == 0

        # Non-zero values should be preserved
        assert sparse[0] == 0.1
        assert sparse[2] == -0.2
        assert sparse[4] == 0.3


class TestTopKSparsifyPercentage:
    """Tests for Top-K percentage-based sparsification."""

    def test_percentage_10_percent(self):
        """Test 10% sparsification."""
        gradients = np.random.randn(1000)
        percentage = 10.0

        sparse, mask, ratio = top_k_sparsify_percentage(gradients, percentage)

        # 10% of 1000 = 100 elements
        assert np.sum(mask) == 100

    def test_percentage_100_percent(self):
        """Test 100% sparsification (keeps all)."""
        gradients = np.random.randn(100)
        percentage = 100.0

        sparse, mask, ratio = top_k_sparsify_percentage(gradients, percentage)

        np.testing.assert_array_equal(sparse, gradients)
        assert np.all(mask)

    def test_percentage_0_percent_raises_error(self):
        """Test that 0% keeps at least 1 element."""
        gradients = np.random.randn(100)
        percentage = 0.0

        sparse, mask, ratio = top_k_sparsify_percentage(gradients, percentage)

        # Should keep at least 1 element
        assert np.sum(mask) >= 1

    def test_percentage_invalid(self):
        """Test that invalid percentage raises error."""
        gradients = np.random.randn(100)

        with pytest.raises(ValueError):
            top_k_sparsify_percentage(gradients, percentage=150)

        with pytest.raises(ValueError):
            top_k_sparsify_percentage(gradients, percentage=-10)


class TestReconstructionCorrectness:
    """Tests for reconstruction correctness (lossless for sparsification)."""

    def test_top_k_reconstruction_correctness(self):
        """Test that Top-K reconstruction is lossless for selected elements."""
        gradients = np.random.randn(1000)
        k = 100

        sparse, mask, _ = top_k_sparsify(gradients, k)

        # Selected elements should match original exactly
        np.testing.assert_array_equal(gradients[mask], sparse[mask])

    def test_random_k_reconstruction_correctness(self):
        """Test that Random-K reconstruction is lossless for selected elements."""
        gradients = np.random.randn(1000)
        k = 100

        sparse, mask, _ = random_k_sparsify(gradients, k, random_state=42)

        # Selected elements should match original exactly
        np.testing.assert_array_equal(gradients[mask], sparse[mask])

    def test_threshold_reconstruction_correctness(self):
        """Test that threshold reconstruction is lossless for selected elements."""
        gradients = np.random.randn(1000)
        threshold = 0.5

        sparse, mask, _ = threshold_sparsify(gradients, threshold)

        # Selected elements should match original exactly
        np.testing.assert_array_equal(gradients[mask], sparse[mask])
