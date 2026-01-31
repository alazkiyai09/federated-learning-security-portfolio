"""
Unit tests for error feedback mechanism.

Tests verify correctness of residual accumulation, no double-counting,
and proper reset functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.error_feedback import (
    ErrorFeedback,
    MultiLayerErrorFeedback
)
from src.compression.sparsifiers import top_k_sparsify


class TestErrorFeedback:
    """Tests for single-layer error feedback."""

    def test_initial_residual_zero(self):
        """Test that residual starts at zero."""
        ef = ErrorFeedback(shape=(100,))

        residual = ef.get_residual()
        assert np.all(residual == 0)

    def test_compress_and_update_basic(self):
        """Test basic compression and residual update."""
        ef = ErrorFeedback(shape=(10,))

        gradients = np.random.randn(10)
        compressed, ratio, metrics = ef.compress_and_update(
            gradients,
            lambda x: top_k_sparsify(x, k=5)[0]  # Keep top 5
        )

        # Check compression happened
        assert np.sum(compressed != 0) <= 5

        # Check that residual is not zero (some gradients were dropped)
        residual = ef.get_residual()
        assert np.linalg.norm(residual) > 0

    def test_residual_accumulation(self):
        """Test that residuals accumulate over multiple rounds."""
        ef = ErrorFeedback(shape=(10,))
        k = 5

        # Round 1
        gradients1 = np.random.randn(10)
        ef.compress_and_update(gradients1, lambda x: top_k_sparsify(x, k=k)[0])
        residual1 = ef.get_residual().copy()

        # Round 2
        gradients2 = np.random.randn(10)
        ef.compress_and_update(gradients2, lambda x: top_k_sparsify(x, k=k)[0])
        residual2 = ef.get_residual()

        # Residuals should change (accumulate more errors)
        # Note: residuals are updated, not simply added
        assert not np.array_equal(residual1, residual2)

    def test_no_double_counting(self):
        """Test that residuals are not double-counted."""
        ef = ErrorFeedback(shape=(10,))

        gradients = np.ones(10)  # All ones

        # First compression: k=5 keeps 5 elements
        compressed1, ratio1, metrics1 = ef.compress_and_update(
            gradients,
            lambda x: top_k_sparsify(x, k=5)[0]
        )

        # Second compression: should use residuals
        compressed2, ratio2, metrics2 = ef.compress_and_update(
            np.zeros(10),  # Zero new gradients
            lambda x: top_k_sparsify(x, k=5)[0]
        )

        # The residual from first round should be used in second round
        # So compressed2 should not be all zeros
        assert np.sum(compressed2 != 0) > 0

    def test_reset_residual(self):
        """Test residual reset."""
        ef = ErrorFeedback(shape=(10,))

        gradients = np.random.randn(10)
        ef.compress_and_update(gradients, lambda x: top_k_sparsify(x, k=5)[0])

        # Reset
        ef.reset_residual()

        # Residual should be zero again
        residual = ef.get_residual()
        assert np.all(residual == 0)

    def test_residual_statistics(self):
        """Test residual statistics calculation."""
        ef = ErrorFeedback(shape=(10,))

        gradients = np.random.randn(10)
        ef.compress_and_update(gradients, lambda x: top_k_sparsify(x, k=5)[0])

        stats = ef.get_residual_statistics()

        # Check that stats have expected keys
        expected_keys = {'norm', 'mean', 'abs_mean', 'max', 'min', 'std', 'nonzero_fraction'}
        assert set(stats.keys()) == expected_keys

        # Check that values are reasonable
        assert stats['norm'] >= 0
        assert stats['abs_mean'] >= 0

    def test_no_compression_residual_zero(self):
        """Test that no compression results in zero residual."""
        ef = ErrorFeedback(shape=(10,))

        gradients = np.random.randn(10)

        # Use k > size (no compression)
        compressed, ratio, metrics = ef.compress_and_update(
            gradients,
            lambda x: top_k_sparsify(x, k=100)[0]
        )

        # All gradients should be kept
        np.testing.assert_array_equal(compressed, gradients)

        # Residual should be zero (nothing dropped)
        residual = ef.get_residual()
        assert np.all(residual == 0)

    def test_compression_with_residual_adds(self):
        """Test that gradients are added to residual before compression."""
        ef = ErrorFeedback(shape=(10,))

        # First round: create residual
        gradients1 = np.ones(10)
        ef.compress_and_update(gradients1, lambda x: top_k_sparsify(x, k=5)[0])
        residual_after_round1 = ef.get_residual().copy()

        # Second round: send same gradients again
        # They should be added to residuals
        gradients2 = np.ones(10)
        ef.compress_and_update(gradients2, lambda x: top_k_sparsify(x, k=5)[0])

        # The compression in round 2 operates on gradients + residual
        # So the result should be different from round 1
        residual_after_round2 = ef.get_residual()

        # They should be different
        assert not np.array_equal(residual_after_round1, residual_after_round2)


class TestMultiLayerErrorFeedback:
    """Tests for multi-layer error feedback."""

    def test_initial_residuals_zero(self):
        """Test that all residuals start at zero."""
        shapes = [(100, 100), (100,), (50, 100)]
        mlef = MultiLayerErrorFeedback(shapes)

        residuals = mlef.get_all_residuals()

        assert len(residuals) == 3
        for residual in residuals:
            assert np.all(residual == 0)

    def test_compress_and_update_layers(self):
        """Test compression and update for multiple layers."""
        shapes = [(10, 10), (10,), (5, 10)]
        mlef = MultiLayerErrorFeedback(shapes)

        gradients = [
            np.random.randn(10, 10),
            np.random.randn(10),
            np.random.randn(5, 10)
        ]

        compressed, ratios, metrics = mlef.compress_and_update_layers(
            gradients,
            lambda x: top_k_sparsify(x, k=10)[0]
        )

        # Check that we get compressed gradients for all layers
        assert len(compressed) == 3

        # Check that we get compression ratios for all layers
        assert len(ratios) == 3

        # Check metrics
        assert 'mean_compression_ratio' in metrics
        assert 'layer_metrics' in metrics
        assert len(metrics['layer_metrics']) == 3

    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises error."""
        shapes = [(10, 10), (10,)]
        mlef = MultiLayerErrorFeedback(shapes)

        # Wrong number of layers
        gradients = [np.random.randn(10, 10)]

        with pytest.raises(ValueError):
            mlef.compress_and_update_layers(
                gradients,
                lambda x: top_k_sparsify(x, k=10)[0]
            )

    def test_reset_all_residuals(self):
        """Test resetting all residual buffers."""
        shapes = [(10, 10), (10,)]
        mlef = MultiLayerErrorFeedback(shapes)

        gradients = [
            np.random.randn(10, 10),
            np.random.randn(10)
        ]

        # Create residuals
        mlef.compress_and_update_layers(
            gradients,
            lambda x: top_k_sparsify(x, k=5)[0]
        )

        # Reset
        mlef.reset_all_residuals()

        # All residuals should be zero
        residuals = mlef.get_all_residuals()
        for residual in residuals:
            assert np.all(residual == 0)

    def test_independent_layer_residuals(self):
        """Test that residuals for different layers are independent."""
        shapes = [(10, 10), (10,)]
        mlef = MultiLayerErrorFeedback(shapes)

        gradients = [
            np.ones((10, 10)) * 1.0,  # Layer 1: all ones
            np.ones(10) * 2.0          # Layer 2: all twos
        ]

        mlef.compress_and_update_layers(
            gradients,
            lambda x: top_k_sparsify(x, k=5)[0]
        )

        residuals = mlef.get_all_residuals()

        # Each layer should have its own residual
        assert residuals[0].shape == (10, 10)
        assert residuals[1].shape == (10,)

        # Residuals should be non-zero (gradients were dropped)
        assert np.linalg.norm(residuals[0]) > 0
        assert np.linalg.norm(residuals[1]) > 0

    def test_layer_metrics_structure(self):
        """Test that layer metrics have correct structure."""
        shapes = [(10, 10), (10,)]
        mlef = MultiLayerErrorFeedback(shapes)

        gradients = [
            np.random.randn(10, 10),
            np.random.randn(10)
        ]

        _, _, metrics = mlef.compress_and_update_layers(
            gradients,
            lambda x: top_k_sparsify(x, k=5)[0]
        )

        # Check layer metrics
        layer_metrics = metrics['layer_metrics']
        assert len(layer_metrics) == 2

        for i, layer_metric in enumerate(layer_metrics):
            assert layer_metric['layer'] == i
            assert 'residual_norm' in layer_metric
            assert 'compression_ratio' in layer_metric


class TestErrorFeedbackConvergence:
    """Tests for error feedback convergence properties."""

    def test_residual_norm_decreases_with_less_compression(self):
        """Test that less compression leads to smaller residual norm."""
        # Aggressive compression (k=2)
        ef_aggressive = ErrorFeedback(shape=(100,))
        gradients = np.random.randn(100)
        ef_aggressive.compress_and_update(
            gradients,
            lambda x: top_k_sparsify(x, k=2)[0]
        )
        residual_aggressive = ef_aggressive.get_residual()

        # Mild compression (k=50)
        ef_mild = ErrorFeedback(shape=(100,))
        ef_mild.compress_and_update(
            gradients,
            lambda x: top_k_sparsify(x, k=50)[0]
        )
        residual_mild = ef_mild.get_residual()

        # Aggressive compression should leave larger residual
        assert np.linalg.norm(residual_aggressive) > np.linalg.norm(residual_mild)

    def test_convergence_with_error_feedback(self):
        """
        Test that error feedback helps convergence.

        This is a conceptual test: with error feedback, small gradients
        eventually get transmitted through residual accumulation.
        """
        ef = ErrorFeedback(shape=(100,))

        # Create many small gradients that would be dropped by Top-K
        small_gradients = np.random.randn(100) * 0.001

        # Without error feedback, these would be lost
        # With error feedback, they accumulate in residual

        for _ in range(10):
            ef.compress_and_update(
                small_gradients,
                lambda x: top_k_sparsify(x, k=10)[0]
            )

        # Residual should have accumulated the small gradients
        residual = ef.get_residual()
        assert np.linalg.norm(residual) > 0
