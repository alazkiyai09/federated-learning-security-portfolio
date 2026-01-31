"""
Unit tests for gradient quantization techniques.

Tests verify correctness of quantization/dequantization cycles,
measurement of quantization error, and compression ratio calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.quantizers import (
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    stochastic_quantize,
    dequantize_stochastic
)


class Test8BitQuantization:
    """Tests for 8-bit uniform quantization."""

    def test_quantize_8bit_basic(self):
        """Test basic 8-bit quantization."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        quantized, (min_val, max_val), ratio = quantize_8bit(gradients)

        # Check dtype
        assert quantized.dtype == np.uint8

        # Check range
        assert quantized.min() >= 0
        assert quantized.max() <= 255

        # Check min/max values
        assert min_val == -1.0
        assert max_val == 1.0

    def test_quantize_dequantize_8bit(self):
        """Test quantization-dequantization cycle."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        quantized, (min_val, max_val), _ = quantize_8bit(gradients)
        dequantized = dequantize_8bit(quantized, min_val, max_val)

        # Check shape
        assert dequantized.shape == gradients.shape

        # Check that values are approximately correct (within quantization error)
        np.testing.assert_allclose(dequantized, gradients, atol=0.01)

    def test_quantize_8bit_all_zeros(self):
        """Test quantization when all values are the same."""
        gradients = np.zeros(100)

        quantized, (min_val, max_val), ratio = quantize_8bit(gradients)

        # All values should be quantized to 0
        assert np.all(quantized == 0)

    def test_quantize_8bit_compression_ratio(self):
        """Test that 8-bit quantization achieves 4x compression."""
        gradients = np.random.randn(1000).astype(np.float32)

        quantized, _, ratio = quantize_8bit(gradients)

        # Float32 = 4 bytes, uint8 = 1 byte, so 4x compression
        assert ratio == 4.0

    def test_quantize_8bit_asymmetric_range(self):
        """Test quantization with asymmetric range."""
        gradients = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])

        quantized, (min_val, max_val), _ = quantize_8bit(gradients)

        assert min_val == -2.0
        assert max_val == 3.0

        dequantized = dequantize_8bit(quantized, min_val, max_val)
        np.testing.assert_allclose(dequantized, gradients, atol=0.02)


class Test4BitQuantization:
    """Tests for 4-bit uniform quantization."""

    def test_quantize_4bit_basic(self):
        """Test basic 4-bit quantization."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        quantized, (min_val, max_val), ratio = quantize_4bit(gradients)

        # Check dtype (we use uint8 but only values 0-15 are valid)
        assert quantized.dtype == np.uint8

        # Check range
        assert quantized.min() >= 0
        assert quantized.max() <= 15

    def test_quantize_dequantize_4bit(self):
        """Test quantization-dequantization cycle for 4-bit."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        quantized, (min_val, max_val), _ = quantize_4bit(gradients)
        dequantized = dequantize_4bit(quantized, min_val, max_val)

        # Check shape
        assert dequantized.shape == gradients.shape

        # 4-bit has more error, so we use larger tolerance
        np.testing.assert_allclose(dequantized, gradients, atol=0.1)

    def test_quantize_4bit_all_zeros(self):
        """Test 4-bit quantization when all values are the same."""
        gradients = np.zeros(100)

        quantized, (min_val, max_val), ratio = quantize_4bit(gradients)

        assert np.all(quantized == 0)

    def test_quantize_4bit_compression_ratio(self):
        """Test that 4-bit quantization achieves 8x compression."""
        gradients = np.random.randn(1000).astype(np.float32)

        quantized, _, ratio = quantize_4bit(gradients)

        # Float32 = 4 bytes, 4-bit = 0.5 byte, so 8x compression
        assert ratio == 8.0

    def test_4bit_more_error_than_8bit(self):
        """Test that 4-bit has more error than 8-bit."""
        gradients = np.random.randn(1000)

        # Quantize to 8-bit
        q8, (min8, max8), _ = quantize_8bit(gradients)
        dq8 = dequantize_8bit(q8, min8, max8)
        error_8bit = np.mean((gradients - dq8) ** 2)

        # Quantize to 4-bit
        q4, (min4, max4), _ = quantize_4bit(gradients)
        dq4 = dequantize_4bit(q4, min4, max4)
        error_4bit = np.mean((gradients - dq4) ** 2)

        # 4-bit should have more error
        assert error_4bit > error_8bit


class TestStochasticQuantization:
    """Tests for stochastic quantization."""

    def test_stochastic_quantize_basic(self):
        """Test basic stochastic quantization."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        bits = 3
        random_state = 42

        quantized, (min_val, max_val), ratio = stochastic_quantize(
            gradients, bits=bits, random_state=random_state
        )

        # Check range (3 bits = 0-7)
        assert quantized.min() >= 0
        assert quantized.max() <= 7

    def test_stochastic_quantize_dequantize(self):
        """Test stochastic quantization-dequantization cycle."""
        gradients = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        bits = 4
        random_state = 42

        quantized, (min_val, max_val), _ = stochastic_quantize(
            gradients, bits=bits, random_state=random_state
        )
        dequantized = dequantize_stochastic(quantized, min_val, max_val, bits)

        # Check shape
        assert dequantized.shape == gradients.shape

        # Stochastic rounding is approximate
        np.testing.assert_allclose(dequantized, gradients, atol=0.1)

    def test_stochastic_quantize_reproducibility(self):
        """Test that same random_state produces same results."""
        gradients = np.random.randn(100)
        bits = 4

        q1, _, _ = stochastic_quantize(gradients, bits=bits, random_state=42)
        q2, _, _ = stochastic_quantize(gradients, bits=bits, random_state=42)

        np.testing.assert_array_equal(q1, q2)

    def test_stochastic_quantize_different_seeds(self):
        """Test that different seeds produce different results."""
        gradients = np.random.randn(100)
        bits = 4

        q1, _, _ = stochastic_quantize(gradients, bits=bits, random_state=42)
        q2, _, _ = stochastic_quantize(gradients, bits=bits, random_state=123)

        # Results should be different (with high probability)
        assert not np.array_equal(q1, q2)

    def test_stochastic_quantize_invalid_bits(self):
        """Test that invalid bits raises error."""
        gradients = np.random.randn(100)

        with pytest.raises(ValueError):
            stochastic_quantize(gradients, bits=0)

        with pytest.raises(ValueError):
            stochastic_quantize(gradients, bits=33)


class TestQuantizationError:
    """Tests for quantization error measurement."""

    def test_max_error_8bit(self):
        """Test maximum quantization error for 8-bit."""
        # Create gradients with known range
        gradients = np.linspace(-1, 1, 1000)

        quantized, (min_val, max_val), _ = quantize_8bit(gradients)
        dequantized = dequantize_8bit(quantized, min_val, max_val)

        max_error = np.max(np.abs(gradients - dequantized))

        # For 8-bit in range [-1, 1], max error should be small
        assert max_error < 0.01

    def test_max_error_4bit(self):
        """Test maximum quantization error for 4-bit."""
        gradients = np.linspace(-1, 1, 1000)

        quantized, (min_val, max_val), _ = quantize_4bit(gradients)
        dequantized = dequantize_4bit(quantized, min_val, max_val)

        max_error = np.max(np.abs(gradients - dequantized))

        # For 4-bit in range [-1, 1], max error should be larger than 8-bit
        assert max_error < 0.2  # 4-bit allows more error

    def test_mse_scales_with_range(self):
        """Test that MSE scales with gradient range."""
        # Small range
        gradients_small = np.linspace(-0.1, 0.1, 1000)
        q_small, (min_s, max_s), _ = quantize_8bit(gradients_small)
        dq_small = dequantize_8bit(q_small, min_s, max_s)
        mse_small = np.mean((gradients_small - dq_small) ** 2)

        # Large range
        gradients_large = np.linspace(-10, 10, 1000)
        q_large, (min_l, max_l), _ = quantize_8bit(gradients_large)
        dq_large = dequantize_8bit(q_large, min_l, max_l)
        mse_large = np.mean((gradients_large - dq_large) ** 2)

        # MSE should scale with range
        assert mse_large > mse_small

    def test_unbiased_stochastic_rounding(self):
        """Test that stochastic rounding is unbiased."""
        # Create values in a range to test stochastic rounding
        # Use range [-1, 1] with 2-bit quantization (4 levels)
        gradients = np.random.uniform(-1, 1, 10000)

        bits = 2
        # Fix random seed for reproducibility
        quantized, (min_val, max_val), _ = stochastic_quantize(
            gradients, bits=bits, random_state=42
        )
        dequantized = dequantize_stochastic(quantized, min_val, max_val, bits)

        # Mean should be close to original mean (unbiased)
        original_mean = np.mean(gradients)
        dequantized_mean = np.mean(dequantized)
        np.testing.assert_allclose(dequantized_mean, original_mean, atol=0.05)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Test quantization of empty array."""
        gradients = np.array([])

        quantized, (min_val, max_val), ratio = quantize_8bit(gradients)

        assert quantized.size == 0

    def test_single_element(self):
        """Test quantization of single element."""
        gradients = np.array([0.5])

        quantized, (min_val, max_val), ratio = quantize_8bit(gradients)

        assert quantized.size == 1
        assert min_val == max_val == 0.5

    def test_nan_handling(self):
        """Test that NaN values are handled."""
        gradients = np.array([0.1, np.nan, 0.3])

        # Should not crash, but behavior is undefined
        quantized, _, _ = quantize_8bit(gradients)
        assert quantized.size == 3

    def test_inf_handling(self):
        """Test that Inf values are handled."""
        gradients = np.array([0.1, np.inf, 0.3])

        # Should not crash
        quantized, _, _ = quantize_8bit(gradients)
        assert quantized.size == 3
