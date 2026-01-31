"""
Gradient quantization techniques for communication-efficient FL.

This module implements various quantization methods that reduce
the precision of gradients, reducing communication overhead.
"""

import numpy as np
from typing import Optional, Tuple
from .utils import (
    measure_bytes,
    measure_compressed_bytes,
    calculate_compression_ratio
)


def quantize_8bit(
    gradients: np.ndarray,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Quantize gradients to 8-bit unsigned integers [0, 255].

    Uses linear scaling to map gradients from [min, max] to [0, 255].
    This is a uniform quantization scheme with deterministic rounding.

    Args:
        gradients: Input gradient array (typically float32/float64)
        random_state: Random seed (not used in uniform quantization,
                     but kept for API consistency)

    Returns:
        (quantized, (min_val, max_val), compression_ratio)
        - quantized: Quantized array (uint8)
        - (min_val, max_val): Min/max values for dequantization
        - compression_ratio: original_bytes / compressed_bytes

    Example:
        >>> grads = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> quantized, (mn, mx), ratio = quantize_8bit(grads)
        >>> quantized
        array([  0,  64, 128, 191, 255], dtype=uint8)
        >>> (mn, mx)
        (-1.0, 1.0)
    """
    if gradients.size == 0:
        return np.array([], dtype=np.uint8), (0.0, 0.0), 1.0

    # Find min and max values
    min_val = np.min(gradients)
    max_val = np.max(gradients)

    # Handle edge case where all values are the same
    if min_val == max_val:
        # All zeros or all same value
        quantized = np.zeros_like(gradients, dtype=np.uint8)
    else:
        # Scale to [0, 255]
        # Formula: q = (x - min) / (max - min) * 255
        scale = max_val - min_val
        quantized = np.round((gradients - min_val) / scale * 255).astype(np.uint8)

    # Calculate compression ratio
    original_bytes = measure_bytes(gradients)
    compressed_bytes = measure_bytes(quantized)
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

    return quantized, (min_val, max_val), compression_ratio


def dequantize_8bit(
    quantized: np.ndarray,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """
    Dequantize 8-bit integers back to float gradients.

    Args:
        quantized: Quantized array (uint8)
        min_val: Minimum value from original gradients
        max_val: Maximum value from original gradients

    Returns:
        Dequantized gradient array (float32)

    Example:
        >>> quantized = np.array([0, 64, 128, 191, 255], dtype=np.uint8)
        >>> grads = dequantize_8bit(quantized, -1.0, 1.0)
        >>> grads  # Approximately [-1.0, -0.5, 0.0, 0.5, 1.0]
        array([-1.   , -0.496,  0.   ,  0.504,  1.   ])
    """
    if min_val == max_val:
        return np.zeros_like(quantized, dtype=np.float32)

    # Scale back from [0, 255] to [min, max]
    # Formula: x = q / 255 * (max - min) + min
    scale = max_val - min_val
    dequantized = (quantized.astype(np.float32) / 255.0 * scale) + min_val

    return dequantized


def quantize_4bit(
    gradients: np.ndarray,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Quantize gradients to 4-bit unsigned integers [0, 15].

    Aggressive quantization for maximum compression. Higher
    accuracy loss compared to 8-bit quantization.

    Args:
        gradients: Input gradient array
        random_state: Random seed (not used in uniform quantization)

    Returns:
        (quantized, (min_val, max_val), compression_ratio)
        - quantized: Quantized array (uint8, but only 4 bits used)
        - (min_val, max_val): Min/max values for dequantization
        - compression_ratio: original_bytes / compressed_bytes

    Note:
        We use uint8 to store 4-bit values (wastes 4 bits per element,
        but numpy doesn't have native uint4). In production, you would
        pack two 4-bit values into a single byte.
    """
    if gradients.size == 0:
        return np.array([], dtype=np.uint8), (0.0, 0.0), 1.0

    # Find min and max values
    min_val = np.min(gradients)
    max_val = np.max(gradients)

    # Handle edge case where all values are the same
    if min_val == max_val:
        quantized = np.zeros_like(gradients, dtype=np.uint8)
    else:
        # Scale to [0, 15]
        scale = max_val - min_val
        quantized = np.round((gradients - min_val) / scale * 15).astype(np.uint8)

    # Calculate compression ratio
    # Note: We report 8x compression (32 bits -> 4 bits), though
    # actual implementation uses uint8 (2x compression)
    original_bytes = measure_bytes(gradients)
    theoretical_compressed_bytes = gradients.size * 0.5  # 4 bits per element
    compression_ratio = calculate_compression_ratio(
        original_bytes,
        theoretical_compressed_bytes
    )

    return quantized, (min_val, max_val), compression_ratio


def dequantize_4bit(
    quantized: np.ndarray,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """
    Dequantize 4-bit integers back to float gradients.

    Args:
        quantized: Quantized array (uint8, values 0-15)
        min_val: Minimum value from original gradients
        max_val: Maximum value from original gradients

    Returns:
        Dequantized gradient array (float32)
    """
    if min_val == max_val:
        return np.zeros_like(quantized, dtype=np.float32)

    # Scale back from [0, 15] to [min, max]
    scale = max_val - min_val
    dequantized = (quantized.astype(np.float32) / 15.0 * scale) + min_val

    return dequantized


def stochastic_quantize(
    gradients: np.ndarray,
    bits: int = 8,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Stochastic quantization with probabilistic rounding.

    Instead of deterministic rounding, this method uses randomized
    rounding where values are rounded up with probability proportional
    to their fractional part. This provides unbiased estimates.

    Args:
        gradients: Input gradient array
        bits: Number of bits for quantization (1-32)
        random_state: Random seed for reproducibility

    Returns:
        (quantized, (min_val, max_val), compression_ratio)

    Example:
        >>> grads = np.array([0.1, 0.2, 0.3])
        >>> quantized, (mn, mx), ratio = stochastic_quantize(grads, bits=3, random_state=42)
        # With 3 bits, values are in [0, 7]
    """
    if not (1 <= bits <= 32):
        raise ValueError(f"Bits must be in [1, 32], got {bits}")

    if gradients.size == 0:
        levels = 2 ** bits - 1
        return np.array([], dtype=np.uint8), (0.0, 0.0), 1.0

    rng = np.random.default_rng(random_state)

    # Find min and max values
    min_val = np.min(gradients)
    max_val = np.max(gradients)

    if min_val == max_val:
        quantized = np.zeros_like(gradients, dtype=np.uint8)
    else:
        # Scale to [0, 2^bits - 1]
        levels = 2 ** bits - 1
        scale = max_val - min_val
        scaled = (gradients - min_val) / scale * levels

        # Stochastic rounding: round to nearest integer with probability
        # For each value, floor(value) + (Bernoulli(frac) ? 1 : 0)
        floor_val = np.floor(scaled)
        frac = scaled - floor_val
        random_samples = rng.random(scaled.shape)
        quantized = (floor_val + (random_samples < frac).astype(np.float32))
        quantized = np.clip(quantized, 0, levels).astype(np.uint8)

    # Calculate compression ratio
    original_bytes = measure_bytes(gradients)
    compressed_bytes = gradients.size * (bits / 8)
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

    return quantized, (min_val, max_val), compression_ratio


def dequantize_stochastic(
    quantized: np.ndarray,
    min_val: float,
    max_val: float,
    bits: int
) -> np.ndarray:
    """
    Dequantize stochastic quantized values back to float gradients.

    Args:
        quantized: Quantized array (uint8)
        min_val: Minimum value from original gradients
        max_val: Maximum value from original gradients
        bits: Number of bits used for quantization

    Returns:
        Dequantized gradient array (float32)
    """
    if min_val == max_val:
        return np.zeros_like(quantized, dtype=np.float32)

    levels = 2 ** bits - 1
    scale = max_val - min_val
    dequantized = (quantized.astype(np.float32) / levels * scale) + min_val

    return dequantized
