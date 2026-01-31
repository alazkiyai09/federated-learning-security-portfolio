"""
Utility functions for measuring compression in federated learning.

This module provides functions to accurately measure bytes transmitted
before and after compression, which is critical for evaluating
communication efficiency in FL systems.
"""

import numpy as np
from typing import Any
import zlib


def measure_bytes(array: np.ndarray) -> int:
    """
    Measure the exact number of bytes used to store a numpy array.

    Args:
        array: Input numpy array

    Returns:
        Exact bytes in memory (using array.nbytes)

    Example:
        >>> arr = np.random.randn(1000, 100)
        >>> measure_bytes(arr)
        800000  # 1000 * 100 * 8 bytes (float64)
    """
    return array.nbytes


def measure_compressed_bytes(
    sparse_array: np.ndarray,
    use_zlib: bool = False,
    compression_level: int = 6
) -> int:
    """
    Measure bytes after compression (sparse or quantized).

    For sparse arrays (many zeros), we can use run-length encoding or
    simply transmit non-zero values with indices. For quantized arrays,
    we transmit the quantized integers.

    Args:
        sparse_array: Array after sparsification/quantization
        use_zlib: Whether to apply additional zlib compression
        compression_level: zlib compression level (0-9, default 6)

    Returns:
        Bytes transmitted after compression

    Note:
        This is a simplified measurement. In production, you would need
        to account for serialization overhead (protocol buffers, etc.)
    """
    # For sparse arrays, we need to store:
    # 1. Non-zero values
    # 2. Their indices
    non_zero_mask = sparse_array != 0
    non_zero_count = np.sum(non_zero_mask)

    if non_zero_count == 0:
        return 0

    # Calculate bytes for values and indices
    values_bytes = sparse_array[non_zero_mask].nbytes

    # For indices, we need to know the original shape
    # Use uint32 for indices (up to 4 billion elements)
    indices = np.where(non_zero_mask)[0]
    indices_bytes = indices.astype(np.uint32).nbytes

    total_bytes = values_bytes + indices_bytes

    # Apply additional zlib compression if requested
    if use_zlib:
        # Serialize to bytes and compress
        data_bytes = sparse_array.tobytes()
        compressed = zlib.compress(data_bytes, level=compression_level)
        return len(compressed)

    return total_bytes


def calculate_compression_ratio(
    original_bytes: int,
    compressed_bytes: int
) -> float:
    """
    Calculate compression ratio.

    Args:
        original_bytes: Bytes before compression
        compressed_bytes: Bytes after compression

    Returns:
        Compression ratio (original / compressed).
        Higher values = better compression.

    Example:
        >>> calculate_compression_ratio(1000, 100)
        10.0  # 10x compression
    """
    if compressed_bytes == 0:
        return float('inf')
    return original_bytes / compressed_bytes


def calculate_bandwidth_savings(
    original_bytes: int,
    compressed_bytes: int
) -> float:
    """
    Calculate percentage bandwidth savings.

    Args:
        original_bytes: Bytes before compression
        compressed_bytes: Bytes after compression

    Returns:
        Percentage savings (0-100).
        Higher values = more bandwidth saved.

    Example:
        >>> calculate_bandwidth_savings(1000, 200)
        80.0  # 80% bandwidth savings
    """
    if original_bytes == 0:
        return 0.0
    return (1 - compressed_bytes / original_bytes) * 100


def estimate_transmission_time(
    bytes_to_transmit: int,
    bandwidth_mbps: float = 100.0
) -> float:
    """
    Estimate transmission time given bandwidth.

    Args:
        bytes_to_transmit: Number of bytes to send
        bandwidth_mbps: Network bandwidth in Mbps (default: 100 Mbps)

    Returns:
        Estimated transmission time in seconds

    Example:
        >>> estimate_transmission_time(1_000_000, bandwidth_mbps=100)
        0.08  # ~80ms to transmit 1MB on 100Mbps link
    """
    # Convert Mbps to bytes per second
    bytes_per_second = (bandwidth_mbps * 1_000_000) / 8
    return bytes_to_transmit / bytes_per_second
