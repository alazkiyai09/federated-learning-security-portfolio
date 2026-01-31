"""
Gradient sparsification techniques for communication-efficient FL.

This module implements various sparsification methods that keep only
the most important gradients, reducing communication overhead.
"""

import numpy as np
from typing import Optional, Tuple
from .utils import (
    measure_bytes,
    measure_compressed_bytes,
    calculate_compression_ratio
)


def top_k_sparsify(
    gradients: np.ndarray,
    k: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Keep only the K largest (by magnitude) gradients.

    Uses np.argpartition for O(n) selection without full sorting.

    Args:
        gradients: Input gradient array
        k: Number of elements to keep (absolute count)
        random_state: Random seed for reproducibility (not used in Top-K,
                     but kept for API consistency)

    Returns:
        (sparse_gradients, mask, compression_ratio)
        - sparse_gradients: Gradients with non-top-k elements set to 0
        - mask: Boolean array indicating which elements were kept
        - compression_ratio: original_bytes / compressed_bytes

    Example:
        >>> grads = np.array([0.1, -0.5, 0.3, 0.05, -0.8])
        >>> sparse, mask, ratio = top_k_sparsify(grads, k=2)
        >>> sparse
        array([ 0.  , -0.5 , 0.  ,  0.  , -0.8 ])
        >>> mask
        array([False,  True, False, False,  True])
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= gradients.size:
        # No sparsification needed
        compression_ratio = 1.0
        return gradients.copy(), np.ones_like(gradients, dtype=bool), compression_ratio

    # Flatten for easier indexing
    original_shape = gradients.shape
    flat_gradients = gradients.flatten()

    # Find k largest elements by magnitude
    # Use argpartition for O(n) performance
    magnitudes = np.abs(flat_gradients)
    # argpartition gives us indices of k largest elements (unsorted)
    top_k_indices = np.argpartition(-magnitudes, k)[:k]

    # Create mask
    mask = np.zeros_like(flat_gradients, dtype=bool)
    mask[top_k_indices] = True

    # Apply mask
    sparse_gradients = flat_gradients * mask

    # Reshape to original shape
    sparse_gradients = sparse_gradients.reshape(original_shape)
    mask = mask.reshape(original_shape)

    # Calculate compression ratio
    original_bytes = measure_bytes(gradients)
    compressed_bytes = measure_compressed_bytes(sparse_gradients)
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

    return sparse_gradients, mask, compression_ratio


def random_k_sparsify(
    gradients: np.ndarray,
    k: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Keep K randomly selected gradients (baseline comparison).

    This is a baseline method to compare against Top-K sparsification.
    Random selection ensures unbiased estimates but may require more
    iterations to converge.

    Args:
        gradients: Input gradient array
        k: Number of elements to keep (absolute count)
        random_state: Random seed for reproducibility

    Returns:
        (sparse_gradients, mask, compression_ratio)

    Example:
        >>> grads = np.array([0.1, -0.5, 0.3, 0.05, -0.8])
        >>> sparse, mask, ratio = random_k_sparsify(grads, k=2, random_state=42)
        # Keeps 2 random elements (which 2 depends on random_state)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= gradients.size:
        # No sparsification needed
        compression_ratio = 1.0
        return gradients.copy(), np.ones_like(gradients, dtype=bool), compression_ratio

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Flatten for easier indexing
    original_shape = gradients.shape
    flat_gradients = gradients.flatten()

    # Randomly select k indices
    total_elements = flat_gradients.size
    selected_indices = rng.choice(total_elements, size=k, replace=False)

    # Create mask
    mask = np.zeros_like(flat_gradients, dtype=bool)
    mask[selected_indices] = True

    # Apply mask
    sparse_gradients = flat_gradients * mask

    # Reshape to original shape
    sparse_gradients = sparse_gradients.reshape(original_shape)
    mask = mask.reshape(original_shape)

    # Calculate compression ratio
    original_bytes = measure_bytes(gradients)
    compressed_bytes = measure_compressed_bytes(sparse_gradients)
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

    return sparse_gradients, mask, compression_ratio


def threshold_sparsify(
    gradients: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Keep only gradients whose magnitude exceeds the threshold.

    Unlike Top-K, this method keeps ALL elements above the threshold,
    regardless of count. The compression ratio varies based on how
    many elements exceed the threshold.

    Args:
        gradients: Input gradient array
        threshold: Minimum magnitude to keep (must be >= 0)

    Returns:
        (sparse_gradients, mask, compression_ratio)

    Example:
        >>> grads = np.array([0.1, -0.5, 0.3, 0.05, -0.8])
        >>> sparse, mask, ratio = threshold_sparsify(grads, threshold=0.3)
        >>> sparse
        array([ 0.  , -0.5 , 0.3 ,  0.  , -0.8 ])
        >>> mask
        array([False,  True,  True, False,  True])
    """
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}")

    # Create mask based on magnitude
    mask = np.abs(gradients) >= threshold

    # Apply mask
    sparse_gradients = gradients * mask

    # Calculate compression ratio
    original_bytes = measure_bytes(gradients)
    compressed_bytes = measure_compressed_bytes(sparse_gradients)
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

    return sparse_gradients, mask, compression_ratio


def top_k_sparsify_percentage(
    gradients: np.ndarray,
    percentage: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Keep top K% of gradients by magnitude.

    Convenience wrapper around top_k_sparsify that accepts
    a percentage instead of absolute count.

    Args:
        gradients: Input gradient array
        percentage: Percentage of elements to keep (0-100)
        random_state: Random seed (not used, for API consistency)

    Returns:
        (sparse_gradients, mask, compression_ratio)

    Example:
        >>> grads = np.random.randn(1000)
        >>> sparse, mask, ratio = top_k_sparsify_percentage(grads, percentage=10.0)
        # Keeps top 10% (100 largest gradients)
    """
    if not (0 <= percentage <= 100):
        raise ValueError(f"Percentage must be in [0, 100], got {percentage}")

    k = max(1, int(gradients.size * percentage / 100))
    return top_k_sparsify(gradients, k, random_state)
