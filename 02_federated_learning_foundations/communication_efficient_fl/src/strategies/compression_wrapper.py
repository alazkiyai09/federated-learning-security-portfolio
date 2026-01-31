"""
Wrapper to apply compression to Flower FL parameters.

This module provides utilities to integrate compression techniques
with Flower's parameter serialization/deserialization.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Any
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays


def apply_compression(
    parameters: list[np.ndarray],
    compress_func: Callable,
    **compress_kwargs
) -> Tuple[list[np.ndarray], dict]:
    """
    Apply compression function to all parameter layers.

    Args:
        parameters: List of numpy arrays (model parameters/gradients)
        compress_func: Compression function to apply
        **compress_kwargs: Additional arguments for compress_func

    Returns:
        (compressed_parameters, metrics)
        - compressed_parameters: List of compressed parameters
        - metrics: Dict with compression metrics

    Example:
        >>> params = [np.random.randn(100, 100), np.random.randn(100)]
        >>> compressed, metrics = apply_compression(
        ...     params,
        ...     lambda x: top_k_sparsify(x, k=1000)[0]
        ... )
        >>> print(metrics['mean_compression_ratio'])
    """
    compressed_parameters = []
    compression_ratios = []

    for i, param in enumerate(parameters):
        result = compress_func(param, **compress_kwargs)

        # Handle different return types
        if isinstance(result, tuple):
            compressed = result[0]
            # Compression ratio is usually the last element
            ratio = result[-1] if len(result) > 1 else 1.0
        else:
            compressed = result
            ratio = 1.0

        compressed_parameters.append(compressed)
        compression_ratios.append(ratio)

    metrics = {
        'mean_compression_ratio': float(np.mean(compression_ratios)),
        'compression_ratios': compression_ratios,
        'num_layers': len(parameters)
    }

    return compressed_parameters, metrics


class CompressionWrapper:
    """
    Wrapper class to apply compression to Flower parameters.

    Integrates with Flower's parameter serialization for seamless
    communication-efficient federated learning.

    Example:
        >>> wrapper = CompressionWrapper(
        ...     compress_func='top_k',
        ...     k=1000
        ... )
        >>> # Apply to Flower parameters object
        >>> compressed_params = wrapper.compress_parameters(flower_params)
    """

    def __init__(
        self,
        compress_func: Optional[str] = None,
        **compress_kwargs
    ):
        """
        Initialize compression wrapper.

        Args:
            compress_func: Name of compression function ('top_k', 'random_k',
                           'threshold', 'quantize_8bit', 'quantize_4bit', etc.)
            **compress_kwargs: Additional arguments for compression function
        """
        self.compress_func_name = compress_func
        self.compress_kwargs = compress_kwargs
        self.compress_func = self._get_compress_func(compress_func)

        # For tracking compression metrics
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0

    def _get_compress_func(self, func_name: Optional[str]) -> Optional[Callable]:
        """
        Get compression function by name.

        Args:
            func_name: Name of the compression function

        Returns:
            Compression function or None
        """
        if func_name is None:
            return None

        # Import compression functions
        from ..compression.sparsifiers import (
            top_k_sparsify,
            random_k_sparsify,
            threshold_sparsify
        )
        from ..compression.quantizers import (
            quantize_8bit,
            quantize_4bit,
            stochastic_quantize
        )

        functions = {
            'top_k': top_k_sparsify,
            'random_k': random_k_sparsify,
            'threshold': threshold_sparsify,
            'quantize_8bit': quantize_8bit,
            'quantize_4bit': quantize_4bit,
            'stochastic_quantize': stochastic_quantize,
        }

        return functions.get(func_name)

    def compress_parameters(
        self,
        parameters: Parameters
    ) -> Tuple[Parameters, dict]:
        """
        Compress Flower parameters object.

        Args:
            parameters: Flower Parameters object

        Returns:
            (compressed_parameters, metrics)
            - compressed_parameters: Compressed Parameters object
            - metrics: Compression metrics

        Example:
            >>> wrapper = CompressionWrapper(compress_func='top_k', k=1000)
            >>> compressed_params, metrics = wrapper.compress_parameters(flower_params)
            >>> print(f"Compression ratio: {metrics['compression_ratio']}")
        """
        if self.compress_func is None:
            return parameters, {'compression_ratio': 1.0}

        # Convert to numpy arrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Track original bytes
        original_bytes = sum(arr.nbytes for arr in ndarrays)
        self.total_bytes_original += original_bytes

        # Apply compression
        compressed_ndarrays, metrics = apply_compression(
            ndarrays,
            self.compress_func,
            **self.compress_kwargs
        )

        # Track compressed bytes
        compressed_bytes = sum(arr.nbytes for arr in compressed_ndarrays)
        self.total_bytes_compressed += compressed_bytes

        # Convert back to Flower parameters
        compressed_parameters = ndarrays_to_parameters(compressed_ndarrays)

        # Update metrics
        metrics.update({
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'cumulative_original_bytes': self.total_bytes_original,
            'cumulative_compressed_bytes': self.total_bytes_compressed,
            'cumulative_compression_ratio': (
                self.total_bytes_original / self.total_bytes_compressed
                if self.total_bytes_compressed > 0 else float('inf')
            )
        })

        return compressed_parameters, metrics

    def decompress_parameters(
        self,
        parameters: Parameters
    ) -> Parameters:
        """
        Decompress parameters (placeholder for future expansion).

        Currently, sparsification uses zero-filled values, so no
        decompression is needed. For quantization, this would
        apply dequantization.

        Args:
            parameters: Compressed Parameters object

        Returns:
            Decompressed Parameters object

        Note:
            For quantization methods, this would call dequantize_* functions.
            For sparsification, zero-filled values are used directly.
        """
        # Placeholder: future implementation would handle quantization
        return parameters

    def reset_metrics(self) -> None:
        """Reset cumulative byte tracking."""
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0

    def get_compression_stats(self) -> dict:
        """
        Get cumulative compression statistics.

        Returns:
            Dict with compression statistics

        Example:
            >>> stats = wrapper.get_compression_stats()
            >>> print(f"Total bytes saved: {stats['bytes_saved']}")
        """
        return {
            'total_bytes_original': self.total_bytes_original,
            'total_bytes_compressed': self.total_bytes_compressed,
            'bytes_saved': self.total_bytes_original - self.total_bytes_compressed,
            'compression_ratio': (
                self.total_bytes_original / self.total_bytes_compressed
                if self.total_bytes_compressed > 0 else float('inf')
            ),
            'bandwidth_savings_pct': (
                (1 - self.total_bytes_compressed / self.total_bytes_original) * 100
                if self.total_bytes_original > 0 else 0.0
            )
        }
