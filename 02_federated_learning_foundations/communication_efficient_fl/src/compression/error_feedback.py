"""
Error feedback mechanism for communication-efficient FL.

This module implements residual accumulation (lazy update) to ensure
that dropped gradients are not lost but accumulated for future updates.
Error feedback is crucial for maintaining convergence with aggressive
compression (high sparsification or low-bit quantization).
"""

import numpy as np
from typing import Callable, Tuple, Optional
from .utils import measure_bytes, calculate_compression_ratio


class ErrorFeedback:
    """
    Accumulate dropped gradients as residuals (lazy update).

    The key insight: when compression drops some gradient values,
    we don't simply discard them. Instead, we accumulate them in a
    residual buffer and apply them in the next round.

    This ensures that even small gradients eventually contribute
    to the model update, maintaining convergence properties.

    Example:
        >>> ef = ErrorFeedback(shape=(100, 100))
        >>> gradients = np.random.randn(100, 100)
        >>> compressed_grads, ratio = ef.compress_and_update(
        ...     gradients,
        ...     lambda x: top_k_sparsify(x, k=100)[0]
        ... )
        # Residuals automatically accumulated internally
    """

    def __init__(self, shape: tuple, dtype: np.dtype = np.float32):
        """
        Initialize error feedback buffer.

        Args:
            shape: Shape of the gradient arrays
            dtype: Data type for residuals (default: float32)
        """
        self.residual = np.zeros(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def compress_and_update(
        self,
        gradients: np.ndarray,
        compress_func: Callable,
        **compress_kwargs
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Apply compression to gradients and update residuals.

        Process:
        1. Add accumulated residuals to current gradients
        2. Apply compression function
        3. Calculate what was dropped (error)
        4. Accumulate error back into residual buffer

        Args:
            gradients: Current gradients to compress
            compress_func: Compression function (e.g., top_k_sparsify)
            **compress_kwargs: Additional arguments for compress_func

        Returns:
            (compressed_gradients, compression_ratio, metrics)
            - compressed_gradients: Compressed gradients ready for transmission
            - compression_ratio: Compression ratio achieved
            - metrics: Dict with additional metrics (residual_norm, etc.)
        """
        # Step 1: Add residuals to gradients (lazy update)
        gradients_with_residual = gradients + self.residual

        # Step 2: Apply compression
        # Handle different return types from compression functions
        result = compress_func(gradients_with_residual, **compress_kwargs)

        if isinstance(result, tuple):
            # Most compression functions return (array, mask/params, ratio)
            compressed_gradients = result[0]
            compression_ratio = result[-1]  # Ratio is usually last element
        else:
            # Function returns just the compressed array
            compressed_gradients = result
            original_bytes = measure_bytes(gradients)
            compressed_bytes = measure_bytes(compressed_gradients)
            compression_ratio = calculate_compression_ratio(
                original_bytes, compressed_bytes
            )

        # Step 3: Calculate error (what was dropped)
        # Error = (gradients + residual) - compressed
        error = gradients_with_residual - compressed_gradients

        # Step 4: Update residual buffer
        self.residual = error

        # Calculate metrics
        metrics = {
            'residual_norm': float(np.linalg.norm(self.residual)),
            'residual_mean': float(np.mean(np.abs(self.residual))),
            'error_norm': float(np.linalg.norm(error)),
            'compression_ratio': compression_ratio
        }

        return compressed_gradients, compression_ratio, metrics

    def get_residual(self) -> np.ndarray:
        """
        Get current residual buffer.

        Returns:
            Current accumulated residuals

        Example:
            >>> ef = ErrorFeedback(shape=(10, 10))
            >>> # ... after several rounds of compression ...
            >>> residual = ef.get_residual()
            >>> print(f"Residual norm: {np.linalg.norm(residual)}")
        """
        return self.residual.copy()

    def reset_residual(self) -> None:
        """
        Reset residual buffer to zeros.

        Use this to start fresh (e.g., at the start of a new training phase).
        """
        self.residual = np.zeros_like(self.residual)

    def get_residual_statistics(self) -> dict:
        """
        Get statistics about current residual buffer.

        Returns:
            Dict with residual statistics (norm, mean, max, etc.)

        Example:
            >>> ef = ErrorFeedback(shape=(100, 100))
            >>> # ... compress some gradients ...
            >>> stats = ef.get_residual_statistics()
            >>> print(stats)
            {'norm': 1.23, 'mean': 0.01, 'max': 0.5, ...}
        """
        return {
            'norm': float(np.linalg.norm(self.residual)),
            'mean': float(np.mean(self.residual)),
            'abs_mean': float(np.mean(np.abs(self.residual))),
            'max': float(np.max(self.residual)),
            'min': float(np.min(self.residual)),
            'std': float(np.std(self.residual)),
            'nonzero_fraction': float(np.mean(self.residual != 0))
        }


class MultiLayerErrorFeedback:
    """
    Error feedback for multiple layers (parameter tensors).

    Manages separate residual buffers for each layer in a neural network.

    Example:
        >>> mlef = MultiLayerErrorFeedback([
        ...     (100, 100),  # Weights layer 1
        ...     (100,),      # Bias layer 1
        ...     (50, 100)    # Weights layer 2
        ... ])
        >>> gradients = [
        ...     np.random.randn(100, 100),
        ...     np.random.randn(100),
        ...     np.random.randn(50, 100)
        ... ]
        >>> compressed_grads, ratios, metrics = mlef.compress_and_update_layers(
        ...     gradients,
        ...     lambda x: top_k_sparsify(x, k=100)[0]
        ... )
    """

    def __init__(self, shapes: list, dtype: np.dtype = np.float32):
        """
        Initialize error feedback buffers for multiple layers.

        Args:
            shapes: List of shapes for each layer
            dtype: Data type for residuals
        """
        self.residuals = [
            np.zeros(shape, dtype=dtype)
            for shape in shapes
        ]
        self.shapes = shapes
        self.dtype = dtype

    def compress_and_update_layers(
        self,
        gradients: list,
        compress_func: Callable,
        **compress_kwargs
    ) -> Tuple[list, list, dict]:
        """
        Apply compression to all layers and update residuals.

        Args:
            gradients: List of gradient arrays (one per layer)
            compress_func: Compression function
            **compress_kwargs: Additional arguments for compress_func

        Returns:
            (compressed_gradients, compression_ratios, metrics)
            - compressed_gradients: List of compressed gradients
            - compression_ratios: List of compression ratios (one per layer)
            - metrics: Dict with aggregated metrics
        """
        if len(gradients) != len(self.residuals):
            raise ValueError(
                f"Number of gradient layers ({len(gradients)}) "
                f"does not match number of residual buffers ({len(self.residuals)})"
            )

        compressed_gradients = []
        compression_ratios = []
        all_metrics = []

        for i, (grad, residual) in enumerate(zip(gradients, self.residuals)):
            # Add residual
            grad_with_residual = grad + residual

            # Apply compression
            result = compress_func(grad_with_residual, **compress_kwargs)

            if isinstance(result, tuple):
                compressed = result[0]
                ratio = result[-1]
            else:
                compressed = result
                original_bytes = measure_bytes(grad)
                compressed_bytes = measure_bytes(compressed)
                ratio = calculate_compression_ratio(original_bytes, compressed_bytes)

            # Calculate error
            error = grad_with_residual - compressed

            # Update residual
            self.residuals[i] = error

            compressed_gradients.append(compressed)
            compression_ratios.append(ratio)

            all_metrics.append({
                'layer': i,
                'residual_norm': float(np.linalg.norm(error)),
                'compression_ratio': ratio
            })

        # Aggregate metrics
        metrics = {
            'mean_compression_ratio': float(np.mean(compression_ratios)),
            'total_compression_ratio': float(np.prod(compression_ratios)),
            'layer_metrics': all_metrics
        }

        return compressed_gradients, compression_ratios, metrics

    def reset_all_residuals(self) -> None:
        """Reset all residual buffers to zeros."""
        for i in range(len(self.residuals)):
            self.residuals[i] = np.zeros_like(self.residuals[i])

    def get_all_residuals(self) -> list:
        """Get all residual buffers."""
        return [r.copy() for r in self.residuals]
