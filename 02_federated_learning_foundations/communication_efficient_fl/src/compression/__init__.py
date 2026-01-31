"""
Communication-efficient compression techniques for federated learning.

This module provides gradient sparsification, quantization, and error
feedback mechanisms to reduce communication overhead in FL systems.
"""

from .utils import (
    measure_bytes,
    measure_compressed_bytes,
    calculate_compression_ratio,
    calculate_bandwidth_savings,
    estimate_transmission_time
)

from .sparsifiers import (
    top_k_sparsify,
    random_k_sparsify,
    threshold_sparsify,
    top_k_sparsify_percentage
)

from .quantizers import (
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    stochastic_quantize,
    dequantize_stochastic
)

from .error_feedback import (
    ErrorFeedback,
    MultiLayerErrorFeedback
)

__all__ = [
    # Utils
    'measure_bytes',
    'measure_compressed_bytes',
    'calculate_compression_ratio',
    'calculate_bandwidth_savings',
    'estimate_transmission_time',

    # Sparsifiers
    'top_k_sparsify',
    'random_k_sparsify',
    'threshold_sparsify',
    'top_k_sparsify_percentage',

    # Quantizers
    'quantize_8bit',
    'dequantize_8bit',
    'quantize_4bit',
    'dequantize_4bit',
    'stochastic_quantize',
    'dequantize_stochastic',

    # Error Feedback
    'ErrorFeedback',
    'MultiLayerErrorFeedback'
]
