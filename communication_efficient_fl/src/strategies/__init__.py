"""
Strategies for communication-efficient federated learning.
"""

from .efficient_fedavg import EfficientFedAvg, AdaptiveCompressionStrategy
from .compression_wrapper import CompressionWrapper

__all__ = [
    'EfficientFedAvg',
    'AdaptiveCompressionStrategy',
    'CompressionWrapper'
]
