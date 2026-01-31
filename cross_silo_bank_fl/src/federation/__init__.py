"""
Federated learning module using Flower framework.
"""

from .flower_client import FraudClient, client_fn
from .strategy import PerBankMetricStrategy, get_per_bank_metrics
from .secure_aggregation import apply_additive_masking

__all__ = [
    "FraudClient",
    "client_fn",
    "PerBankMetricStrategy",
    "get_per_bank_metrics",
    "apply_additive_masking"
]
