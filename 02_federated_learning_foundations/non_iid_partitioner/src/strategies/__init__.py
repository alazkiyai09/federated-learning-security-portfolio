"""
Partition strategies for non-IID data distribution.
"""

from .iid import iid_partition
from .label_skew import dirichlet_partition
from .quantity_skew import power_law_allocation
from .feature_skew import feature_based_partition
from .realistic_bank import realistic_bank_partition

__all__ = [
    "iid_partition",
    "dirichlet_partition",
    "power_law_allocation",
    "feature_based_partition",
    "realistic_bank_partition"
]
