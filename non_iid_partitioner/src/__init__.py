"""
Non-IID Data Partitioner for Federated Learning

A toolkit for simulating realistic heterogeneous data distributions in FL experiments.
"""

from .partitioner import NonIIDPartitioner
from .visualization import (
    plot_client_distribution,
    plot_quantity_distribution,
    compute_heterogeneity_metrics
)

__version__ = "0.1.0"
__all__ = [
    "NonIIDPartitioner",
    "plot_client_distribution",
    "plot_quantity_distribution",
    "compute_heterogeneity_metrics"
]
