"""
Metrics module exports.
"""

from .personalized_metrics import (
    compute_per_client_metrics,
    compute_personalization_benefit,
    aggregate_metrics,
    compute_worst_client_performance,
    compute_performance_variance,
    save_per_client_metrics,
    load_per_client_metrics,
    compare_methods_per_client,
)
from .visualization import (
    plot_per_client_violin,
    plot_personalization_vs_generalization,
    plot_alpha_sensitivity,
)

__all__ = [
    'compute_per_client_metrics',
    'compute_personalization_benefit',
    'aggregate_metrics',
    'compute_worst_client_performance',
    'compute_performance_variance',
    'save_per_client_metrics',
    'load_per_client_metrics',
    'compare_methods_per_client',
    'plot_per_client_violin',
    'plot_personalization_vs_generalization',
    'plot_alpha_sensitivity',
]
