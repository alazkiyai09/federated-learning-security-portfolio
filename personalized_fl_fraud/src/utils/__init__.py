"""
Utilities module exports.
"""

from .partitioning import DataPartitioner, create_synthetic_fraud_data
from .reproducibility import (
    set_random_seed,
    CheckpointManager,
    ExperimentTracker,
    save_model_parameters,
    load_model_parameters
)
from .metrics import compute_fraud_metrics, compute_fairness_metrics, compute_gini

# Optional imports (may require additional dependencies)
try:
    from .compute_tracking import ComputeTracker, MemoryTracker, compare_compute_budgets
    _has_compute_tracking = True
except ImportError:
    ComputeTracker = None
    MemoryTracker = None
    compare_compute_budgets = None
    _has_compute_tracking = False

__all__ = [
    'DataPartitioner',
    'create_synthetic_fraud_data',
    'set_random_seed',
    'CheckpointManager',
    'ExperimentTracker',
    'save_model_parameters',
    'load_model_parameters',
    'compute_fraud_metrics',
    'compute_fairness_metrics',
    'compute_gini',
]

# Add optional exports
if _has_compute_tracking:
    __all__.extend(['ComputeTracker', 'MemoryTracker', 'compare_compute_budgets'])
