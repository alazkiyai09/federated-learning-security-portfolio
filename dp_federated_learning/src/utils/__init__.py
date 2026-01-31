"""
Utility Functions
=================

Helper utilities:
- Privacy calibration (compute sigma from epsilon)
- Visualization tools (Pareto curves)
"""

from .privacy_calibration import compute_noise_multiplier_from_epsilon
from .visualization import plot_pareto_curve, plot_convergence

__all__ = [
    "compute_noise_multiplier_from_epsilon",
    "plot_pareto_curve",
    "plot_convergence",
]
