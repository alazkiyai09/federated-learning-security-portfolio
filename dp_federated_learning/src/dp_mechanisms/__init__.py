"""
Differential Privacy Mechanisms
================================

Core primitives for implementing DP-SGD:
- Per-sample gradient computation
- L2 gradient clipping
- Gaussian noise addition
- Privacy accounting (RDP accountant)
"""

from .gradient_clipper import compute_per_sample_gradients, clip_gradients_l2
from .noise_addition import add_gaussian_noise, compute_noise_multiplier
from .privacy_accountant import RDPAccountant

__all__ = [
    "compute_per_sample_gradients",
    "clip_gradients_l2",
    "add_gaussian_noise",
    "compute_noise_multiplier",
    "RDPAccountant",
]
