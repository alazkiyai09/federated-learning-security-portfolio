"""
Differentially Private Federated Learning Framework
===================================================

This package implements formal differential privacy guarantees for federated
learning systems, with applications to fraud detection.

Main Components:
- dp_mechanisms: Core DP primitives (clipping, noise, accounting)
- dp_strategies: Local/central/shuffle DP strategies
- models: DP-SGD optimizers (custom and Opacus wrapper)
- fl_system: FL clients and servers with DP
- experiments: Utility-privacy analysis tools
- utils: Privacy calibration and visualization

Reference:
    Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
"""

__version__ = "0.1.0"
__author__ = "Privacy Research Team"
