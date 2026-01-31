"""
Vertical Federated Learning for Fraud Detection

This package implements split learning for vertically partitioned data
where different parties hold different features for the same users.

Architecture:
- Party A: Transaction features → embeddings
- Party B: Credit features → embeddings
- Server: Combined embeddings → fraud prediction
"""

__version__ = "0.1.0"
