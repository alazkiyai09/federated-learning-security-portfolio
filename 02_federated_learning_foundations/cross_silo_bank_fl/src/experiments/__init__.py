"""
Experiment modules for comparing local, federated, and centralized approaches.
"""

from .local_baseline import train_local_models
from .centralized_baseline import train_centralized_model

__all__ = ["train_local_models", "train_centralized_model"]
