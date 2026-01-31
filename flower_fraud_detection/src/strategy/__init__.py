"""
Custom Flower Strategies for Fraud Detection

Implements FedAvg, FedProx, and FedAdam strategies extending
fl.server.strategy.Strategy.
"""

from .fedadam import FedAdamCustom
from .fedavg import FedAvgCustom
from .fedprox import FedProxCustom

__all__ = ["FedAvgCustom", "FedProxCustom", "FedAdamCustom"]
