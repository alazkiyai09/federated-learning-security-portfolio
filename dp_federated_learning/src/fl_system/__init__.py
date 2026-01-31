"""
Federated Learning with DP
==========================

FL components with differential privacy:
- DPClient: Client with DP-SGD training
- DPServer: Server with DP-aware aggregation
"""

from .dp_client import DPClient
from .dp_server import DPServer

__all__ = [
    "DPClient",
    "DPServer",
]
