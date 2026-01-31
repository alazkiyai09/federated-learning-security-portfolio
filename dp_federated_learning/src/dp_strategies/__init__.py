"""
Differential Privacy Strategies
================================

DP deployment strategies for federated learning:
- Local DP: Noise added at each client before transmission
- Central DP: Noise added at server after aggregation
- Shuffle DP: Amplification via intermediate shuffling
"""

from .local_dp import LocalDPStrategy
from .central_dp import CentralDPStrategy
from .shuffle_dp import ShuffleDPStrategy

__all__ = [
    "LocalDPStrategy",
    "CentralDPStrategy",
    "ShuffleDPStrategy",
]
