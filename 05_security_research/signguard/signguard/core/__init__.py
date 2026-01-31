"""Core components for SignGuard."""

from signguard.core.types import (
    ModelUpdate,
    SignedUpdate,
    AnomalyScore,
    ReputationInfo,
    AggregationResult,
    ClientConfig,
    ServerConfig,
    ExperimentConfig,
)
from signguard.core.client import SignGuardClient, create_client
from signguard.core.server import SignGuardServer

__all__ = [
    # Types
    "ModelUpdate",
    "SignedUpdate",
    "AnomalyScore",
    "ReputationInfo",
    "AggregationResult",
    "ClientConfig",
    "ServerConfig",
    "ExperimentConfig",
    # Main components
    "SignGuardClient",
    "SignGuardServer",
    "create_client",
]
