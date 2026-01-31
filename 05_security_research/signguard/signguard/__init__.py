"""SignGuard: Cryptographic Signature-Based Defense for Federated Learning.

This package provides a complete implementation of SignGuard, a defense mechanism
that combines cryptographic authentication, multi-factor anomaly detection, and
dynamic reputation scoring to secure federated learning systems against Byzantine
attacks.

Example:
    >>> from signguard import SignGuardClient, SignGuardServer
    >>> client = SignGuardClient(client_id="client_0", model=model, ...)
    >>> server = SignGuardServer(global_model=model, ...)
    >>> # Training loop
    >>> for round in range(num_rounds):
    ...     updates = [client.train(global_model) for client in clients]
    ...     signed_updates = [client.sign_update(u) for u in updates]
    ...     result = server.aggregate(signed_updates)
    ...     global_model = result.global_model
"""

__version__ = "0.1.0"
__author__ = "Researcher"
__email__ = "researcher@university.edu"

# Core exports
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

# Detection exports
from signguard.detection.ensemble import EnsembleDetector

# Reputation exports
from signguard.reputation.decay_reputation import DecayReputationSystem

# Aggregation exports
from signguard.aggregation.weighted_aggregator import WeightedAggregator

# Crypto exports
from signguard.crypto import SignatureManager, KeyStore, KeyManager

# Attack exports
from signguard.attacks import (
    Attack,
    LabelFlipAttack,
    BackdoorAttack,
    ModelPoisonAttack,
)

# Defense exports
from signguard.defenses import (
    KrumDefense,
    TrimmedMeanDefense,
    FoolsGoldDefense,
    BulyanDefense,
)

# Utility exports
from signguard.utils.metrics import (
    compute_accuracy,
    compute_attack_success_rate,
    compute_detection_metrics,
    compute_overhead_metrics,
)

__all__ = [
    # Version
    "__version__",
    # Core types
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
    # Detection
    "EnsembleDetector",
    # Reputation
    "DecayReputationSystem",
    # Aggregation
    "WeightedAggregator",
    # Crypto
    "SignatureManager",
    "KeyStore",
    "KeyManager",
    # Attacks
    "Attack",
    "LabelFlipAttack",
    "BackdoorAttack",
    "ModelPoisonAttack",
    # Defenses
    "KrumDefense",
    "TrimmedMeanDefense",
    "FoolsGoldDefense",
    "BulyanDefense",
    # Utilities
    "compute_accuracy",
    "compute_attack_success_rate",
    "compute_detection_metrics",
    "compute_overhead_metrics",
]
