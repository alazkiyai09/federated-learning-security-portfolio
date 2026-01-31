"""
Clients module exports.
"""

from .personalized_client import (
    BaseClient,
    FedAvgClient,
    FedPerClient,
    DittoClient,
    PerFedAvgClient,
)
from .wrappers import (
    create_fedavg_client,
    create_fedper_client,
    create_ditto_client,
    create_per_fedavg_client,
    create_client,
    CLIENT_FACTORIES,
)

__all__ = [
    'BaseClient',
    'FedAvgClient',
    'FedPerClient',
    'DittoClient',
    'PerFedAvgClient',
    'create_fedavg_client',
    'create_fedper_client',
    'create_ditto_client',
    'create_per_fedavg_client',
    'create_client',
    'CLIENT_FACTORIES',
]
