"""
FedProx Strategy Implementation

Extends FedAvg with proximal term to handle non-IID data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from src.strategy.fedavg import FedAvgCustom


class FedProxCustom(FedAvgCustom):
    """
    FedProx Strategy for handling non-IID data.

    Extends FedAvg by adding a proximal term to the client's local objective:
    ||w - w_global||^2, which constrains local updates to stay close to
    the global model.

    The proximal term coefficient mu controls the strength of this constraint.
    """

    def __init__(
        self,
        proximal_mu: float = 0.01,
        **kwargs,
    ) -> None:
        """
        Initialize FedProx strategy.

        Args:
            proximal_mu: Coefficient for proximal term (default: 0.01)
                        Higher values constrain local updates more strongly
            **kwargs: Additional arguments passed to FedAvgCustom
        """
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        return f"FedProxCustom(mu={self.proximal_mu}, fraction_fit={self.fraction_fit})"

    def configure_fit(
        self,
        rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[Any, FitIns]]:
        """
        Configure training with proximal term coefficient.

        Adds the proximal_mu parameter to the client configuration so
        clients can incorporate it into their local training.

        Args:
            rnd: Current round number
            parameters: Current global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, fit_ins) tuples
        """
        # Get base configuration from parent
        configurations = super().configure_fit(rnd, parameters, client_manager)

        # Add proximal term coefficient to config
        configs_with_proximal = []
        for client, fit_ins in configurations:
            config = dict(fit_ins.config)
            config["proximal_mu"] = self.proximal_mu
            configs_with_proximal.append((client, FitIns(fit_ins.parameters, config)))

        return configs_with_proximal
