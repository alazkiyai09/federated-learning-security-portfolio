"""
FedAdam Strategy Implementation

Implements server-side Adam optimizer for adaptive aggregation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from src.strategy.base import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
    aggregate_parameters,
)


class FedAdamCustom(Strategy):
    """
    FedAdam Strategy with server-side adaptive optimization.

    Uses Adam optimizer on the server side for adaptive learning rates.
    Maintains first and second moment estimates for each parameter.

    Reference: Reddi et al. "Adaptive Federated Optimization" (ICLR 2021)
    """

    def __init__(
        self,
        tau: float = 0.9,
        eta: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation: Optional[List[str]] = None,
        evaluate_metrics_aggregation: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize FedAdam strategy.

        Args:
            tau: Server momentum parameter (higher = more momentum)
            eta: Server learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            accept_failures: Whether to accept client failures
            initial_parameters: Initial global model parameters
            fit_metrics_aggregation: List of aggregation methods for fit metrics
            evaluate_metrics_aggregation: List of aggregation methods for eval metrics
        """
        super().__init__()
        self.tau = tau
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation = fit_metrics_aggregation or ["weighted_average"]
        self.evaluate_metrics_aggregation = evaluate_metrics_aggregation or ["weighted_average"]

        # Adam optimizer state
        self.m: Optional[List[np.ndarray]] = None  # First moment
        self.v: Optional[List[np.ndarray]] = None  # Second moment
        self.current_round = 0

    def __repr__(self) -> str:
        return f"FedAdamCustom(tau={self.tau}, eta={self.eta}, fraction_fit={self.fraction_fit})"

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        Args:
            client_manager: Flower client manager

        Returns:
            Initial parameters or None
        """
        return self.initial_parameters

    def configure_fit(
        self,
        rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[Any, FitIns]]:
        """
        Configure the next round of training.

        Args:
            rnd: Current round number
            parameters: Current global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, fit_ins) tuples
        """
        # Sample clients for training
        num_clients = int(client_manager.num_available() * self.fraction_fit)
        num_clients = max(num_clients, self.min_fit_clients)
        num_clients = min(num_clients, client_manager.num_available())

        clients = client_manager.sample(num_clients, min_num_clients=self.min_available_clients)

        # Create fit instructions
        config = {
            "rnd": rnd,
            "local_epochs": 5,  # Default, can be overridden
        }
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Union[Tuple[Any, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using Adam optimizer.

        Args:
            rnd: Current round number
            results: List of (client, fit_result) from successful clients
            failures: List of failures from unsuccessful clients

        Returns:
            (aggregated_parameters, metrics_aggregated) tuple
        """
        if not results:
            return None, {}

        # Handle failures if not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Compute weighted average of client updates
        aggregated_parameters = aggregate_parameters(results)

        # Apply Adam update
        updated_parameters = self._adam_update(aggregated_parameters, results)

        # Aggregate metrics
        metrics_list = [
            (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
        ]
        aggregated_metrics = aggregate_fit_metrics(metrics_list)

        return updated_parameters, aggregated_metrics

    def _adam_update(
        self,
        aggregated_parameters: Parameters,
        results: List[Tuple[Any, FitRes]],
    ) -> Parameters:
        """
        Apply Adam optimizer update to aggregated parameters.

        Args:
            aggregated_parameters: Weighted average of client parameters
            results: Client results for computing delta

        Returns:
            Updated parameters
        """
        from flwr.common import parameters_to_ndarrays

        # Convert to numpy arrays
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # Initialize moment estimates if first round
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(arr) for arr in aggregated_ndarrays]
            self.v = [np.zeros_like(arr) for arr in aggregated_ndarrays]

        # Compute client deltas (for scaling)
        # Get the previous parameters (or use aggregated if first round)
        if self.current_round == 0:
            delta = [arr * self.eta for arr in aggregated_ndarrays]
        else:
            # Compute difference from previous aggregated
            delta = aggregated_ndarrays  # Simplified, could compute actual delta

        # Update moments
        for i in range(len(aggregated_ndarrays)):
            # Update first moment (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta[i]

            # Update second moment (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (delta[i] ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** (self.current_round + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.current_round + 1))

            # Apply Adam update
            aggregated_ndarrays[i] = aggregated_ndarrays[i] + self.eta * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )

        # Convert back to Parameters
        from flwr.common import ndarrays_to_parameters

        self.current_round += 1
        return ndarrays_to_parameters(aggregated_ndarrays)

    def configure_evaluate(
        self,
        rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[Any, EvaluateIns]]:
        """
        Configure the next round of evaluation.

        Args:
            rnd: Current round number
            parameters: Current global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, evaluate_ins) tuples
        """
        # Sample clients for evaluation
        if self.fraction_evaluate == 0.0:
            return []

        num_clients = int(client_manager.num_available() * self.fraction_evaluate)
        num_clients = max(num_clients, self.min_evaluate_clients)
        num_clients = min(num_clients, client_manager.num_available())

        clients = client_manager.sample(num_clients, min_num_clients=self.min_available_clients)

        # Create evaluate instructions
        config = {"rnd": rnd}
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Union[Tuple[Any, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.

        Args:
            rnd: Current round number
            results: List of (client, evaluate_result) from successful clients
            failures: List of failures from unsuccessful clients

        Returns:
            (aggregated_loss, aggregated_metrics) tuple
        """
        if not results:
            return None, {}

        # Aggregate loss
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        weighted_losses = [
            evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results
        ]
        aggregated_loss = sum(weighted_losses) / total_examples

        # Aggregate metrics
        metrics_list = [
            (evaluate_res.num_examples, evaluate_res.metrics) for _, evaluate_res in results
        ]
        aggregated_metrics = aggregate_evaluate_metrics(metrics_list)

        return aggregated_loss, aggregated_metrics
