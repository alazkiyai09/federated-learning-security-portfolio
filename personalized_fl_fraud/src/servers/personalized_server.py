"""
Personalized Server Strategies

Implements server-side aggregation strategies for each personalization method.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy


class PersonalizedServer:
    """
    Server for personalized federated learning experiments.

    Wraps Flower server strategies and provides:
    1. Custom aggregation functions
    2. Metric collection
    3. Per-client result tracking
    """

    def __init__(
        self,
        strategy: str = "fedavg",
        fraction_fit: float = 0.8,
        min_fit_clients: int = 6,
        min_available_clients: int = 8,
        config: Optional[Dict] = None
    ):
        """
        Initialize personalized server.

        Args:
            strategy: Strategy name (fedavg, fedper, ditto, per_fedavg)
            fraction_fit: Fraction of clients to sample each round
            min_fit_clients: Minimum clients for training
            min_available_clients: Minimum available clients
            config: Optional configuration dict
        """
        self.strategy_name = strategy
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.config = config or {}

        # Track metrics
        self.round_metrics = []
        self.current_round = 0

        # Get Flower strategy
        self.fl_strategy = self._create_fl_strategy()

    def _create_fl_strategy(self) -> FedAvg:
        """Create Flower server strategy."""
        return FedAvg(
            fraction_fit=self.fraction_fit,
            min_fit_clients=self.min_fit_clients,
            min_available_clients=self.min_available_clients,
            on_fit_config_fn=self._on_fit_config,
            on_evaluate_config_fn=self._on_evaluate_config,
            fit_metrics_aggregation_fn=self._fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn=self._evaluate_metrics_aggregation,
        )

    def _on_fit_config(self, round_num: int) -> Dict[str, Scalar]:
        """Return training configuration for clients."""
        config = {
            "round": round_num,
            "local_epochs": self.config.get("local_epochs", 5),
        }

        # Add method-specific config
        if self.strategy_name == "ditto":
            config["proximal_mu"] = self.config.get("lambda_regularization", 0.5)
        elif self.strategy_name == "fedper":
            config["freeze_feature_extractor"] = True

        return config

    def _on_evaluate_config(self, round_num: int) -> Dict[str, Scalar]:
        """Return evaluation configuration for clients."""
        return {"round": round_num}

    def _fit_metrics_aggregation(
        self,
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate training metrics from clients.

        Args:
            metrics: List of (num_samples, metrics) tuples

        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}

        # Weight by number of samples
        total_samples = sum(num_samples for num_samples, _ in metrics)

        aggregated = {}
        for metric_name in metrics[0][1].keys():
            if metric_name in ["loss"]:
                # Average loss
                weighted_sum = sum(
                    num_samples * client_metrics.get(metric_name, 0)
                    for num_samples, client_metrics in metrics
                )
                aggregated[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0
            elif metric_name in ["auc", "pr_auc", "f1_score"]:
                # Average these metrics
                values = [
                    client_metrics.get(metric_name, 0)
                    for _, client_metrics in metrics
                ]
                aggregated[metric_name] = np.mean(values) if values else 0

        return aggregated

    def _evaluate_metrics_aggregation(
        self,
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients."""
        return self._fit_metrics_aggregation(metrics)

    def get_strategy(self) -> FedAvg:
        """Get Flower server strategy."""
        return self.fl_strategy

    def log_round_metrics(
        self,
        round_num: int,
        loss: float,
        metrics: Dict[str, Scalar]
    ) -> None:
        """Log metrics for a round."""
        self.current_round = round_num
        self.round_metrics.append({
            'round': round_num,
            'loss': loss,
            **metrics
        })

    def get_all_metrics(self) -> List[Dict]:
        """Get all logged metrics."""
        return self.round_metrics

    def get_summary(self) -> Dict:
        """Get summary of all rounds."""
        if not self.round_metrics:
            return {}

        return {
            'total_rounds': len(self.round_metrics),
            'final_loss': self.round_metrics[-1].get('loss', 0),
            'final_auc': self.round_metrics[-1].get('auc', 0),
            'all_metrics': self.round_metrics
        }


def create_server(
    strategy_name: str = "fedavg",
    fraction_fit: float = 0.8,
    min_fit_clients: int = 6,
    min_available_clients: int = 8,
    config: Optional[Dict] = None
) -> PersonalizedServer:
    """
    Factory function to create a personalized server.

    Args:
        strategy_name: Name of personalization strategy
        fraction_fit: Fraction of clients to sample each round
        min_fit_clients: Minimum clients for training
        min_available_clients: Minimum available clients
        config: Configuration dictionary

    Returns:
        PersonalizedServer instance
    """
    return PersonalizedServer(
        strategy=strategy_name,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        config=config
    )
