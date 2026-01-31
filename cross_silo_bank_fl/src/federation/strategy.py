"""
Custom FedAvg strategy with per-bank metric tracking.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import flwr as fl
from flwr.common import Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import pandas as pd


class PerBankMetricStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that tracks metrics per bank.

    Extends Flower's FedAvg to:
    - Track per-bank metrics across rounds
    - Store detailed training history
    - Calculate weighted averages properly
    """

    def __init__(self, *args, **kwargs):
        """Initialize the strategy."""
        super().__init__(*args, **kwargs)

        # Track per-bank metrics
        self.per_bank_metrics = defaultdict(lambda: defaultdict(list))
        self.round_metrics = []

        # Track sample counts
        self.bank_sample_counts = defaultdict(int)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients.

        Args:
            rnd: Current round number
            results: List of (client_proxy, fit_res) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, metrics_dict)
        """
        # Call parent to get standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd=rnd,
            results=results,
            failures=failures
        )

        # Store per-bank metrics from this round
        round_metrics = {'round': rnd}

        for client_proxy, fit_res in results:
            # Extract metrics from FitRes
            metrics = fit_res.metrics
            bank_id = metrics.get('bank_id', 'unknown')

            # Store metrics
            for key, value in metrics.items():
                if key not in ['bank_id']:
                    self.per_bank_metrics[bank_id][key].append(value)
                    round_metrics[f"{bank_id}_{key}"] = value

            # Store sample count
            self.bank_sample_counts[bank_id] = metrics.get('n_train_samples', 0)

        self.round_metrics.append(round_metrics)

        # Calculate aggregated metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}

        # Add weighted average AUC
        weighted_auc = self._calculate_weighted_metric('auc_roc', results)
        if weighted_auc is not None:
            aggregated_metrics['weighted_avg_auc'] = weighted_auc

        # Add weighted average F1
        weighted_f1 = self._calculate_weighted_metric('f1', results)
        if weighted_f1 is not None:
            aggregated_metrics['weighted_avg_f1'] = weighted_f1

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics from clients.

        Args:
            rnd: Current round number
            results: List of (client_proxy, eval_res) tuples
            failures: List of failures

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Call parent
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            rnd=rnd,
            results=results,
            failures=failures
        )

        # Add per-bank evaluation metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}

        eval_metrics = {}
        for client_proxy, eval_res in results:
            bank_id = eval_res.metrics.get('bank_id', 'unknown')
            for key, value in eval_res.metrics.items():
                if key != 'bank_id':
                    eval_metrics[f"eval_{bank_id}_{key}"] = value

        aggregated_metrics.update(eval_metrics)

        return aggregated_loss, aggregated_metrics

    def _calculate_weighted_metric(
        self,
        metric_name: str,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Optional[float]:
        """
        Calculate sample-weighted average of a metric.

        Args:
            metric_name: Name of metric to average
            results: List of client results

        Returns:
            Weighted average or None if not available
        """
        values = []
        weights = []

        for _, fit_res in results:
            if metric_name in fit_res.metrics:
                num_samples = fit_res.num_examples
                metric_value = fit_res.metrics[metric_name]

                values.append(metric_value)
                weights.append(num_samples)

        if not values:
            return None

        # Calculate weighted average
        total_weight = sum(weights)
        weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_weight

        return weighted_avg

    def get_per_bank_metrics(self) -> Dict[str, Dict[str, List]]:
        """
        Get per-bank metrics history.

        Returns:
            Dictionary mapping bank_id -> metric_name -> list of values
        """
        return dict(self.per_bank_metrics)

    def get_round_metrics(self) -> List[Dict]:
        """
        Get metrics for each round.

        Returns:
            List of metric dictionaries per round
        """
        return self.round_metrics

    def get_final_metrics(self) -> pd.DataFrame:
        """
        Create DataFrame of final per-bank metrics.

        Returns:
            DataFrame with final metrics for each bank
        """
        final_metrics = []

        for bank_id, metrics_dict in self.per_bank_metrics.items():
            metric_row = {'bank_id': bank_id}

            for metric_name, values in metrics_dict.items():
                if values:
                    metric_row[f'{metric_name}_final'] = values[-1]
                    metric_row[f'{metric_name}_best'] = max(values)

            final_metrics.append(metric_row)

        return pd.DataFrame(final_metrics)


def create_strategy(
    fraction_fit: float = 1.0,
    min_fit_clients: int = 5,
    min_available_clients: int = 5
) -> PerBankMetricStrategy:
    """
    Create FedAvg strategy with per-bank tracking.

    Args:
        fraction_fit: Fraction of clients to use for training
        min_fit_clients: Minimum number of clients for training
        min_available_clients: Minimum number of available clients

    Returns:
        Configured PerBankMetricStrategy
    """
    strategy = PerBankMetricStrategy(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        # Use standard FedAvg parameters
        min_evaluate_clients=min_fit_clients,
        fraction_evaluate=fraction_fit,
    )

    return strategy


def get_per_bank_metrics(
    strategy: PerBankMetricStrategy,
    metric_name: str = 'auc_roc'
) -> pd.DataFrame:
    """
    Extract per-bank metrics from strategy.

    Args:
        strategy: Trained strategy object
        metric_name: Name of metric to extract

    Returns:
        DataFrame with metrics per bank per round
    """
    per_bank_data = []

    metrics_dict = strategy.get_per_bank_metrics()

    for bank_id, bank_metrics in metrics_dict.items():
        if metric_name in bank_metrics:
            values = bank_metrics[metric_name]
            for round_num, value in enumerate(values):
                per_bank_data.append({
                    'round': round_num + 1,
                    'bank_id': bank_id,
                    metric_name: value
                })

    return pd.DataFrame(per_bank_data)
