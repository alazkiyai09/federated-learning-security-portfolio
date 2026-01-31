"""
Communication-efficient FedAvg strategy for Flower.

This module implements a custom FedAvg strategy with integrated
gradient compression for bandwidth-efficient federated learning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .compression_wrapper import CompressionWrapper
from ..compression.error_feedback import MultiLayerErrorFeedback


class EfficientFedAvg(FedAvg):
    """
    FedAvg with communication-efficient gradient compression.

    Extends Flower's FedAvg strategy to apply compression to
    client updates, reducing bandwidth consumption.

    Args:
        compress_func: Name of compression function ('top_k', 'quantize_8bit', etc.)
        error_feedback: Whether to use error feedback (residual accumulation)
        **compress_kwargs: Additional arguments for compression function
        **fedavg_kwargs: Additional arguments for FedAvg (min_fit_clients, etc.)

    Example:
        >>> strategy = EfficientFedAvg(
        ...     compress_func='top_k',
        ...     k=1000,
        ...     error_feedback=True,
        ...     min_fit_clients=10,
        ...     min_available_clients=10
        ... )
    """

    def __init__(
        self,
        compress_func: Optional[str] = None,
        error_feedback: bool = False,
        **compress_kwargs
    ):
        # Get FedAvg-specific kwargs
        fedavg_kwargs = {}
        fedavg_param_names = [
            'min_fit_clients',
            'min_evaluate_clients',
            'min_available_clients',
            'fraction_fit',
            'fraction_evaluate',
            'evaluate_metrics_aggregation_fn',
            'initial_parameters',
            'fit_metrics_aggregation_fn',
            'on_fit_config_fn',
            'on_evaluate_config_fn',
            'accept_failures',
            'drop_remaining_client'
        ]

        for key in fedavg_param_names:
            if key in compress_kwargs:
                fedavg_kwargs[key] = compress_kwargs.pop(key)

        # Initialize FedAvg parent class
        super().__init__(**fedavg_kwargs)

        # Initialize compression wrapper
        self.compression_wrapper = CompressionWrapper(
            compress_func=compress_func,
            **compress_kwargs
        )

        # Initialize error feedback
        self.error_feedback_enabled = error_feedback
        self.error_feedback: Optional[MultiLayerErrorFeedback] = None

        # Metrics tracking
        self.round_compression_ratios: List[float] = []
        self.round_bandwidth_savings: List[float] = []
        self.cumulative_bytes_original: int = 0
        self.cumulative_bytes_compressed: int = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates with compression.

        This method extends FedAvg's aggregate_fit to:
        1. Optionally decompress client updates (for quantization)
        2. Aggregate using standard FedAvg
        3. Apply compression to aggregated model before sending to clients

        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed clients

        Returns:
            (aggregated_parameters, metrics)
        """
        # Call parent's aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is None:
            return None, aggregated_metrics

        # Apply compression to aggregated parameters
        if self.compression_wrapper.compress_func is not None:
            compressed_parameters, compression_metrics = \
                self.compression_wrapper.compress_parameters(aggregated_parameters)

            # Update metrics
            self.round_compression_ratios.append(
                compression_metrics.get('compression_ratio', 1.0)
            )
            self.cumulative_bytes_original += compression_metrics.get(
                'original_bytes', 0
            )
            self.cumulative_bytes_compressed += compression_metrics.get(
                'compressed_bytes', 0
            )

            # Add compression metrics to aggregated metrics
            aggregated_metrics.update({
                'compression_ratio': compression_metrics.get('compression_ratio', 1.0),
                'bandwidth_savings_pct': compression_metrics.get('bandwidth_savings_pct', 0.0),
                'original_bytes': compression_metrics.get('original_bytes', 0),
                'compressed_bytes': compression_metrics.get('compressed_bytes', 0)
            })

            return compressed_parameters, aggregated_metrics

        return aggregated_parameters, aggregated_metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure clients for training round.

        Extends FedAvg's configure_fit to optionally initialize
        error feedback buffers based on parameter shapes.

        Args:
            server_round: Current round number
            parameters: Current model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, fit_instructions) tuples
        """
        # Initialize error feedback if enabled and not yet initialized
        if self.error_feedback_enabled and self.error_feedback is None:
            # Get parameter shapes
            ndarrays = parameters_to_ndarrays(parameters)
            shapes = [arr.shape for arr in ndarrays]

            # Initialize error feedback buffers
            self.error_feedback = MultiLayerErrorFeedback(shapes=shapes)

        # Call parent's configure_fit
        return super().configure_fit(server_round, parameters, client_manager)

    def get_compression_metrics(self) -> Dict[str, Scalar]:
        """
        Get cumulative compression metrics.

        Returns:
            Dict with compression statistics

        Example:
            >>> metrics = strategy.get_compression_metrics()
            >>> print(f"Average compression ratio: {metrics['avg_compression_ratio']}")
        """
        stats = self.compression_wrapper.get_compression_stats()

        if len(self.round_compression_ratios) > 0:
            stats.update({
                'avg_compression_ratio': float(np.mean(self.round_compression_ratios)),
                'std_compression_ratio': float(np.std(self.round_compression_ratios)),
                'min_compression_ratio': float(np.min(self.round_compression_ratios)),
                'max_compression_ratio': float(np.max(self.round_compression_ratios)),
                'total_rounds': len(self.round_compression_ratios)
            })

        return stats

    def reset_compression_metrics(self) -> None:
        """Reset compression metrics tracking."""
        self.round_compression_ratios = []
        self.round_bandwidth_savings = []
        self.compression_wrapper.reset_metrics()

    def enable_error_feedback(self, shapes: List[tuple]) -> None:
        """
        Enable error feedback with specified shapes.

        Args:
            shapes: List of parameter shapes for error feedback buffers

        Example:
            >>> strategy.enable_error_feedback([(100, 100), (100,), (50, 100)])
        """
        self.error_feedback_enabled = True
        self.error_feedback = MultiLayerErrorFeedback(shapes=shapes)

    def disable_error_feedback(self) -> None:
        """Disable error feedback and reset buffers."""
        self.error_feedback_enabled = False
        self.error_feedback = None


class AdaptiveCompressionStrategy(EfficientFedAvg):
    """
    Adaptive compression strategy that adjusts compression level based on
    training progress and bandwidth constraints.

    This strategy can start with aggressive compression and gradually
    reduce it as training progresses, ensuring convergence while
    maximizing bandwidth savings.

    Args:
        initial_compression_ratio: Target compression ratio for first round
        final_compression_ratio: Target compression ratio for last round
        total_rounds: Total number of training rounds
        compress_func: Base compression function name
        **kwargs: Additional arguments for EfficientFedAvg

    Example:
        >>> strategy = AdaptiveCompressionStrategy(
        ...     initial_compression_ratio=10.0,  # Start with 10x compression
        ...     final_compression_ratio=2.0,    # End with 2x compression
        ...     total_rounds=100,
        ...     compress_func='top_k'
        ... )
    """

    def __init__(
        self,
        initial_compression_ratio: float = 10.0,
        final_compression_ratio: float = 2.0,
        total_rounds: int = 100,
        compress_func: str = 'top_k',
        **kwargs
    ):
        super().__init__(compress_func=compress_func, **kwargs)

        self.initial_compression_ratio = initial_compression_ratio
        self.final_compression_ratio = final_compression_ratio
        self.total_rounds = total_rounds
        self.current_round = 0

    def _get_target_k(self, parameters: Parameters) -> int:
        """
        Calculate target K for current round (for Top-K sparsification).

        Args:
            parameters: Current model parameters

        Returns:
            Target K value
        """
        # Calculate progress (0 to 1)
        progress = min(self.current_round / self.total_rounds, 1.0)

        # Interpolate compression ratio
        target_ratio = (
            self.initial_compression_ratio * (1 - progress) +
            self.final_compression_ratio * progress
        )

        # Calculate K from compression ratio
        ndarrays = parameters_to_ndarrays(parameters)
        total_elements = sum(arr.size for arr in ndarrays)
        k = max(1, int(total_elements / target_ratio))

        return k

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate with adaptive compression.

        Adjusts compression level based on training progress.
        """
        self.current_round = server_round

        # Get current parameters to calculate adaptive K
        if results and len(results) > 0:
            current_params = results[0][1].parameters

            # Update compression wrapper with adaptive K
            if self.compression_wrapper.compress_func_name == 'top_k':
                k = self._get_target_k(current_params)
                self.compression_wrapper.compress_kwargs['k'] = k

        # Call parent's aggregate_fit
        return super().aggregate_fit(server_round, results, failures)
