"""
Compute Budget Tracking for Fair Comparison

Tracks:
1. FLOPs (Floating Point Operations)
2. Communication cost (bytes transferred)
3. Training time

Ensures fair comparison across personalization methods.
"""

from typing import Dict, List, Optional
from time import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import os

import numpy as np
import torch


class ComputeTracker:
    """
    Track compute budget usage for fair method comparison.

    Monitors:
    - FLOPs per forward/backward pass
    - Communication bytes per round
    - Training time per client/round

    Example:
        >>> tracker = ComputeTracker()
        >>> tracker.start_round()
        >>> # ... do training ...
        >>> tracker.end_round()
        >>> print(tracker.get_summary())
    """

    def __init__(self):
        """Initialize compute tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracking metrics."""
        self.round_metrics = []

        self.current_round = None
        self.current_start_time = None
        self.current_flops = 0

    def start_round(self, round_num: int) -> None:
        """
        Start tracking a new communication round.

        Args:
            round_num: Round number
        """
        self.current_round = {
            'round': round_num,
            'start_time': time(),
            'flops': 0,
            'communication_bytes': 0,
            'training_time': 0,
            'clients': {}
        }

    def end_round(self) -> Dict[str, float]:
        """
        End current round and return metrics.

        Returns:
            Round metrics dictionary
        """
        if self.current_round is None:
            raise ValueError("No round in progress. Call start_round() first.")

        self.current_round['end_time'] = time()
        self.current_round['total_time'] = (
            self.current_round['end_time'] - self.current_round['start_time']
        )

        round_summary = {
            'round': self.current_round['round'],
            'flops': self.current_round['flops'],
            'communication_bytes': self.current_round['communication_bytes'],
            'training_time': self.current_round['training_time'],
            'total_time': self.current_round['total_time']
        }

        self.round_metrics.append(round_summary)

        # Reset current round
        self.current_round = None

        return round_summary

    def track_client_training(
        self,
        client_id: int,
        n_batches: int,
        batch_size: int,
        input_dim: int,
        hidden_dims: List[int],
        n_epochs: int,
        n_local_steps: int
    ) -> Dict[str, float]:
        """
        Track compute for client training.

        Args:
            client_id: Client identifier
            n_batches: Number of batches processed
            batch_size: Batch size
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            n_epochs: Number of local epochs
            n_local_steps: Number of local optimization steps

        Returns:
            Client compute metrics
        """
        # Compute FLOPs
        flops_per_forward = self._estimate_forward_flops(
            input_dim, hidden_dims, batch_size
        )
        flops_per_backward = self._estimate_backward_flops(
            input_dim, hidden_dims, batch_size
        )

        total_flops = n_batches * n_epochs * (flops_per_forward + flops_per_backward)

        client_metrics = {
            'client_id': client_id,
            'flops': total_flops,
            'n_batches': n_batches,
            'n_epochs': n_epochs,
            'n_local_steps': n_local_steps
        }

        if self.current_round is not None:
            self.current_round['clients'][client_id] = client_metrics
            self.current_round['flops'] += total_flops

        return client_metrics

    def track_communication(
        self,
        parameters: List[np.ndarray],
        direction: str = "both"
    ) -> int:
        """
        Track communication cost for parameter transfer.

        Args:
            parameters: List of parameter arrays
            direction: 'upload', 'download', or 'both'

        Returns:
            Bytes transferred
        """
        bytes_count = self._compute_parameter_bytes(parameters)

        if direction in ['upload', 'both']:
            upload_bytes = bytes_count
        else:
            upload_bytes = 0

        if direction in ['download', 'both']:
            download_bytes = bytes_count
        else:
            download_bytes = 0

        total_bytes = upload_bytes + download_bytes

        if self.current_round is not None:
            self.current_round['communication_bytes'] += total_bytes

        return total_bytes

    def track_training_time(self, elapsed_seconds: float) -> None:
        """
        Track training time.

        Args:
            elapsed_seconds: Time elapsed in seconds
        """
        if self.current_round is not None:
            self.current_round['training_time'] += elapsed_seconds

    def get_round_summary(self, round_num: int) -> Optional[Dict]:
        """
        Get summary for specific round.

        Args:
            round_num: Round number

        Returns:
            Round metrics or None if not found
        """
        for round_data in self.round_metrics:
            if round_data['round'] == round_num:
                return round_data
        return None

    def get_summary(self) -> Dict[str, any]:
        """
        Get overall compute summary.

        Returns:
            Summary dictionary with total metrics
        """
        if not self.round_metrics:
            return {
                'total_rounds': 0,
                'total_flops': 0,
                'total_communication_bytes': 0,
                'total_training_time': 0,
                'total_time': 0
            }

        return {
            'total_rounds': len(self.round_metrics),
            'total_flops': sum(r['flops'] for r in self.round_metrics),
            'total_communication_bytes': sum(
                r['communication_bytes'] for r in self.round_metrics
            ),
            'total_training_time': sum(
                r['training_time'] for r in self.round_metrics
            ),
            'total_time': sum(r['total_time'] for r in self.round_metrics),
            'avg_flops_per_round': np.mean([r['flops'] for r in self.round_metrics]),
            'avg_communication_per_round': np.mean(
                [r['communication_bytes'] for r in self.round_metrics]
            ),
            'avg_training_time_per_round': np.mean(
                [r['training_time'] for r in self.round_metrics]
            )
        }

    def _estimate_forward_flops(
        self,
        input_dim: int,
        hidden_dims: List[int],
        batch_size: int
    ) -> int:
        """Estimate FLOPs for forward pass."""
        dims = [input_dim] + hidden_dims + [1]

        flops = 0
        for i in range(len(dims) - 1):
            m, n = dims[i], dims[i + 1]
            # Linear layer: 2*m*n*batch_size (multiply-add)
            flops += 2 * m * n * batch_size

        # Activation functions
        flops += sum(hidden_dims + [1]) * batch_size

        return flops

    def _estimate_backward_flops(
        self,
        input_dim: int,
        hidden_dims: List[int],
        batch_size: int
    ) -> int:
        """Estimate FLOPs for backward pass (~2x forward)."""
        return 2 * self._estimate_forward_flops(input_dim, hidden_dims, batch_size)

    def _compute_parameter_bytes(self, parameters: List[np.ndarray]) -> int:
        """Compute bytes for parameter list (float32)."""
        total_params = sum(p.size for p in parameters)
        return total_params * 4  # float32 = 4 bytes


class MemoryTracker:
    """
    Track memory usage during training.

    Useful for detecting memory leaks and optimizing data loading.
    Requires psutil package.
    """

    def __init__(self):
        """Initialize memory tracker."""
        if not HAS_PSUTIL:
            raise ImportError(
                "MemoryTracker requires psutil. Install with: pip install psutil"
            )
        self.process = psutil.Process(os.getpid())
        self.snapshots = []

    def snapshot(self, label: str = "") -> Dict[str, float]:
        """
        Take a memory snapshot.

        Args:
            label: Optional label for the snapshot

        Returns:
            Memory metrics dictionary
        """
        if not HAS_PSUTIL:
            return {'label': label, 'rss_mb': 0, 'vms_mb': 0, 'timestamp': time()}

        memory_info = self.process.memory_info()

        snapshot = {
            'label': label,
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'timestamp': time()
        }

        self.snapshots.append(snapshot)

        return snapshot

    def get_peak_memory_mb(self) -> float:
        """
        Get peak memory usage (RSS).

        Returns:
            Peak memory in MB
        """
        if not self.snapshots:
            return 0.0

        return max(s['rss_mb'] for s in self.snapshots)

    def get_summary(self) -> Dict[str, float]:
        """
        Get memory usage summary.

        Returns:
            Summary dictionary
        """
        if not self.snapshots:
            return {
                'peak_rss_mb': 0,
                'avg_rss_mb': 0,
                'n_snapshots': 0
            }

        return {
            'peak_rss_mb': self.get_peak_memory_mb(),
            'avg_rss_mb': np.mean([s['rss_mb'] for s in self.snapshots]),
            'n_snapshots': len(self.snapshots)
        }


def compare_compute_budgets(
    trackers: Dict[str, ComputeTracker]
) -> Dict[str, Dict[str, float]]:
    """
    Compare compute budgets across methods.

    Args:
        trackers: Dictionary mapping method names to ComputeTrackers

    Returns:
        Comparison dictionary
    """
    comparison = {}

    for method_name, tracker in trackers.items():
        summary = tracker.get_summary()
        comparison[method_name] = summary

    # Add relative comparisons
    if trackers:
        baseline_method = list(trackers.keys())[0]
        baseline_flops = trackers[baseline_method].get_summary()['total_flops']

        for method_name, tracker in trackers.items():
            method_flops = tracker.get_summary()['total_flops']
            comparison[method_name]['relative_flops'] = (
                method_flops / baseline_flops if baseline_flops > 0 else 0
            )

    return comparison
