"""
Bandwidth tracking for communication-efficient FL experiments.

This module provides utilities to track bytes transmitted during
federated training, enabling accurate measurement of bandwidth savings.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BandwidthMetrics:
    """
    Container for bandwidth metrics.

    Attributes:
        uplink_bytes: Total bytes sent from clients to server
        downlink_bytes: Total bytes sent from server to clients
        total_bytes: Sum of uplink and downlink bytes
        compression_ratio: Compression ratio achieved
        bandwidth_savings_pct: Percentage bandwidth saved
        num_messages: Number of messages transmitted
        round_number: Round number for these metrics
    """
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    total_bytes: int = 0
    compression_ratio: float = 1.0
    bandwidth_savings_pct: float = 0.0
    num_messages: int = 0
    round_number: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'uplink_bytes': self.uplink_bytes,
            'downlink_bytes': self.downlink_bytes,
            'total_bytes': self.total_bytes,
            'compression_ratio': self.compression_ratio,
            'bandwidth_savings_pct': self.bandwidth_savings_pct,
            'num_messages': self.num_messages,
            'round_number': self.round_number
        }


class BandwidthTracker:
    """
    Track bandwidth consumption during federated training.

    Tracks bytes transmitted in both directions (uplink/downlink)
    and calculates compression metrics.

    Example:
        >>> tracker = BandwidthTracker()
        >>> tracker.log_uplink(bytes_sent=1000, compressed_bytes=100, round_num=1)
        >>> tracker.log_downlink(bytes_sent=500, compressed_bytes=50, round_num=1)
        >>> metrics = tracker.get_round_metrics(round_num=1)
        >>> print(f"Total bytes: {metrics.total_bytes}")
    """

    def __init__(self):
        """Initialize bandwidth tracker."""
        self.round_metrics: Dict[int, BandwidthMetrics] = {}
        self.cumulative_uplink = 0
        self.cumulative_downlink = 0
        self.cumulative_compression_ratios: List[float] = []

    def log_uplink(
        self,
        bytes_sent: int,
        compressed_bytes: int,
        round_num: int,
        client_id: Optional[str] = None
    ) -> None:
        """
        Log uplink transmission (client -> server).

        Args:
            bytes_sent: Original bytes (before compression)
            compressed_bytes: Bytes after compression
            round_num: Round number
            client_id: Optional client identifier
        """
        if round_num not in self.round_metrics:
            self.round_metrics[round_num] = BandwidthMetrics(round_number=round_num)

        metrics = self.round_metrics[round_num]
        metrics.uplink_bytes += compressed_bytes
        metrics.total_bytes += compressed_bytes
        metrics.num_messages += 1

        # Calculate compression ratio
        if compressed_bytes > 0:
            compression_ratio = bytes_sent / compressed_bytes
            metrics.compression_ratio = compression_ratio
            self.cumulative_compression_ratios.append(compression_ratio)

        # Calculate bandwidth savings
        if bytes_sent > 0:
            savings = (1 - compressed_bytes / bytes_sent) * 100
            metrics.bandwidth_savings_pct = savings

        # Update cumulative
        self.cumulative_uplink += compressed_bytes

    def log_downlink(
        self,
        bytes_sent: int,
        compressed_bytes: int,
        round_num: int
    ) -> None:
        """
        Log downlink transmission (server -> client).

        Args:
            bytes_sent: Original bytes (before compression)
            compressed_bytes: Bytes after compression
            round_num: Round number
        """
        if round_num not in self.round_metrics:
            self.round_metrics[round_num] = BandwidthMetrics(round_number=round_num)

        metrics = self.round_metrics[round_num]
        metrics.downlink_bytes += compressed_bytes
        metrics.total_bytes += compressed_bytes
        metrics.num_messages += 1

        # Update cumulative
        self.cumulative_downlink += compressed_bytes

    def get_round_metrics(self, round_num: int) -> Optional[BandwidthMetrics]:
        """
        Get metrics for a specific round.

        Args:
            round_num: Round number

        Returns:
            BandwidthMetrics for the round, or None if not found
        """
        return self.round_metrics.get(round_num)

    def get_cumulative_metrics(self) -> Dict:
        """
        Get cumulative metrics across all rounds.

        Returns:
            Dict with cumulative statistics
        """
        total_bytes = self.cumulative_uplink + self.cumulative_downlink

        avg_compression_ratio = (
            np.mean(self.cumulative_compression_ratios)
            if self.cumulative_compression_ratios else 1.0
        )

        return {
            'total_uplink_bytes': self.cumulative_uplink,
            'total_downlink_bytes': self.cumulative_downlink,
            'total_bytes': total_bytes,
            'avg_compression_ratio': avg_compression_ratio,
            'total_messages': sum(m.num_messages for m in self.round_metrics.values()),
            'num_rounds': len(self.round_metrics)
        }

    def get_all_rounds_metrics(self) -> List[Dict]:
        """
        Get metrics for all rounds.

        Returns:
            List of dicts, one per round
        """
        return [m.to_dict() for m in self.round_metrics.values()]

    def calculate_cost_savings(
        self,
        cost_per_gb: float = 0.01
    ) -> Dict:
        """
        Calculate cost savings from compression.

        Args:
            cost_per_gb: Cost per GB of data transfer (default: $0.01/GB)

        Returns:
            Dict with cost savings information
        """
        # Calculate total uncompressed bytes (using compression ratio)
        if self.cumulative_compression_ratios:
            avg_ratio = np.mean(self.cumulative_compression_ratios)
            uncompressed_bytes = self.cumulative_uplink * avg_ratio
        else:
            uncompressed_bytes = self.cumulative_uplink

        # Convert to GB
        compressed_gb = self.cumulative_uplink / (1024**3)
        uncompressed_gb = uncompressed_bytes / (1024**3)

        # Calculate costs
        compressed_cost = compressed_gb * cost_per_gb
        uncompressed_cost = uncompressed_gb * cost_per_gb
        savings = uncompressed_cost - compressed_cost

        return {
            'compressed_gb': compressed_gb,
            'uncompressed_gb': uncompressed_gb,
            'compressed_cost': compressed_cost,
            'uncompressed_cost': uncompressed_cost,
            'cost_savings': savings,
            'cost_savings_pct': (savings / uncompressed_cost * 100)
            if uncompressed_cost > 0 else 0.0
        }

    def export_to_csv(self, filepath: str) -> None:
        """
        Export metrics to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'round_number',
                'uplink_bytes',
                'downlink_bytes',
                'total_bytes',
                'compression_ratio',
                'bandwidth_savings_pct',
                'num_messages'
            ])
            writer.writeheader()
            for metrics in self.round_metrics.values():
                writer.writerow(metrics.to_dict())

    def reset(self) -> None:
        """Reset all tracking."""
        self.round_metrics = {}
        self.cumulative_uplink = 0
        self.cumulative_downlink = 0
        self.cumulative_compression_ratios = []


class BandwidthComparator:
    """
    Compare bandwidth consumption across different compression strategies.

    Example:
        >>> comparator = BandwidthComparator()
        >>> comparator.log_strategy('baseline', tracker_baseline)
        >>> comparator.log_strategy('top_k', tracker_topk)
        >>> comparison = comparator.compare()
        >>> print(comparison)
    """

    def __init__(self):
        """Initialize bandwidth comparator."""
        self.strategies: Dict[str, BandwidthTracker] = {}

    def log_strategy(self, name: str, tracker: BandwidthTracker) -> None:
        """
        Log a strategy's bandwidth tracker.

        Args:
            name: Strategy name
            tracker: BandwidthTracker instance
        """
        self.strategies[name] = tracker

    def compare(self) -> Dict:
        """
        Compare all strategies.

        Returns:
            Dict with comparison metrics
        """
        comparison = {}

        for name, tracker in self.strategies.items():
            metrics = tracker.get_cumulative_metrics()
            comparison[name] = metrics

        # Calculate savings relative to baseline
        if 'baseline' in self.strategies:
            baseline_bytes = self.strategies['baseline'].cumulative_uplink
            for name, tracker in self.strategies.items():
                if name != 'baseline':
                    bytes_saved = baseline_bytes - tracker.cumulative_uplink
                    comparison[name]['bytes_saved_vs_baseline'] = bytes_saved
                    comparison[name]['pct_savings_vs_baseline'] = (
                        bytes_saved / baseline_bytes * 100
                    )

        return comparison

    def plot_comparison(
        self,
        output_path: Optional[str] = None,
        metric: str = 'total_bytes'
    ) -> None:
        """
        Plot comparison across strategies.

        Args:
            output_path: Path to save plot (optional)
            metric: Metric to plot ('total_bytes', 'compression_ratio', etc.)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return

        names = list(self.strategies.keys())
        values = []

        for name in names:
            metrics = self.strategies[name].get_cumulative_metrics()
            values.append(metrics.get(metric, 0))

        plt.figure(figsize=(10, 6))
        plt.bar(names, values)
        plt.xlabel('Strategy')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} by Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

        plt.close()
