"""
Compression metrics and Pareto frontier analysis.

This module provides utilities to analyze the trade-off between
compression ratio and model accuracy, enabling selection of
optimal compression strategies.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class CompressionResult:
    """
    Container for compression experiment results.

    Attributes:
        strategy_name: Name of compression strategy
        compression_ratio: Compression ratio achieved
        accuracy: Model accuracy with compression
        bandwidth_savings_pct: Percentage bandwidth saved
        final_loss: Final training loss
        training_time: Training time in seconds
        bytes_transmitted: Total bytes transmitted
    """
    strategy_name: str
    compression_ratio: float
    accuracy: float
    bandwidth_savings_pct: float
    final_loss: float
    training_time: float
    bytes_transmitted: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'compression_ratio': self.compression_ratio,
            'accuracy': self.accuracy,
            'bandwidth_savings_pct': self.bandwidth_savings_pct,
            'final_loss': self.final_loss,
            'training_time': self.training_time,
            'bytes_transmitted': self.bytes_transmitted
        }


class CompressionMetricsAnalyzer:
    """
    Analyze compression vs accuracy trade-offs.

    Computes Pareto frontier, accuracy degradation, and other metrics
    to help select optimal compression strategies.

    Example:
        >>> analyzer = CompressionMetricsAnalyzer()
        >>> analyzer.add_result(CompressionResult(
        ...     strategy_name='top_k_1%',
        ...     compression_ratio=100.0,
        ...     accuracy=0.95,
        ...     bandwidth_savings_pct=99.0,
        ...     final_loss=0.1,
        ...     training_time=100.0,
        ...     bytes_transmitted=1000
        ... ))
        >>> pareto = analyzer.get_pareto_frontier()
    """

    def __init__(self):
        """Initialize metrics analyzer."""
        self.results: List[CompressionResult] = []

    def add_result(self, result: CompressionResult) -> None:
        """
        Add a compression result.

        Args:
            result: CompressionResult to add
        """
        self.results.append(result)

    def get_pareto_frontier(
        self,
        maximize_accuracy: bool = True,
        maximize_compression: bool = True
    ) -> List[CompressionResult]:
        """
        Compute Pareto frontier (non-dominated solutions).

        A solution is on the Pareto frontier if no other solution
        is better in both accuracy and compression ratio.

        Args:
            maximize_accuracy: Whether higher accuracy is better (usually True)
            maximize_compression: Whether higher compression ratio is better (usually True)

        Returns:
            List of CompressionResult on the Pareto frontier
        """
        if not self.results:
            return []

        pareto = []
        dominated = set()

        for i, result_i in enumerate(self.results):
            is_dominated = False

            for j, result_j in enumerate(self.results):
                if i == j:
                    continue

                # Check if result_j dominates result_i
                # result_j dominates if it's better in BOTH objectives
                acc_better = (
                    result_j.accuracy > result_i.accuracy
                    if maximize_accuracy
                    else result_j.accuracy < result_i.accuracy
                )
                comp_better = (
                    result_j.compression_ratio > result_i.compression_ratio
                    if maximize_compression
                    else result_j.compression_ratio < result_i.compression_ratio
                )

                if acc_better and comp_better:
                    is_dominated = True
                    dominated.add(i)
                    break

            if not is_dominated:
                pareto.append(result_i)

        return pareto

    def calculate_accuracy_degradation(
        self,
        baseline_accuracy: float
    ) -> Dict[str, float]:
        """
        Calculate accuracy degradation relative to baseline.

        Args:
            baseline_accuracy: Accuracy without compression

        Returns:
            Dict mapping strategy names to accuracy degradation
        """
        degradation = {}

        for result in self.results:
            acc_degradation = baseline_accuracy - result.accuracy
            degradation[result.strategy_name] = acc_degradation

        return degradation

    def get_optimal_strategy(
        self,
        accuracy_threshold: Optional[float] = None,
        compression_preference: float = 0.5
    ) -> Optional[CompressionResult]:
        """
        Get optimal strategy based on preferences.

        Args:
            accuracy_threshold: Minimum acceptable accuracy (optional)
            compression_preference: Preference for compression (0-1)
                                   0 = prioritize accuracy, 1 = prioritize compression

        Returns:
            Best CompressionResult or None
        """
        # Filter by accuracy threshold
        candidates = self.results
        if accuracy_threshold is not None:
            candidates = [r for r in self.results if r.accuracy >= accuracy_threshold]

        if not candidates:
            return None

        # Calculate normalized scores
        accuracies = np.array([r.accuracy for r in candidates])
        compressions = np.array([r.compression_ratio for r in candidates])

        # Normalize to [0, 1]
        if accuracies.max() > accuracies.min():
            norm_acc = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min())
        else:
            norm_acc = np.ones_like(accuracies)

        if compressions.max() > compressions.min():
            norm_comp = (compressions - compressions.min()) / (compressions.max() - compressions.min())
        else:
            norm_comp = np.ones_like(compressions)

        # Weighted score
        scores = (1 - compression_preference) * norm_acc + compression_preference * norm_comp
        best_idx = np.argmax(scores)

        return candidates[best_idx]

    def plot_pareto_frontier(
        self,
        output_path: Optional[str] = None,
        show_baseline: bool = True,
        baseline_accuracy: Optional[float] = None
    ) -> None:
        """
        Plot Pareto frontier (compression ratio vs accuracy).

        Args:
            output_path: Path to save plot (optional)
            show_baseline: Whether to show baseline point
            baseline_accuracy: Baseline accuracy (no compression)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return

        # Extract data
        compression_ratios = [r.compression_ratio for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        labels = [r.strategy_name for r in self.results]

        # Get Pareto frontier
        pareto = self.get_pareto_frontier()
        pareto_comp = [r.compression_ratio for r in pareto]
        pareto_acc = [r.accuracy for r in pareto]

        plt.figure(figsize=(10, 6))

        # Plot all results
        plt.scatter(compression_ratios, accuracies, alpha=0.5, label='All Strategies')

        # Plot Pareto frontier
        if pareto_comp:
            plt.scatter(pareto_comp, pareto_acc, s=100, c='red', marker='*',
                       label='Pareto Frontier')

        # Plot baseline if provided
        if show_baseline and baseline_accuracy is not None:
            plt.scatter(1.0, baseline_accuracy, s=100, c='green', marker='o',
                       label='Baseline (No Compression)')

        # Annotate points
        for i, label in enumerate(labels):
            plt.annotate(label, (compression_ratios[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel('Compression Ratio')
        plt.ylabel('Accuracy')
        plt.title('Compression vs Accuracy Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()

        plt.close()

    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export results to file.

        Args:
            filepath: Path to output file
            format: Output format ('json' or 'csv')
        """
        if format == 'json':
            data = {
                'results': [r.to_dict() for r in self.results],
                'pareto_frontier': [r.to_dict() for r in self.get_pareto_frontier()]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                if not self.results:
                    return

                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def compare_strategies(
        results_dict: Dict[str, Dict]
    ) -> Dict:
        """
        Compare multiple compression strategies.

        Args:
            results_dict: Dict mapping strategy names to their metrics

        Returns:
            Dict with comparison metrics
        """
        comparison = {}

        for strategy, metrics in results_dict.items():
            comparison[strategy] = {
                'accuracy': metrics.get('accuracy', 0.0),
                'compression_ratio': metrics.get('compression_ratio', 1.0),
                'bandwidth_savings_pct': metrics.get('bandwidth_savings_pct', 0.0)
            }

        # Find best in each category
        if comparison:
            best_accuracy = max(comparison.items(),
                              key=lambda x: x[1]['accuracy'])
            best_compression = max(comparison.items(),
                                  key=lambda x: x[1]['compression_ratio'])
            best_savings = max(comparison.items(),
                              key=lambda x: x[1]['bandwidth_savings_pct'])

            comparison['_best_accuracy'] = {
                'strategy': best_accuracy[0],
                'value': best_accuracy[1]['accuracy']
            }
            comparison['_best_compression'] = {
                'strategy': best_compression[0],
                'value': best_compression[1]['compression_ratio']
            }
            comparison['_best_savings'] = {
                'strategy': best_savings[0],
                'value': best_savings[1]['bandwidth_savings_pct']
            }

        return comparison


def generate_pareto_report(
    results: List[CompressionResult],
    output_path: str,
    baseline_accuracy: Optional[float] = None
) -> None:
    """
    Generate a comprehensive report with Pareto analysis.

    Args:
        results: List of CompressionResult
        output_path: Path to output markdown file
        baseline_accuracy: Baseline accuracy (no compression)
    """
    analyzer = CompressionMetricsAnalyzer()
    for result in results:
        analyzer.add_result(result)

    pareto = analyzer.get_pareto_frontier()
    degradation = analyzer.calculate_accuracy_degradation(
        baseline_accuracy if baseline_accuracy else results[0].accuracy
    )

    with open(output_path, 'w') as f:
        f.write("# Compression Analysis Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total strategies evaluated: {len(results)}\n")
        f.write(f"- Pareto-optimal strategies: {len(pareto)}\n\n")

        f.write("## Pareto Frontier\n\n")
        f.write("| Strategy | Compression Ratio | Accuracy | Bandwidth Savings |\n")
        f.write("|----------|-------------------|----------|-------------------|\n")
        for p in pareto:
            f.write(f"| {p.strategy_name} | {p.compression_ratio:.2f}x | "
                   f"{p.accuracy:.4f} | {p.bandwidth_savings_pct:.1f}% |\n")

        f.write("\n## All Strategies\n\n")
        f.write("| Strategy | Compression Ratio | Accuracy | Degradation | Savings |\n")
        f.write("|----------|-------------------|----------|-------------|----------|\n")
        for r in results:
            degrad = degradation.get(r.strategy_name, 0.0)
            f.write(f"| {r.strategy_name} | {r.compression_ratio:.2f}x | "
                   f"{r.accuracy:.4f} | {degrad:.4f} | {r.bandwidth_savings_pct:.1f}% |\n")

        f.write("\n## Recommendations\n\n")
        if pareto:
            best_acc = max(pareto, key=lambda x: x.accuracy)
            best_comp = max(pareto, key=lambda x: x.compression_ratio)
            f.write(f"- **Best accuracy**: {best_acc.strategy_name} ({best_acc.accuracy:.4f})\n")
            f.write(f"- **Best compression**: {best_comp.strategy_name} ({best_comp.compression_ratio:.2f}x)\n")
