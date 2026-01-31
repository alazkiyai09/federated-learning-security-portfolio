"""
Convergence tracking and visualization for federated learning.

Tracks metrics across rounds and generates publication-quality plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


class ConvergenceTracker:
    """
    Track and visualize federated learning convergence.

    Records metrics per training round and generates plots for analysis.

    Attributes:
        metrics: Dictionary storing metric history

    Example:
        >>> tracker = ConvergenceTracker()
        >>> tracker.update(0, {'train_loss': 2.5, 'test_accuracy': 0.65})
        >>> tracker.plot_convergence('results/convergence.png')
    """

    def __init__(self):
        """Initialize tracker with empty metrics."""
        self.metrics = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'num_clients': [],
            'total_samples': []
        }

    def update(
        self,
        round_num: int,
        metrics: Dict[str, float],
        num_clients: Optional[int] = None,
        total_samples: Optional[int] = None
    ) -> None:
        """
        Record metrics for a training round.

        Args:
            round_num: Current round number
            metrics: Dictionary with any of:
                - train_loss
                - train_accuracy
                - test_loss
                - test_accuracy
            num_clients: Number of clients trained (optional)
            total_samples: Total training samples (optional)
        """
        self.metrics['round'].append(round_num)

        # Initialize with None if not provided
        for key in ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']:
            if key in metrics:
                self.metrics[key].append(metrics[key])
            else:
                self.metrics[key].append(None)

        self.metrics['num_clients'].append(num_clients)
        self.metrics['total_samples'].append(total_samples)

    def plot_convergence(
        self,
        save_path: str,
        figsize: tuple = (12, 5),
        dpi: int = 150,
        show_plot: bool = False
    ) -> None:
        """
        Create convergence plots with training and test metrics.

        Generates a 2-subplot figure:
        - Left: Loss over rounds
        - Right: Accuracy over rounds

        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            dpi: Resolution for saved figure
            show_plot: Whether to display plot interactively
        """
        rounds = self.metrics['round']

        # Remove None values for each metric
        train_loss = [v for v in self.metrics['train_loss'] if v is not None]
        test_loss = [v for v in self.metrics['test_loss'] if v is not None]
        train_acc = [v for v in self.metrics['train_accuracy'] if v is not None]
        test_acc = [v for v in self.metrics['test_accuracy'] if v is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        if train_loss:
            train_rounds = rounds[:len(train_loss)]
            ax1.plot(train_rounds, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        if test_loss:
            test_rounds = rounds[:len(test_loss)]
            ax1.plot(test_rounds, test_loss, 'r-', label='Test Loss', linewidth=2, marker='s', markersize=3)

        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Convergence - Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        if train_acc:
            train_rounds = rounds[:len(train_acc)]
            ax2.plot(train_rounds, train_acc, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
        if test_acc:
            test_rounds = rounds[:len(test_acc)]
            ax2.plot(test_rounds, test_acc, 'r-', label='Test Accuracy', linewidth=2, marker='s', markersize=3)

        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training Convergence - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_loss_only(
        self,
        save_path: str,
        figsize: tuple = (8, 6),
        dpi: int = 150
    ) -> None:
        """
        Create loss-only convergence plot.

        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            dpi: Resolution for saved figure
        """
        rounds = self.metrics['round']
        train_loss = [v for v in self.metrics['train_loss'] if v is not None]
        test_loss = [v for v in self.metrics['test_loss'] if v is not None]

        plt.figure(figsize=figsize)

        if train_loss:
            train_rounds = rounds[:len(train_loss)]
            plt.plot(train_rounds, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        if test_loss:
            test_rounds = rounds[:len(test_loss)]
            plt.plot(test_rounds, test_loss, 'r-', label='Test Loss', linewidth=2, marker='s', markersize=4)

        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Federated Learning Convergence', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_accuracy_only(
        self,
        save_path: str,
        figsize: tuple = (8, 6),
        dpi: int = 150
    ) -> None:
        """
        Create accuracy-only convergence plot.

        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            dpi: Resolution for saved figure
        """
        rounds = self.metrics['round']
        train_acc = [v for v in self.metrics['train_accuracy'] if v is not None]
        test_acc = [v for v in self.metrics['test_accuracy'] if v is not None]

        plt.figure(figsize=figsize)

        if train_acc:
            train_rounds = rounds[:len(train_acc)]
            plt.plot(train_rounds, train_acc, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
        if test_acc:
            test_rounds = rounds[:len(test_acc)]
            plt.plot(test_rounds, test_acc, 'r-', label='Test Accuracy', linewidth=2, marker='s', markersize=4)

        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Federated Learning Convergence', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def save_metrics(self, path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: Path to save JSON file
        """
        # Convert to JSON-serializable format
        serializable = {}
        for key, values in self.metrics.items():
            serializable[key] = [
                float(v) if v is not None else None
                for v in values
            ]

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def load_metrics(self, path: str) -> None:
        """
        Load metrics from JSON file.

        Args:
            path: Path to load JSON file from
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self.metrics = data

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Get final round metrics.

        Returns:
            Dict with final values for each metric
        """
        final = {}

        for key in ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']:
            values = [v for v in self.metrics[key] if v is not None]
            if values:
                final[key] = values[-1]
            else:
                final[key] = None

        return final

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best metrics across all rounds.

        Returns:
            Dict with best (min for loss, max for accuracy) values
        """
        best = {}

        # For loss, minimum is best
        for loss_key in ['train_loss', 'test_loss']:
            values = [v for v in self.metrics[loss_key] if v is not None]
            if values:
                best[f'best_{loss_key}'] = min(values)
                best[f'{loss_key}_round'] = self.metrics['round'][self.metrics[loss_key].index(min(values))]

        # For accuracy, maximum is best
        for acc_key in ['train_accuracy', 'test_accuracy']:
            values = [v for v in self.metrics[acc_key] if v is not None]
            if values:
                best[f'best_{acc_key}'] = max(values)
                best[f'{acc_key}_round'] = self.metrics['round'][self.metrics[acc_key].index(max(values))]

        return best

    def print_summary(self) -> None:
        """Print summary statistics of training."""
        final = self.get_final_metrics()
        best = self.get_best_metrics()

        print("\n" + "="*60)
        print("FEDERATED LEARNING SUMMARY")
        print("="*60)

        if final['test_accuracy'] is not None:
            print(f"Final Test Accuracy:  {final['test_accuracy']:.4f}")
        if final['train_accuracy'] is not None:
            print(f"Final Train Accuracy: {final['train_accuracy']:.4f}")
        if final['test_loss'] is not None:
            print(f"Final Test Loss:      {final['test_loss']:.4f}")
        if final['train_loss'] is not None:
            print(f"Final Train Loss:     {final['train_loss']:.4f}")

        print("\nBest Performance:")
        if 'best_test_accuracy' in best:
            print(f"Best Test Accuracy:  {best['best_test_accuracy']:.4f} (Round {best['test_accuracy_round']})")
        if 'best_train_accuracy' in best:
            print(f"Best Train Accuracy: {best['best_train_accuracy']:.4f} (Round {best['train_accuracy_round']})")

        print("="*60 + "\n")

    def compare_trackers(
        self,
        other_tracker: 'ConvergenceTracker',
        save_path: str,
        label1: str = "Run 1",
        label2: str = "Run 2",
        figsize: tuple = (12, 5),
        dpi: int = 150
    ) -> None:
        """
        Compare two trackers on the same plot.

        Useful for comparing different hyperparameter settings.

        Args:
            other_tracker: Another ConvergenceTracker instance
            save_path: Path to save comparison plot
            label1: Label for this tracker
            label2: Label for other tracker
            figsize: Figure size
            dpi: Resolution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Extract metrics for both trackers
        rounds1 = self.metrics['round']
        test_acc1 = [v for v in self.metrics['test_accuracy'] if v is not None]
        test_loss1 = [v for v in self.metrics['test_loss'] if v is not None]

        rounds2 = other_tracker.metrics['round']
        test_acc2 = [v for v in other_tracker.metrics['test_accuracy'] if v is not None]
        test_loss2 = [v for v in other_tracker.metrics['test_loss'] if v is not None]

        # Plot loss
        if test_loss1:
            ax1.plot(rounds1[:len(test_loss1)], test_loss1, label=label1, linewidth=2, marker='o', markersize=3)
        if test_loss2:
            ax1.plot(rounds2[:len(test_loss2)], test_loss2, label=label2, linewidth=2, marker='s', markersize=3)

        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        if test_acc1:
            ax2.plot(rounds1[:len(test_acc1)], test_acc1, label=label1, linewidth=2, marker='o', markersize=3)
        if test_acc2:
            ax2.plot(rounds2[:len(test_acc2)], test_acc2, label=label2, linewidth=2, marker='s', markersize=3)

        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
