"""
Visualization utilities for training results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with 'train_losses', 'val_losses', 'val_aucs', etc.
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_losses']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC-ROC
    if 'val_aucs' in history:
        axes[2].plot(epochs, history['val_aucs'], 'g-', label='Val AUC', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].set_title('AUC-ROC over Epochs')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_chart(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> None:
    """
    Plot bar chart comparing models.

    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        metric_names: List of metrics to plot
        save_path: Path to save figure
        title: Plot title
    """
    models = list(metrics_dict.keys())
    num_models = len(models)
    num_metrics = len(metric_names)

    x = np.arange(num_metrics)
    width = 0.8 / num_models

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, num_models))

    for i, model in enumerate(models):
        values = [metrics_dict[model].get(m, 0) for m in metric_names]
        offset = (i - num_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i])

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gradient_leakage(
    leakage_metrics: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot gradient leakage risk over training.

    Args:
        leakage_metrics: List of leakage metric dictionaries
        save_path: Path to save figure
    """
    epochs = [m['epoch'] for m in leakage_metrics]
    risk_a = [m['party_a']['leakage_risk_percent'] for m in leakage_metrics]
    risk_b = [m['party_b']['leakage_risk_percent'] for m in leakage_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, risk_a, 'r-', label='Party A Embedding', linewidth=2, marker='o')
    ax.plot(epochs, risk_b, 'b-', label='Party B Embedding', linewidth=2, marker='s')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Leakage Risk (%)')
    ax.set_title('Gradient Leakage Risk Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_privacy_utility_tradeoff(
    privacy_risks: Dict[str, float],
    performance: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot privacy-utility tradeoff for different methods.

    Args:
        privacy_risks: Dictionary of {method: privacy_risk_score}
        performance: Dictionary of {method: performance_score (e.g., AUC)}
        save_path: Path to save figure
    """
    methods = list(privacy_risks.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        ax.scatter(
            privacy_risks[method],
            performance[method],
            s=200,
            alpha=0.7,
            label=method
        )
        ax.annotate(
            method,
            (privacy_risks[method], performance[method]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax.set_xlabel('Privacy Risk (Leakage %)')
    ax.set_ylabel('Performance (AUC-ROC)')
    ax.set_title('Privacy-Utility Tradeoff')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
