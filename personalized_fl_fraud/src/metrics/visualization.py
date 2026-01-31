"""
Visualization Functions for Personalized FL

Provides plotting functions for:
1. Violin plots of per-client performance
2. Personalization vs generalization trade-off curves
3. Alpha sensitivity analysis
"""

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_client_violin(
    metrics_by_method: Dict[str, Dict[float, Dict[int, Dict[str, float]]]],
    metric_name: str = "auc",
    alpha: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create violin plot showing per-client performance distribution.

    Args:
        metrics_by_method: Nested dict {method: {alpha: {client_id: {metric: value}}}}
        metric_name: Metric to plot (default: 'auc')
        alpha: Alpha value to plot (if None, use first available)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Set style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    plot_data = []
    methods = []

    for method_name, alpha_dict in metrics_by_method.items():
        # Select alpha value
        if alpha is None:
            selected_alpha = next(iter(alpha_dict.keys()))
        else:
            selected_alpha = alpha

        if selected_alpha not in alpha_dict:
            continue

        client_metrics = alpha_dict[selected_alpha]

        # Extract metric values
        values = [
            client_metrics.get(client_id, {}).get(metric_name, 0)
            for client_id in sorted(client_metrics.keys())
        ]

        plot_data.append(values)
        methods.append(method_name)

    # Create violin plots
    parts = ax.violinplot(
        plot_data,
        positions=range(len(methods)),
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    # Styling
    colors = sns.color_palette("husl", len(methods))
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f'Per-Client {metric_name.upper()} Distribution')

    if alpha:
        ax.set_title(f'Per-Client {metric_name.upper()} Distribution (α={alpha})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_personalization_vs_generalization(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot personalization benefit vs generalization trade-off.

    Args:
        results: Results dictionary with global and personalized metrics
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data (this is a placeholder - actual implementation depends on results format)
    methods = list(results.keys())

    global_performance = []
    personalized_performance = []

    for method in methods:
        if method in results:
            method_results = results[method]
            # Extract global vs personalized performance
            # This will vary based on actual results structure
            global_performance.append(method_results.get('global_auc', 0))
            personalized_performance.append(method_results.get('personalized_auc', 0))

    # Plot
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, global_performance, width, label='Global Model', alpha=0.8)
    ax.bar(x + width/2, personalized_performance, width, label='Personalized', alpha=0.8)

    ax.set_ylabel('AUC')
    ax.set_title('Personalization vs Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_alpha_sensitivity(
    results: Dict[str, Dict[float, Dict[str, float]]],
    metric_name: str = "auc",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot method performance across different alpha (non-IID) values.

    Args:
        results: Dict {method: {alpha: {metric: value}}}
        metric_name: Metric to plot
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique alpha values and methods
    alpha_values = sorted(set(
        alpha
        for method_results in results.values()
        for alpha in method_results.keys()
    ))

    methods = list(results.keys())

    # Plot each method
    for method in methods:
        if method not in results:
            continue

        method_results = results[method]
        values = []

        for alpha in alpha_values:
            if alpha in method_results:
                values.append(method_results[alpha].get(metric_name, 0))
            else:
                values.append(0)

        ax.plot(alpha_values, values, marker='o', label=method, linewidth=2)

    ax.set_xlabel('Alpha (Dirichlet Concentration)')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f'Method Performance vs Non-IID Level')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add annotation about alpha meaning
    ax.text(
        0.02, 0.98,
        'Lower α = More Non-IID\nHigher α = More IID',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_client_radar(
    per_client_metrics: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10)
) -> plt.Figure:
    """
    Create radar chart comparing per-client performance.

    Args:
        per_client_metrics: Dict {client_id: {metric: value}}
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    from math import pi

    # Select clients to plot (limit to top 10 for readability)
    client_ids = sorted(per_client_metrics.keys())[:10]

    # Select metrics to plot
    metric_names = ['auc', 'pr_auc', 'f1_score', 'precision', 'recall']

    # Prepare data
    values = []
    for client_id in client_ids:
        client_values = [
            per_client_metrics[client_id].get(m, 0)
            for m in metric_names
        ]
        values.append(client_values)

    # Create radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(metric_names)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Plot each client
    colors = plt.cm.viridis(np.linspace(0, 1, len(client_ids)))

    for client_id, client_values, color in zip(client_ids, values, colors):
        client_values += client_values[:1]
        ax.plot(angles, client_values, 'o-', linewidth=1, label=f'Client {client_id}', color=color)
        ax.fill(angles, client_values, alpha=0.1, color=color)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('Per-Client Performance Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_curves(
    metrics_by_round: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot training curves across rounds for each method.

    Args:
        metrics_by_round: Dict {method: [{round, loss, auc, ...}, ...]}
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    ax = axes[0]
    for method, round_metrics in metrics_by_round.items():
        rounds = [m['round'] for m in round_metrics if 'round' in m]
        losses = [m.get('loss', 0) for m in round_metrics if 'round' in m]
        ax.plot(rounds, losses, marker='o', label=method, linewidth=2)

    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot AUC
    ax = axes[1]
    for method, round_metrics in metrics_by_round.items():
        rounds = [m['round'] for m in round_metrics if 'round' in m]
        aucs = [m.get('auc', 0) for m in round_metrics if 'round' in m]
        ax.plot(rounds, aucs, marker='o', label=method, linewidth=2)

    ax.set_xlabel('Round')
    ax.set_ylabel('AUC')
    ax.set_title('Performance (AUC)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
