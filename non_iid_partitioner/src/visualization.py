"""
Visualization utilities for partition analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, Tuple, List, Optional
from .utils import entropy, gini_coefficient


def plot_client_distribution(partitions: Dict[int, np.ndarray],
                            y: np.ndarray,
                            n_classes: Optional[int] = None,
                            title: str = "Class Distribution per Client",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing class distribution across clients.

    Args:
        partitions: Dictionary mapping client_id to sample indices
        y: Full label array
        n_classes: Number of classes (inferred from y if None)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if None, displays)

    Returns:
        matplotlib Figure object
    """
    if n_classes is None:
        n_classes = len(np.unique(y))

    n_clients = len(partitions)

    # Build distribution matrix
    dist_matrix = np.zeros((n_clients, n_classes))
    client_ids = sorted(partitions.keys())

    for client_id in client_ids:
        indices = partitions[client_id]
        client_labels = y[indices]
        unique, counts = np.unique(client_labels, return_counts=True)

        # Normalize by client size to get proportions
        client_total = len(indices)
        for label, count in zip(unique, counts):
            dist_matrix[client_id, label] = count / client_total

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_clients))
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(n_clients):
        for j in range(n_classes):
            text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def plot_quantity_distribution(partitions: Dict[int, np.ndarray],
                               title: str = "Sample Count per Client",
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart showing number of samples per client.

    Args:
        partitions: Dictionary mapping client_id to sample indices
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if None, displays)

    Returns:
        matplotlib Figure object
    """
    client_ids = sorted(partitions.keys())
    sample_counts = [len(partitions[cid]) for cid in client_ids]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(client_ids, sample_counts, color='steelblue', edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    mean_count = np.mean(sample_counts)
    ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def plot_label_distribution_comparison(partitions: Dict[int, np.ndarray],
                                       y: np.ndarray,
                                       n_classes: Optional[int] = None,
                                       figsize: Tuple[int, int] = (14, 8),
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare label distributions across clients with stacked bar chart.

    Args:
        partitions: Dictionary mapping client_id to sample indices
        y: Full label array
        n_classes: Number of classes (inferred from y if None)
        figsize: Figure size
        save_path: Path to save figure (if None, displays)

    Returns:
        matplotlib Figure object
    """
    if n_classes is None:
        n_classes = len(np.unique(y))

    n_clients = len(partitions)
    client_ids = sorted(partitions.keys())

    # Build class counts matrix
    class_counts = np.zeros((n_clients, n_classes))

    for i, client_id in enumerate(client_ids):
        indices = partitions[client_id]
        client_labels = y[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_counts[i, label] = count

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    bottom = np.zeros(n_clients)

    for class_id in range(n_classes):
        ax.bar(client_ids, class_counts[:, class_id], bottom=bottom,
              label=f'Class {class_id}', color=colors[class_id], edgecolor='black')
        bottom += class_counts[:, class_id]

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Label Distribution Across Clients', fontsize=14, fontweight='bold')
    ax.legend(title='Classes', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def compute_heterogeneity_metrics(partitions: Dict[int, np.ndarray],
                                 y: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics quantifying the heterogeneity of the partition.

    Metrics:
    - mean_label_entropy: Average entropy of label distributions (lower = more skew)
    - std_label_entropy: Standard deviation of label entropies
    - gini_coefficient: Inequality in sample sizes (0 = equal, 1 = maximal)
    - cv_samples: Coefficient of variation of sample sizes
    - max_min_ratio: Ratio of largest to smallest client size

    Args:
        partitions: Dictionary mapping client_id to sample indices
        y: Full label array

    Returns:
        Dictionary with metric names and values
    """
    n_classes = len(np.unique(y))

    # Compute label entropy for each client
    entropies = []
    sample_sizes = []

    for indices in partitions.values():
        client_labels = y[indices]
        sample_sizes.append(len(indices))

        # Compute label distribution
        unique, counts = np.unique(client_labels, return_counts=True)
        label_dist = counts / counts.sum()

        # Compute entropy
        ent = entropy(label_dist)
        entropies.append(ent)

    entropies = np.array(entropies)
    sample_sizes = np.array(sample_sizes)

    # Compute metrics
    metrics = {
        'mean_label_entropy': float(np.mean(entropies)),
        'std_label_entropy': float(np.std(entropies)),
        'gini_coefficient': float(gini_coefficient(sample_sizes)),
        'cv_samples': float(np.std(sample_sizes) / np.mean(sample_sizes)),
        'max_min_ratio': float(np.max(sample_sizes) / np.min(sample_sizes)),
        'mean_samples': float(np.mean(sample_sizes)),
        'std_samples': float(np.std(sample_sizes))
    }

    return metrics


def create_partition_report(partitions: Dict[int, np.ndarray],
                           y: np.ndarray,
                           save_dir: Optional[str] = None,
                           prefix: str = "partition") -> Dict[str, plt.Figure]:
    """
    Create a comprehensive visualization report for a partition.

    Generates:
    - Class distribution heatmap
    - Sample count bar chart
    - Stacked label distribution chart

    Args:
        partitions: Dictionary mapping client_id to sample indices
        y: Full label array
        save_dir: Directory to save figures (if None, doesn't save)
        prefix: Prefix for saved filenames

    Returns:
        Dictionary with figure names and Figure objects
    """
    figures = {}

    # Class distribution heatmap
    fig1 = plot_client_distribution(
        partitions, y,
        title="Class Distribution per Client",
        save_path=f"{save_dir}/{prefix}_heatmap.png" if save_dir else None
    )
    figures['heatmap'] = fig1

    # Quantity distribution
    fig2 = plot_quantity_distribution(
        partitions,
        title="Sample Count per Client",
        save_path=f"{save_dir}/{prefix}_quantity.png" if save_dir else None
    )
    figures['quantity'] = fig2

    # Stacked bar chart
    fig3 = plot_label_distribution_comparison(
        partitions, y,
        save_path=f"{save_dir}/{prefix}_stacked.png" if save_dir else None
    )
    figures['stacked'] = fig3

    return figures
