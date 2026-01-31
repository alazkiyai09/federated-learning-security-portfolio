"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_per_bank_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'auc',
    save_path: Optional[str] = None
) -> None:
    """
    Plot per-bank comparison across approaches.

    Args:
        comparison_df: DataFrame with comparison metrics
        metric: Metric to plot ('auc' or 'f1')
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bank_ids = comparison_df['bank_id'].values
    x = np.arange(len(bank_ids))
    width = 0.25

    # Extract values
    local_vals = comparison_df[f'local_{metric}'].values
    fl_vals = comparison_df[f'fl_{metric}'].values
    cent_vals = comparison_df[f'centralized_{metric}'].values

    # Plot bars
    ax.bar(x - width, local_vals, width, label='Local', color='#3498db', alpha=0.8)
    ax.bar(x, fl_vals, width, label='Federated', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, cent_vals, width, label='Centralized', color='#e74c3c', alpha=0.8)

    # Formatting
    ax.set_xlabel('Bank', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} (ROC-AUC)' if metric == 'auc' else f'{metric.upper()} Score',
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Bank {metric.upper()} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bank_ids, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (local, fl, cent) in enumerate(zip(local_vals, fl_vals, cent_vals)):
        if not np.isnan(local):
            ax.text(i - width, local, f'{local:.3f}', ha='center', va='bottom', fontsize=8)
        if not np.isnan(fl):
            ax.text(i, fl, f'{fl:.3f}', ha='center', va='bottom', fontsize=8)
        if not np.isnan(cent):
            ax.text(i + width, cent, f'{cent:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-bank comparison to {save_path}")

    plt.show()


def plot_learning_curves(
    fl_results: Dict,
    bank_ids: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curves for federated training.

    Args:
        fl_results: Results from federated training
        bank_ids: List of bank IDs to plot (if None, plot all)
        save_path: Optional path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    per_bank_metrics = fl_results['per_bank_metrics']

    if bank_ids is None:
        bank_ids = sorted(per_bank_metrics.keys())

    # Plot AUC progression
    for bank_id in bank_ids:
        if bank_id in per_bank_metrics:
            metrics = per_bank_metrics[bank_id]
            auc_values = metrics.get('auc_roc', [])
            rounds = range(1, len(auc_values) + 1)

            ax1.plot(rounds, auc_values, marker='o', label=bank_id, linewidth=2)

    ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('Federated Learning Progression (AUC)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # Plot F1 progression
    for bank_id in bank_ids:
        if bank_id in per_bank_metrics:
            metrics = per_bank_metrics[bank_id]
            f1_values = metrics.get('f1', [])
            rounds = range(1, len(f1_values) + 1)

            ax2.plot(rounds, f1_values, marker='s', label=bank_id, linewidth=2)

    ax2.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Federated Learning Progression (F1)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")

    plt.show()


def plot_fraud_analysis(
    bank_data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
) -> None:
    """
    Plot fraud analysis across banks.

    Args:
        bank_data: Dictionary mapping bank_id to DataFrame
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect statistics
    bank_stats = []
    for bank_id, df in bank_data.items():
        stats = {
            'bank_id': bank_id,
            'fraud_rate': df['is_fraud'].mean(),
            'avg_amount': df['amount'].mean(),
            'n_international': df['is_international'].sum(),
            'n_transactions': len(df)
        }
        bank_stats.append(stats)

    stats_df = pd.DataFrame(bank_stats)

    # 1. Fraud rates
    axes[0, 0].bar(stats_df['bank_id'], stats_df['fraud_rate'] * 100,
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Fraud Rate (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Fraud Rate by Bank', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Average transaction amounts
    axes[0, 1].bar(stats_df['bank_id'], stats_df['avg_amount'],
                   color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Average Amount ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Average Transaction Amount', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. International transactions
    axes[1, 0].bar(stats_df['bank_id'],
                   stats_df['n_international'] / stats_df['n_transactions'] * 100,
                   color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('International Ratio (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('International Transaction Ratio', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Data volume
    axes[1, 1].bar(stats_df['bank_id'], stats_df['n_transactions'] / 1000,
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Transactions (K)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Transaction Volume', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fraud analysis to {save_path}")

    plt.show()


def plot_improvement_analysis(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot improvement analysis for FL vs Local.

    Args:
        comparison_df: DataFrame with comparison metrics
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bank_ids = comparison_df['bank_id'].values
    improvements = comparison_df['fl_vs_local_auc_improvement'].values

    # Color bars by improvement (red = negative, green = positive)
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in improvements]

    bars = ax.bar(bank_ids, improvements, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Bank', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Federated vs Local Model Improvement', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    red_patch = mpatches.Patch(color='#e74c3c', label='Worse than Local')
    green_patch = mpatches.Patch(color='#2ecc71', label='Better than Local')
    ax.legend(handles=[red_patch, green_patch], fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement analysis to {save_path}")

    plt.show()


def create_summary_figure(
    comparison_df: pd.DataFrame,
    fl_results: Dict,
    bank_data: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None
) -> None:
    """
    Create summary figure with multiple subplots.

    Args:
        comparison_df: Comparison metrics DataFrame
        fl_results: Federated learning results
        bank_data: Bank data dictionary
        output_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid spec
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Per-bank comparison
    ax1 = fig.add_subplot(gs[0, :])
    bank_ids = comparison_df['bank_id'].values
    x = np.arange(len(bank_ids))
    width = 0.25

    local_auc = comparison_df['local_auc'].values
    fl_auc = comparison_df['fl_auc'].values
    cent_auc = comparison_df['centralized_auc'].values

    ax1.bar(x - width, local_auc, width, label='Local', color='#3498db', alpha=0.8)
    ax1.bar(x, fl_auc, width, label='Federated', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width, cent_auc, width, label='Centralized', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Bank', fontsize=11, fontweight='bold')
    ax1.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax1.set_title('Per-Bank Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bank_ids, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Learning curves
    ax2 = fig.add_subplot(gs[1, 0])
    per_bank_metrics = fl_results['per_bank_metrics']

    for bank_id in sorted(per_bank_metrics.keys())[:3]:  # Show first 3 for clarity
        metrics = per_bank_metrics[bank_id]
        auc_values = metrics.get('auc_roc', [])
        rounds = range(1, len(auc_values) + 1)
        ax2.plot(rounds, auc_values, marker='o', label=bank_id, linewidth=2)

    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax2.set_title('Federated Learning Progression', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.5, 1.0])

    # 3. Improvement analysis
    ax3 = fig.add_subplot(gs[1, 1])
    improvements = comparison_df['fl_vs_local_auc_improvement'].values
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in improvements]

    ax3.bar(bank_ids, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Bank', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax3.set_title('FL vs Local Improvement', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Silo Federated Learning: Fraud Detection Results',
                 fontsize=16, fontweight='bold', y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {output_path}")

    plt.show()
