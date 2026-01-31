"""
Metrics calculation and comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json


def compute_all_metrics(
    local_results: Dict,
    fl_results: Dict,
    centralized_results: Dict
) -> pd.DataFrame:
    """
    Create comprehensive comparison table.

    Args:
        local_results: Results from local baseline
        fl_results: Results from federated learning
        centralized_results: Results from centralized model

    Returns:
        DataFrame with all metrics
    """
    comparison_data = []

    # Get list of banks
    bank_ids = set(local_results.keys())

    for bank_id in sorted(bank_ids):
        row = {'bank_id': bank_id}

        # Local metrics
        if bank_id in local_results:
            local_metrics = local_results[bank_id]['test_metrics']
            row['local_auc'] = local_metrics['auc_roc']
            row['local_f1'] = local_metrics['f1']
            row['local_precision'] = local_metrics['precision']
            row['local_recall'] = local_metrics['recall']
        else:
            row['local_auc'] = None
            row['local_f1'] = None

        # Federated metrics
        if 'per_bank_metrics' in fl_results and bank_id in fl_results['per_bank_metrics']:
            fl_metrics_dict = fl_results['per_bank_metrics'][bank_id]
            row['fl_auc'] = fl_metrics_dict.get('auc_roc', [None])[-1]
            row['fl_f1'] = fl_metrics_dict.get('f1', [None])[-1]
        else:
            row['fl_auc'] = None
            row['fl_f1'] = None

        # Centralized metrics (per-bank performance of centralized model)
        if 'per_bank_metrics' in centralized_results:
            if bank_id in centralized_results['per_bank_metrics']:
                cent_metrics = centralized_results['per_bank_metrics'][bank_id]
                row['centralized_auc'] = cent_metrics['auc_roc']
                row['centralized_f1'] = cent_metrics['f1']
            else:
                row['centralized_auc'] = None
                row['centralized_f1'] = None
        else:
            # Use overall centralized metrics for all banks
            cent_metrics = centralized_results.get('test_metrics', {})
            row['centralized_auc'] = cent_metrics.get('auc_roc', None)
            row['centralized_f1'] = cent_metrics.get('f1', None)

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Calculate improvements
    df['fl_vs_local_auc_improvement'] = (
        (df['fl_auc'] - df['local_auc']) / df['local_auc'] * 100
    ).round(2)

    df['fl_vs_local_f1_improvement'] = (
        (df['fl_f1'] - df['local_f1']) / df['local_f1'] * 100
    ).round(2)

    df['centralized_gap_auc'] = (
        (df['centralized_auc'] - df['fl_auc']) / df['centralized_auc'] * 100
    ).round(2)

    return df


def compute_improvement(
    comparison_df: pd.DataFrame
) -> Dict[str, Dict]:
    """
    Compute improvement statistics.

    Args:
        comparison_df: DataFrame from compute_all_metrics

    Returns:
        Dictionary with improvement statistics
    """
    # Remove rows with None values
    valid_df = comparison_df.dropna()

    improvements = {}

    # FL vs Local improvements
    improvements['fl_vs_local_auc'] = {
        'mean': valid_df['fl_vs_local_auc_improvement'].mean(),
        'std': valid_df['fl_vs_local_auc_improvement'].std(),
        'min': valid_df['fl_vs_local_auc_improvement'].min(),
        'max': valid_df['fl_vs_local_auc_improvement'].max()
    }

    improvements['fl_vs_local_f1'] = {
        'mean': valid_df['fl_vs_local_f1_improvement'].mean(),
        'std': valid_df['fl_vs_local_f1_improvement'].std(),
        'min': valid_df['fl_vs_local_f1_improvement'].min(),
        'max': valid_df['fl_vs_local_f1_improvement'].max()
    }

    # Centralized gap (how close FL is to upper bound)
    improvements['centralized_gap_auc'] = {
        'mean': valid_df['centralized_gap_auc'].mean(),
        'std': valid_df['centralized_gap_auc'].std(),
        'min': valid_df['centralized_gap_auc'].min(),
        'max': valid_df['centralized_gap_auc'].max()
    }

    return improvements


def create_comparison_table(
    local_results: Dict,
    fl_results: Dict,
    centralized_results: Dict
) -> pd.DataFrame:
    """
    Create formatted comparison table for README.

    Args:
        local_results: Results from local baseline
        fl_results: Results from federated learning
        centralized_results: Results from centralized model

    Returns:
        Formatted DataFrame
    """
    comparison_df = compute_all_metrics(
        local_results, fl_results, centralized_results
    )

    # Select and rename columns
    table_df = comparison_df[[
        'bank_id',
        'local_auc',
        'fl_auc',
        'centralized_auc',
        'fl_vs_local_auc_improvement'
    ]].copy()

    # Format values
    table_df['local_auc'] = table_df['local_auc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    table_df['fl_auc'] = table_df['fl_auc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    table_df['centralized_auc'] = table_df['centralized_auc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    table_df['fl_vs_local_auc_improvement'] = table_df['fl_vs_local_auc_improvement'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )

    table_df.columns = [
        'Bank',
        'Local AUC',
        'Federated AUC',
        'Centralized AUC',
        'FL vs Local'
    ]

    return table_df


def calculate_aggregate_metrics(
    local_results: Dict,
    fl_results: Dict,
    centralized_results: Dict
) -> Dict:
    """
    Calculate aggregate metrics across all approaches.

    Args:
        local_results: Results from local baseline
        fl_results: Results from federated learning
        centralized_results: Results from centralized model

    Returns:
        Dictionary with aggregate metrics
    """
    aggregates = {}

    # Local aggregate
    local_aucs = [r['test_metrics']['auc_roc'] for r in local_results.values()]
    aggregates['local'] = {
        'mean_auc': np.mean(local_aucs),
        'std_auc': np.std(local_aucs),
        'min_auc': np.min(local_aucs),
        'max_auc': np.max(local_aucs)
    }

    # Federated aggregate
    if 'final_metrics' in fl_results:
        fl_aucs = fl_results['final_metrics']['auc_roc_final'].values
        aggregates['federated'] = {
            'mean_auc': np.mean(fl_aucs),
            'std_auc': np.std(fl_aucs),
            'min_auc': np.min(fl_aucs),
            'max_auc': np.max(fl_aucs)
        }

    # Centralized aggregate
    if 'test_metrics' in centralized_results:
        cent_auc = centralized_results['test_metrics']['auc_roc']
        aggregates['centralized'] = {
            'mean_auc': cent_auc,
            'std_auc': 0.0,
            'min_auc': cent_auc,
            'max_auc': cent_auc
        }

    return aggregates


def save_metrics(
    comparison_df: pd.DataFrame,
    improvements: Dict,
    aggregates: Dict,
    output_path: str
) -> None:
    """
    Save metrics to files.

    Args:
        comparison_df: Comparison DataFrame
        improvements: Improvements dictionary
        aggregates: Aggregates dictionary
        output_path: Path to save metrics
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison_df.to_csv(output_dir / 'comparison_metrics.csv', index=False)

    # Save improvements
    with open(output_dir / 'improvements.json', 'w') as f:
        json.dump(improvements, f, indent=2)

    # Save aggregates
    with open(output_dir / 'aggregates.json', 'w') as f:
        json.dump(aggregates, f, indent=2)

    print(f"Metrics saved to {output_path}")
