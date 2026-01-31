"""
Metrics computation for fraud detection.

Provides functions for computing:
- ROC AUC
- Precision-Recall AUC
- Recall at specific FPR
- F1 score
- Confusion matrix metrics
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve
)


def compute_fraud_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive fraud detection metrics.

    Args:
        predictions: Predicted probabilities (n_samples,)
        targets: Ground truth labels (n_samples,)
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary of metrics
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)

    metrics = {}

    # ROC AUC
    try:
        metrics['auc'] = float(roc_auc_score(targets, predictions))
    except ValueError:
        # Handle edge case where all labels are the same
        metrics['auc'] = 0.5

    # Precision-Recall AUC
    try:
        metrics['pr_auc'] = float(average_precision_score(targets, predictions))
    except ValueError:
        metrics['pr_auc'] = 0.0

    # Recall at 1% FPR
    metrics['recall_at_fpr_1pct'] = compute_recall_at_fpr(
        targets, predictions, target_fpr=0.01
    )

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, binary_preds, average='binary', zero_division=0
    )

    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()

    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    # Detection rate (recall)
    metrics['detection_rate'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # False alarm rate (FPR)
    metrics['false_alarm_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Positive predictive value (precision)
    metrics['ppv'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return metrics


def compute_recall_at_fpr(
    targets: np.ndarray,
    predictions: np.ndarray,
    target_fpr: float = 0.01
) -> float:
    """
    Compute recall at a specific false positive rate.

    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
        target_fpr: Target FPR (default: 0.01 for 1%)

    Returns:
        Recall at target FPR
    """
    try:
        fpr, tpr, thresholds = roc_curve(targets, predictions)

        # Find closest FPR to target
        idx = np.argmin(np.abs(fpr - target_fpr))

        return float(tpr[idx])
    except ValueError:
        return 0.0


def compute_fairness_metrics(
    per_client_metrics: Dict[int, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute fairness metrics across clients.

    Args:
        per_client_metrics: Dictionary mapping client_id to metrics

    Returns:
        Fairness metrics dictionary
    """
    if not per_client_metrics:
        return {
            'performance_variance': 0.0,
            'performance_std': 0.0,
            'worst_client_performance': 0.0,
            'best_client_performance': 0.0,
            'performance_range': 0.0,
            'gini_coefficient': 0.0
        }

    # Extract AUC values
    auc_values = [
        m.get('auc', 0.5)
        for m in per_client_metrics.values()
    ]

    # Basic statistics
    fairness = {
        'performance_mean': float(np.mean(auc_values)),
        'performance_variance': float(np.var(auc_values)),
        'performance_std': float(np.std(auc_values)),
        'worst_client_performance': float(np.min(auc_values)),
        'best_client_performance': float(np.max(auc_values)),
        'performance_range': float(np.max(auc_values) - np.min(auc_values))
    }

    # Gini coefficient (inequality measure)
    fairness['gini_coefficient'] = float(compute_gini(auc_values))

    return fairness


def compute_gini(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Args:
        values: Array of values (e.g., per-client performance)

    Returns:
        Gini coefficient (0 = perfect equality, 1 = maximal inequality)
    """
    if len(values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)
    n = len(values)

    # Compute cumulative sum
    cumulative = np.cumsum(sorted_values)
    cumulative = np.insert(cumulative, 0, 0)

    # Compute Gini coefficient
    # Using the formula: G = (2 * sum(i * y_i)) / (n * sum(y)) - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    return max(0.0, min(1.0, gini))
