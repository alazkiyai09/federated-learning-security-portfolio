"""
Evaluation metrics for fraud detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from typing import Dict, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities for class 1

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    y_true_dict: Dict[str, np.ndarray],
    y_prob_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true_dict: Dictionary of {model_name: true_labels}
        y_prob_dict: Dictionary of {model_name: predicted_probs}
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    for name in y_true_dict.keys():
        y_true = y_true_dict[name]
        y_prob = y_prob_dict[name]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curves(
    y_true_dict: Dict[str, np.ndarray],
    y_prob_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        y_true_dict: Dictionary of {model_name: true_labels}
        y_prob_dict: Dictionary of {model_name: predicted_probs}
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    for name in y_true_dict.keys():
        y_true = y_true_dict[name]
        y_prob = y_prob_dict[name]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc = average_precision_score(y_true, y_prob)

        plt.plot(recall, precision, label=f'{name} (AP = {auc:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_metrics_table(
    model_metrics: Dict[str, Dict[str, float]],
    title: str = "Model Comparison"
) -> None:
    """
    Print metrics comparison table.

    Args:
        model_metrics: Dictionary of {model_name: {metric: value}}
        title: Table title
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"{'-'*80}")

    for model_name, metrics in model_metrics.items():
        print(f"{model_name:<30} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} "
              f"{metrics.get('auc_roc', 0.0):>10.4f}")

    print(f"{'='*80}\n")
