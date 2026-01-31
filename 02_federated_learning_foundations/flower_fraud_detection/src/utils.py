"""
Utility Functions for Flower Fraud Detection

Includes metrics aggregation, TensorBoard logging, and helper functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def weighted_average(
    metrics: List[Tuple[int, Dict[str, Any]]]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted average of metrics across clients.

    Each client's metrics are weighted by their number of samples.

    Args:
        metrics: List of (num_samples, metrics_dict) tuples from clients

    Returns:
        (num_samples, aggregated_metrics) tuple where:
        - num_samples: total number of samples across all clients
        - aggregated_metrics: dict of metric names to weighted averages
    """
    if not metrics:
        return 0, {}

    # Calculate total number of samples
    num_samples_total = sum(num_samples for num_samples, _ in metrics)

    # Aggregate each metric
    aggregated_metrics = {}
    for metric_name in metrics[0][1].keys():
        weighted_sum = sum(
            num_samples * metric_dict.get(metric_name, 0)
            for num_samples, metric_dict in metrics
        )
        aggregated_metrics[metric_name] = weighted_sum / num_samples_total

    return num_samples_total, aggregated_metrics


def aggregate_q(values: List[float], q: float) -> float:
    """
    Compute q-th quantile of values (for robust aggregation).

    Args:
        values: List of values
        q: Quantile to compute (0.5 for median)

    Returns:
        Quantile value
    """
    return np.quantile(values, q)


def save_metrics(
    metrics: Dict[str, List[float]],
    filepath: Path,
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metric name to list of values per round
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: Path) -> Dict[str, List[float]]:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to load file from

    Returns:
        Dictionary of metric name to list of values per round
    """
    with open(filepath, "r") as f:
        return json.load(f)


class TensorBoardLogger:
    """
    TensorBoard logger for tracking federated learning experiments.

    Logs metrics per round for server-side aggregation results.
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of this experiment
        """
        self.log_path = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(str(self.log_path))
        self.current_round = 0

    def log_metrics(
        self,
        round_num: int,
        metrics: Dict[str, float],
        phase: str = "server",
    ) -> None:
        """
        Log metrics for a specific round.

        Args:
            round_num: Current federated round
            metrics: Dictionary of metric names to values
            phase: Phase of training (server, client_train, client_eval)
        """
        self.current_round = round_num

        for metric_name, value in metrics.items():
            tag = f"{phase}/{metric_name}"
            self.writer.add_scalar(tag, value, round_num)

        self.writer.flush()

    def log_training(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
    ) -> None:
        """
        Log training metrics for a round.

        Args:
            round_num: Current round
            loss: Training loss
            accuracy: Training accuracy
            precision: Training precision
            recall: Training recall
            f1: Training F1 score
        """
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self.log_metrics(round_num, metrics, phase="train")

    def log_evaluation(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
    ) -> None:
        """
        Log evaluation metrics for a round.

        Args:
            round_num: Current round
            loss: Validation loss
            accuracy: Validation accuracy
            precision: Validation precision
            recall: Validation recall
            f1: Validation F1 score
        """
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self.log_metrics(round_num, metrics, phase="eval")

    def log_custom_metric(
        self,
        round_num: int,
        metric_name: str,
        value: float,
        phase: str = "custom",
    ) -> None:
        """
        Log a custom metric.

        Args:
            round_num: Current round
            metric_name: Name of the metric
            value: Value of the metric
            phase: Phase identifier
        """
        self.log_metrics(round_num, {metric_name: value}, phase=phase)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


def compute_fraud_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics for fraud detection.

    Args:
        predictions: Predicted probabilities or binary labels
        targets: Ground truth binary labels

    Returns:
        Dictionary with accuracy, precision, recall, f1, auc
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    # Convert probabilities to binary predictions
    if predictions.max() <= 1.0:
        pred_binary = (predictions > 0.5).astype(int)
    else:
        pred_binary = predictions.astype(int)

    # Flatten if needed
    pred_binary = pred_binary.flatten()
    targets = targets.flatten()

    # Compute metrics
    accuracy = accuracy_score(targets, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, pred_binary, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
