"""
Unit Tests for Utility Functions

Tests metrics aggregation, logging, and helper functions.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.utils import (
    weighted_average,
    aggregate_q,
    save_metrics,
    load_metrics,
    TensorBoardLogger,
    compute_fraud_metrics,
    set_seed,
)


def test_weighted_average_basic():
    """Test basic weighted average calculation."""
    metrics = [
        (100, {"accuracy": 0.8, "loss": 0.5}),
        (200, {"accuracy": 0.9, "loss": 0.4}),
    ]

    total_samples, aggregated = weighted_average(metrics)

    # Verify total samples
    assert total_samples == 300

    # Verify weighted averages
    expected_accuracy = (100 * 0.8 + 200 * 0.9) / 300
    expected_loss = (100 * 0.5 + 200 * 0.4) / 300

    assert abs(aggregated["accuracy"] - expected_accuracy) < 1e-6
    assert abs(aggregated["loss"] - expected_loss) < 1e-6


def test_weighted_average_empty():
    """Test weighted average with empty metrics."""
    total_samples, aggregated = weighted_average([])

    assert total_samples == 0
    assert aggregated == {}


def test_weighted_average_single_client():
    """Test weighted average with single client."""
    metrics = [(100, {"accuracy": 0.8, "loss": 0.5})]

    total_samples, aggregated = weighted_average(metrics)

    assert total_samples == 100
    assert aggregated["accuracy"] == 0.8
    assert aggregated["loss"] == 0.5


def test_aggregate_q_median():
    """Test quantile aggregation for median."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    median = aggregate_q(values, 0.5)

    assert median == 3.0


def test_aggregate_q_percentile():
    """Test quantile aggregation for percentile."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    q25 = aggregate_q(values, 0.25)
    q75 = aggregate_q(values, 0.75)

    assert q25 == 2.0
    assert q75 == 4.0


def test_compute_fraud_metrics_perfect():
    """Test fraud metrics computation with perfect predictions."""
    predictions = np.array([0, 1, 0, 1, 0, 1])
    targets = np.array([0, 1, 0, 1, 0, 1])

    metrics = compute_fraud_metrics(predictions, targets)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_fraud_metrics_imperfect():
    """Test fraud metrics computation with imperfect predictions."""
    predictions = np.array([0, 1, 0, 0, 0, 1])  # One false positive
    targets = np.array([0, 1, 0, 1, 0, 1])     # One false negative

    metrics = compute_fraud_metrics(predictions, targets)

    # Check metrics are in valid range
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_compute_fraud_metrics_with_probabilities():
    """Test metrics computation with probability predictions."""
    predictions = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.8])  # Probabilities
    targets = np.array([0, 1, 0, 1, 0, 1])

    metrics = compute_fraud_metrics(predictions, targets)

    # Check metrics are valid (should auto-convert to binary)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_save_and_load_metrics(tmp_path):
    """Test saving and loading metrics to/from JSON."""
    import json

    metrics = {
        "accuracy": [0.8, 0.85, 0.9],
        "loss": [0.5, 0.4, 0.3],
    }

    filepath = tmp_path / "metrics.json"

    # Save metrics
    save_metrics(metrics, filepath)

    # Verify file exists and is valid JSON
    assert filepath.exists()

    # Load metrics
    loaded_metrics = load_metrics(filepath)

    # Verify loaded metrics match original
    assert loaded_metrics == metrics


def test_tensorboard_logger_initialization(tmp_path):
    """Test TensorBoard logger initialization."""
    log_dir = str(tmp_path)
    logger = TensorBoardLogger(log_dir, "test_experiment")

    assert logger.log_path == tmp_path / "test_experiment"
    assert logger.current_round == 0

    logger.close()


def test_tensorboard_logger_log_metrics(tmp_path):
    """Test TensorBoard logger metrics logging."""
    log_dir = str(tmp_path)
    logger = TensorBoardLogger(log_dir, "test_experiment")

    # Log some metrics
    logger.log_metrics(
        round_num=1,
        metrics={"accuracy": 0.8, "loss": 0.5},
        phase="train",
    )

    logger.log_metrics(
        round_num=1,
        metrics={"accuracy": 0.75, "loss": 0.55},
        phase="eval",
    )

    # Verify log directory was created
    assert logger.log_path.exists()

    logger.close()


def test_tensorboard_logger_log_training(tmp_path):
    """Test TensorBoard logger training metrics."""
    log_dir = str(tmp_path)
    logger = TensorBoardLogger(log_dir, "test_experiment")

    logger.log_training(
        round_num=1,
        loss=0.5,
        accuracy=0.8,
        precision=0.85,
        recall=0.75,
        f1=0.8,
    )

    assert logger.current_round == 1

    logger.close()


def test_set_seed():
    """Test random seed setting for reproducibility."""
    import random

    set_seed(42)

    # Get some random values
    val1 = random.random()
    val2 = np.random.rand()

    # Reset seed
    set_seed(42)

    # Get same random values
    val1_again = random.random()
    val2_again = np.random.rand()

    # Verify they match
    assert val1 == val1_again
    assert val2 == val2_again


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
