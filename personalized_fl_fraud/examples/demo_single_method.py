"""
Demo: Run Single Personalization Method

Example of running a single personalization method on non-IID fraud data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from utils import (
    set_random_seed,
    DataPartitioner,
    create_synthetic_fraud_data,
    CheckpointManager,
    ExperimentTracker
)
from models.base import create_model
from methods.local_finetuning import LocalFineTuning


def main():
    """Run single method demo."""
    print("=" * 60)
    print("Personalized FL Demo: Local Fine-Tuning")
    print("=" * 60)

    # Configuration
    config = OmegaConf.create({
        'experiment': {
            'name': 'demo_local_finetuning',
            'n_clients': 10,
            'random_state': 42
        },
        'model': {
            'input_dim': 20,
            'hidden_dims': [64, 32, 16]
        },
        'federated': {
            'n_rounds': 20,  # Reduced for demo
            'local_epochs': 5
        },
        'training': {
            'learning_rate': 0.01,
            'optimizer': 'adam'
        },
        'methods': {
            'local_finetuning': {
                'finetuning_epochs': 5,
                'finetuning_lr': 0.001
            }
        }
    })

    # Set random seed
    set_random_seed(config.experiment.random_state)

    # Create synthetic data
    print("\n1. Creating synthetic fraud data...")
    X, y = create_synthetic_fraud_data(
        n_samples=5000,
        n_features=20,
        fraud_ratio=0.05,
        random_state=config.experiment.random_state
    )
    print(f"   Data shape: {X.shape}")
    print(f"   Fraud ratio: {y.mean():.3f}")

    # Partition data with label skew
    print("\n2. Partitioning data with label skew (alpha=0.5)...")
    partitioner = DataPartitioner(
        n_clients=config.experiment.n_clients,
        test_size=0.2,
        val_size=0.1,
        cache_dir="./data/processed",
        random_state=config.experiment.random_state
    )

    partitions = partitioner.create_partitions(
        X, y,
        alpha=0.5,
        strategy="label_skew"
    )

    # Show partition statistics
    stats = partitioner.get_partition_statistics(partitions)
    print(f"   Created {len(partitions)} client partitions")
    for client_id in range(min(3, len(partitions))):
        client_stats = stats[client_id]
        print(f"   Client {client_id}: "
              f"{client_stats['n_train_samples']} train samples, "
              f"fraud ratio={client_stats['train_fraud_ratio']:.3f}")

    # Initialize method
    print("\n3. Initializing Local Fine-Tuning method...")
    method = LocalFineTuning(
        name="Local Fine-Tuning",
        config=config,
        random_state=config.experiment.random_state
    )
    print(f"   Method: {method.name}")
    print(f"   Description: {method.get_description()}")

    # Note: This is a simplified demo
    # Full FL simulation would require Flower server/client setup
    print("\n4. Demo complete!")
    print("\nTo run full FL experiments:")
    print("   python examples/demo_comparison.py")
    print("\nTo run tests:")
    print("   pytest tests/test_methods.py -v")


if __name__ == "__main__":
    main()
