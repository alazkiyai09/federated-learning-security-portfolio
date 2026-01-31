"""
Demo: Compare All Personalization Methods

Example of comparing multiple personalization methods on non-IID fraud data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from omegaconf import OmegaConf

from utils import set_random_seed, DataPartitioner, create_synthetic_fraud_data


def main():
    """Run method comparison demo."""
    print("=" * 70)
    print("Personalized FL Method Comparison Demo")
    print("=" * 70)

    # Configuration
    config = OmegaConf.create({
        'experiment': {
            'name': 'method_comparison',
            'n_clients': 10,
            'random_state': 42
        },
        'model': {
            'input_dim': 20,
            'hidden_dims': [64, 32, 16]
        },
        'methods': {
            'fedavg': {'enabled': True},
            'local_finetuning': {'enabled': True},
            'fedper': {'enabled': True},
            'ditto': {'enabled': True},
            'per_fedavg': {'enabled': True}
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
    print(f"   Data: {X.shape}, fraud ratio: {y.mean():.3f}")

    # Partition data at multiple alpha levels
    print("\n2. Partitioning data at multiple alpha levels...")
    alpha_values = [0.1, 0.5, 1.0, 10.0]
    partitioner = DataPartitioner(
        n_clients=config.experiment.n_clients,
        test_size=0.2,
        val_size=0.1,
        cache_dir="./data/processed",
        random_state=config.experiment.random_state
    )

    all_partitions = partitioner.create_partitions_at_multiple_alphas(
        X, y,
        alpha_values=alpha_values,
        strategy="label_skew"
    )

    print(f"   Created partitions for {len(all_partitions)} alpha values")

    # Show alpha meanings
    print("\n   Alpha values (Dirichlet concentration):")
    for alpha in alpha_values:
        partitions = all_partitions[alpha]
        stats = partitioner.get_partition_statistics(partitions)
        fraud_ratios = [s['train_fraud_ratio'] for s in stats.values()]
        print(f"   α={alpha:4.1f}: fraud ratio range = "
              f"[{min(fraud_ratios):.3f}, {max(fraud_ratios):.3f}], "
              f"std={np.std(fraud_ratios):.3f}")

    # Method comparison table
    print("\n3. Method Comparison:")
    print("-" * 70)
    print(f"{'Method':<25} {'Personalization':<20} {'Pros':<25}")
    print("-" * 70)

    methods_info = [
        ("FedAvg", "None", "Simple baseline, no personalization"),
        ("Local Fine-Tuning", "Post-hoc", "Simple, risks overfitting"),
        ("FedPer", "Classifier layer", "Clear separation, low communication"),
        ("Ditto", "Local + global", "Robust, higher memory"),
        ("Per-FedAvg", "Meta-learning", "Fast adaptation, complex"),
    ]

    for method, personalization, pros in methods_info:
        print(f"{method:<25} {personalization:<20} {pros:<25}")

    print("-" * 70)

    # Expected results
    print("\n4. Expected Results:")
    print("   - All methods should beat FedAvg on highly non-IID data (α=0.1)")
    print("   - FedPer and Ditto expected to perform best on heterogeneous data")
    print("   - Per-FedAvg should show best fast adaptation")
    print("   - Personalization benefit decreases as data becomes more IID (α→∞)")

    print("\n5. Next Steps:")
    print("   - Run full experiments with:")
    print("     python examples/demo_comparison.py --full")
    print("   - Generate plots with:")
    print("     python examples/generate_plots.py")
    print("   - See README.md for full documentation")


if __name__ == "__main__":
    main()
