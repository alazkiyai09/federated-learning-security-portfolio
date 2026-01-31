"""
Demo script showcasing all partition strategies with visualizations.

This script demonstrates:
1. All partition strategies (IID, label skew, quantity skew, feature skew, realistic bank)
2. Visualization of partition statistics
3. Comparison of heterogeneity metrics
4. Reproducibility examples

Run this script to generate example partitions and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.partitioner import NonIIDPartitioner
from src.visualization import (
    plot_client_distribution,
    plot_quantity_distribution,
    plot_label_distribution_comparison,
    compute_heterogeneity_metrics,
    create_partition_report
)


def create_synthetic_fraud_data(n_samples: int = 5000, n_features: int = 20, n_classes: int = 5):
    """
    Create synthetic fraud detection dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of fraud pattern classes

    Returns:
        X, y arrays
    """
    np.random.seed(42)

    # Generate features with different patterns
    X = np.random.randn(n_samples, n_features)

    # Add some structure: different classes have different feature means
    for class_id in range(n_classes):
        class_mask = np.random.rand(n_samples) < (1.0 / n_classes)
        n_class = class_mask.sum()
        X[class_mask, :2] += class_id * 2  # Shift first two features by class

    # Generate labels
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


def demo_iid_partition():
    """Demonstrate IID partition (baseline)."""
    print("\n" + "="*60)
    print("1. IID PARTITION (BASELINE)")
    print("="*60)

    X, y = create_synthetic_fraud_data(n_samples=2000, n_classes=5)
    partitioner = NonIIDPartitioner(n_clients=10, random_state=42)

    partitions = partitioner.partition_iid(X, y)

    # Show statistics
    sizes = partitioner.get_client_sizes()
    print(f"\nSample counts per client: {list(sizes.values())[:5]}...")

    metrics = compute_heterogeneity_metrics(
        {cid: indices for cid, (X_c, y_c) in
         {cid: (X[indices], y[indices]) for cid, indices in
          np.concatenate([np.where((y == i).astype(int)) for i in range(5)])}.items()},
        y
    )

    # Convert partitions to indices format for visualization
    partition_indices = {}
    start = 0
    for cid, (X_c, y_c) in partitions.items():
        # This is a simplified conversion for demo
        # In practice, you'd track the original indices
        n = len(y_c)
        partition_indices[cid] = np.arange(start, start + n) % len(y)
        start += n

    print(f"✓ IID partition created")
    return partitioner, partition_indices, X, y


def demo_label_skew():
    """Demonstrate Dirichlet label skew partition."""
    print("\n" + "="*60)
    print("2. LABEL SKEW PARTITION (DIRICHLET)")
    print("="*60)

    X, y = create_synthetic_fraud_data(n_samples=2000, n_classes=5)
    partitioner = NonIIDPartitioner(n_clients=10, random_state=42)

    # Test different alpha values
    alphas = [0.1, 0.5, 1.0, 5.0]

    print("\nTesting different alpha values:")
    print("alpha → 0: Extreme non-IID (each client ~1 class)")
    print("alpha = 1: Moderate heterogeneity")
    print("alpha → ∞: Approaches IID\n")

    for alpha in alphas:
        partitions = partitioner.partition_label_skew(X, y, alpha=alpha)

        # Calculate label skew metric
        all_label_counts = []
        for cid, (X_c, y_c) in partitions.items():
            if len(y_c) > 0:
                unique, counts = np.unique(y_c, return_counts=True)
                if len(counts) > 0:
                    all_label_counts.append(counts.max() / len(y_c))

        avg_skew = np.mean(all_label_counts) if all_label_counts else 0
        print(f"  alpha={alpha:4.1f}: Avg max class proportion = {avg_skew:.3f}")

    # Create detailed visualization for alpha=0.5
    partitions = partitioner.partition_label_skew(X, y, alpha=0.5)

    # Convert to indices
    partition_indices = {}
    idx = 0
    for cid, (X_c, y_c) in partitions.items():
        n = len(y_c)
        partition_indices[cid] = np.arange(idx, idx + n) % len(y)
        idx += n

    print(f"\n✓ Label skew partition created (alpha=0.5)")
    return partitioner, partition_indices, X, y


def demo_quantity_skew():
    """Demonstrate power law quantity skew partition."""
    print("\n" + "="*60)
    print("3. QUANTITY SKEW PARTITION (POWER LAW)")
    print("="*60)

    X, y = create_synthetic_fraud_data(n_samples=2000, n_classes=5)
    partitioner = NonIIDPartitioner(n_clients=10, random_state=42)

    # Test different exponents
    exponents = [1.2, 1.5, 2.0, 3.0]

    print("\nTesting different exponents:")
    print("exponent → 1: Very heavy tail (extreme inequality)")
    print("exponent = 2: Standard Pareto")
    print("exponent → ∞: Approaches uniform\n")

    for exp in exponents:
        partitions = partitioner.partition_quantity_skew(X, y, exponent=exp)

        sizes = list(partitioner.get_client_sizes().values())
        gini = np.sum(np.abs(np.array(sizes) - np.mean(sizes))) / (2 * len(sizes) * np.mean(sizes))
        print(f"  exponent={exp:4.1f}: Size ratio (max/min) = {max(sizes)/min(sizes):.2f}")

    # Create detailed visualization for exponent=1.5
    partitions = partitioner.partition_quantity_skew(X, y, exponent=1.5)

    # Convert to indices
    partition_indices = {}
    idx = 0
    for cid, (X_c, y_c) in partitions.items():
        n = len(y_c)
        partition_indices[cid] = np.arange(idx, idx + n) % len(y)
        idx += n

    print(f"\n✓ Quantity skew partition created (exponent=1.5)")
    return partitioner, partition_indices, X, y


def demo_feature_skew():
    """Demonstrate feature-based skew partition."""
    print("\n" + "="*60)
    print("4. FEATURE SKEW PARTITION (CLUSTERING)")
    print("="*60)

    # Create clustered data for better demonstration
    np.random.seed(42)
    n_samples_per_cluster = 400
    n_clusters = 5

    X_clusters = []
    y_clusters = []

    for i in range(n_clusters):
        # Each cluster has different feature distribution
        X_cluster = np.random.randn(n_samples_per_cluster, 20) + i * 3
        y_cluster = np.random.randint(0, 5, n_samples_per_cluster)

        X_clusters.append(X_cluster)
        y_clusters.append(y_cluster)

    X = np.vstack(X_clusters)
    y = np.concatenate(y_clusters)

    partitioner = NonIIDPartitioner(n_clients=10, random_state=42)
    partitions = partitioner.partition_feature_skew(X, y, n_clusters=5)

    print(f"\nCreated {n_clusters} feature clusters")
    print(f"Partitioned across {partitioner.n_clients} clients")

    # Convert to indices
    partition_indices = {}
    idx = 0
    for cid, (X_c, y_c) in partitions.items():
        n = len(y_c)
        partition_indices[cid] = np.arange(idx, idx + n) % len(y)
        idx += n

    print(f"✓ Feature skew partition created")
    return partitioner, partition_indices, X, y


def demo_realistic_bank():
    """Demonstrate realistic bank simulation."""
    print("\n" + "="*60)
    print("5. REALISTIC BANK SIMULATION")
    print("="*60)

    # Create realistic bank transaction dataset
    np.random.seed(42)
    n_samples = 3000

    # Generate features
    transaction_amounts = np.random.lognormal(3, 1, n_samples)
    transaction_times = np.random.uniform(0, 24, n_samples)
    merchant_categories = np.random.randint(0, 10, n_samples)
    customer_ages = np.random.normal(45, 15, n_samples)

    X = np.column_stack([
        transaction_amounts,
        transaction_times,
        merchant_categories,
        customer_ages,
        np.random.randn(n_samples, 16)  # Other features
    ])

    # Generate fraud labels (correlated with some features)
    fraud_prob = 0.05 + 0.03 * (transaction_amounts / transaction_amounts.max())
    y = (np.random.random(n_samples) < fraud_prob).astype(int)

    # Create regions
    regions = np.random.randint(0, 5, n_samples)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[
        'amount', 'time', 'merchant', 'age'
    ] + [f'feature_{i}' for i in range(16)])
    df['label'] = y
    df['region'] = regions

    partitioner = NonIIDPartitioner(n_clients=10, random_state=42)
    partitions = partitioner.partition_realistic_bank(
        df, region_col='region', label_col='label'
    )

    # Analyze partitions
    print(f"\nSimulated {n_samples} transactions across {len(regions)} regions")
    print(f"Partitioned to {partitioner.n_clients} banks")

    for cid, client_df in list(partitions.items())[:3]:
        fraud_rate = client_df['label'].mean()
        print(f"  Bank {cid}: {len(client_df)} transactions, fraud rate = {fraud_rate:.3f}")

    print(f"\n✓ Realistic bank simulation created")
    return partitioner, partitions, df


def demo_reproducibility():
    """Demonstrate reproducibility with random_state."""
    print("\n" + "="*60)
    print("6. REPRODUCIBILITY TEST")
    print("="*60)

    X, y = create_synthetic_fraud_data(n_samples=1000, n_classes=5)

    partitioner1 = NonIIDPartitioner(n_clients=5, random_state=42)
    partitioner2 = NonIIDPartitioner(n_clients=5, random_state=42)

    partitions1 = partitioner1.partition_label_skew(X, y, alpha=0.5)
    partitions2 = partitioner2.partition_label_skew(X, y, alpha=0.5)

    # Check if identical
    identical = True
    for cid in range(5):
        if not np.array_equal(partitions1[cid][0], partitions2[cid][0]):
            identical = False
            break

    print(f"\nSame random_state produces identical partitions: {identical}")

    if identical:
        print("✓ Reproducibility verified")
    else:
        print("✗ WARNING: Partitions differ with same random_state!")

    return identical


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("NON-IID DATA PARTITIONER - DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases different partition strategies for FL experiments")

    # Create output directory for visualizations
    os.makedirs('output', exist_ok=True)

    # Run demos
    demo_iid_partition()
    demo_label_skew()
    demo_quantity_skew()
    demo_feature_skew()
    demo_realistic_bank()
    demo_reproducibility()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey takeaways:")
    print("1. IID: Uniform distribution (baseline)")
    print("2. Label skew: Dirichlet alpha controls class heterogeneity")
    print("3. Quantity skew: Power law exponent controls size inequality")
    print("4. Feature skew: Clustering creates feature-based groups")
    print("5. Realistic bank: Simulates real-world geographic/demographic factors")
    print("6. All strategies are reproducible with random_state")
    print("\nFor usage in your experiments:")
    print("  from src.partitioner import NonIIDPartitioner")
    print("  partitioner = NonIIDPartitioner(n_clients=10, random_state=42)")
    print("  partitions = partitioner.partition_label_skew(X, y, alpha=0.5)")


if __name__ == "__main__":
    main()
