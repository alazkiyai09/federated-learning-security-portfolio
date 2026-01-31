# Non-IID Data Partitioner for Federated Learning

A comprehensive toolkit for simulating realistic heterogeneous data distributions in federated learning experiments. Essential for evaluating FL algorithms under non-IID scenarios that mirror real-world deployments.

## Why Non-IID Data Matters in FL

In federated learning, **data heterogeneity** is the primary challenge that differentiates FL from centralized learning. Real FL systems face multiple types of non-IID data:

1. **Label distribution skew**: Different clients have different class proportions
   - *Example*: Regional banks see different fraud patterns

2. **Quantity skew**: Clients have vastly different amounts of data
   - *Example*: Active vs. inactive users

3. **Feature distribution skew**: Clients' data comes from different feature distributions
   - *Example*: Demographic or geographic differences

4. **Concept drift**: The relationship between features and labels varies across clients
   - *Example*: Fraud patterns evolve differently by region

This toolkit provides **validated, reproducible** methods to simulate all these scenarios.

## Installation

```bash
cd non_iid_partitioner
pip install -r requirements.txt
```

### Dependencies

- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0

## Quick Start

```python
from src.partitioner import NonIIDPartitioner
import numpy as np

# Prepare your data
X = np.random.randn(10000, 50)  # Features
y = np.random.randint(0, 5, 10000)  # Labels (5 classes)

# Initialize partitioner
partitioner = NonIIDPartitioner(n_clients=20, random_state=42)

# Partition with label skew (Dirichlet)
partitions = partitioner.partition_label_skew(X, y, alpha=0.5)

# Access client data
for client_id, (X_client, y_client) in partitions.items():
    print(f"Client {client_id}: {len(y_client)} samples")
    # Train local model on X_client, y_client
```

## Partition Strategies

### 1. IID Partition (Baseline)

Random uniform distribution across all clients. Use as the baseline to compare against non-IID methods.

```python
partitions = partitioner.partition_iid(X, y)
```

**Characteristics:**
- Each client: ~same number of samples
- Label distribution: similar across all clients
- Represents idealized scenario (rarely occurs in practice)

---

### 2. Label Skew (Dirichlet Distribution)

**Most important strategy** for FL research. Controls class distribution heterogeneity using the Dirichlet concentration parameter `alpha`.

```python
# Moderate heterogeneity
partitions = partitioner.partition_label_skew(X, y, alpha=1.0)

# Extreme non-IID (each client dominated by 1-2 classes)
partitions = partitioner.partition_label_skew(X, y, alpha=0.1)

# Near-IID
partitions = partitioner.partition_label_skew(X, y, alpha=100.0)
```

**Mathematical Foundation:**

For $K$ classes and $N$ clients, we sample proportions $\mathbf{p}_i \sim \text{Dir}(\alpha)$ for each client $i$:

$$
p_{i,k} \sim \frac{\theta_k}{\sum_{j=1}^K \theta_j}, \quad \theta_k \sim \text{Gamma}(\alpha, 1)
$$

**Effect of Alpha:**

| Alpha | Heterogeneity | Description |
|-------|--------------|-------------|
| 0.01 - 0.1 | Extreme | Each client has 1-2 dominant classes |
| 0.5 - 1.0 | High | Significant class imbalance across clients |
| 2.0 - 5.0 | Moderate | Noticeable but manageable skew |
| 10.0 - 100.0 | Low | Approaches IID distribution |

**Use Cases:**
- Fraud detection: Regional banks see different fraud types
- Medical imaging: Hospitals specialize in different conditions
- NLP: Users from different regions use different languages

---

### 3. Quantity Skew (Power Law Distribution)

Simulates scenarios where clients have vastly different amounts of data, following a heavy-tailed distribution.

```python
# Heavy skew (few clients have most data)
partitions = partitioner.partition_quantity_skew(X, y, exponent=1.2)

# Moderate skew
partitions = partitioner.partition_quantity_skew(X, y, exponent=2.0)

# Mild skew
partitions = partitioner.partition_quantity_skew(X, y, exponent=3.0)
```

**Mathematical Foundation:**

Sample counts follow Pareto distribution:
$$
P(x) \sim x^{-\gamma}
$$

where $\gamma$ is the exponent parameter.

**Effect of Exponent:**

| Exponent | Inequality | Description |
|----------|-----------|-------------|
| 1.1 - 1.3 | Extreme | Top 10% clients have 70%+ of data |
| 1.5 - 2.0 | High | Top 20% clients have 50%+ of data |
| 2.5 - 3.5 | Moderate | Noticeable size variation |
| 5.0+ | Low | Relatively uniform sizes |

**Use Cases:**
- Mobile devices: Active vs. inactive users
- Healthcare: Large hospitals vs. small clinics
- Finance: High-volume vs. low-volume merchants

---

### 4. Feature Skew (Clustering-Based)

Groups similar samples (in feature space) and assigns to clients. Simulates geographic or demographic segmentation.

```python
# Using K-means clustering
partitions = partitioner.partition_feature_skew(X, y, n_clusters=5)

# For large datasets, use MiniBatchKMeans
partitions = partitioner.partition_feature_skew(
    X, y, n_clusters=10, use_minibatch=True
)
```

**How It Works:**
1. Cluster data in feature space (K-means)
2. Assign clusters to clients
3. Clients receive data from different feature regions

**Use Cases:**
- Geographic distribution (e.g., banks in different regions)
- Demographic segmentation (e.g., age groups, income levels)
- Temporal shifts (e.g., seasonal variations)

---

### 5. Realistic Bank Simulation

Domain-specific strategy for fraud detection that combines geographic and demographic factors.

```python
import pandas as pd

# Your transaction DataFrame
df = pd.DataFrame({
    'amount': transaction_amounts,
    'time': transaction_times,
    'merchant': merchant_categories,
    'age': customer_ages,
    'region': region_labels,  # Optional, will infer if missing
    'label': fraud_labels
})

# Partition with realistic constraints
partitions = partitioner.partition_realistic_bank(
    df,
    region_col='region',
    label_col='label',
    balance_within_regions=True
)
```

**Features:**
- Uses existing region labels or infers from features
- Balances fraud classes within regions (optional)
- Simulates real-world bank deployment scenarios

**Use Cases:**
- Fraud detection research
- Financial FL experiments
- Geographic-based studies

---

## Visualization & Analysis

### Generate Partition Reports

```python
from src.visualization import create_partition_report

# Create comprehensive visualizations
figures = create_partition_report(
    partitions_as_indices,
    y,
    save_dir='output',
    prefix='label_skew_alpha05'
)

# This generates:
# - label_skew_alpha05_heatmap.png: Class distribution per client
# - label_skew_alpha05_quantity.png: Sample count per client
# - label_skew_alpha05_stacked.png: Stacked bar chart of class distributions
```

### Compute Heterogeneity Metrics

```python
from src.visualization import compute_heterogeneity_metrics

metrics = compute_heterogeneity_metrics(partitions_as_indices, y)

print(f"Mean label entropy: {metrics['mean_label_entropy']:.3f}")
print(f"Gini coefficient: {metrics['gini_coefficient']:.3f}")
print(f"Max/min client size ratio: {metrics['max_min_ratio']:.2f}")
```

**Metrics Explained:**

| Metric | Interpretation | Value Range |
|--------|---------------|-------------|
| `mean_label_entropy` | Average diversity of labels per client | 0 (single class) to log(K) (uniform) |
| `gini_coefficient` | Inequality in sample sizes | 0 (perfect equality) to 1 (maximal inequality) |
| `cv_samples` | Coefficient of variation of sizes | 0 (all equal) to higher values |
| `max_min_ratio` | Size ratio between largest and smallest client | 1 (equal) to ∞ (extreme) |

### Access Partition Statistics

```python
stats = partitioner.get_partition_statistics()

for client_id, client_stats in stats.items():
    print(f"Client {client_id}:")
    print(f"  Samples: {client_stats['n_samples']}")
    print(f"  Label distribution: {client_stats['label_distribution']}")
```

## Advanced Usage

### Combining Multiple Skews

```python
# First apply quantity skew to get unequal client sizes
partitions = partitioner.partition_quantity_skew(X, y, exponent=1.5)

# Then further partition each client's data with label skew
for client_id, (X_client, y_client) in partitions.items():
    if len(y_client) > 50:  # Only for clients with enough data
        # Further sub-partition or apply additional skew
        pass
```

### Working with Indices Only

```python
from src.strategies.label_skew import dirichlet_partition_indices

# Get just the indices (more memory efficient)
partition_indices = dirichlet_partition_indices(y, n_clients=10, alpha=0.5)

# Use indices to create custom data loaders
for client_id, indices in partition_indices.items():
    X_client = X[indices]
    y_client = y[indices]
    # Create DataLoader, etc.
```

### Custom Region Labels

```python
# Create synthetic regions based on feature clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
regions = kmeans.fit_predict(X)

# Use regions for realistic bank simulation
partitions = partitioner.partition_realistic_bank_arrays(
    X, y,
    region_labels=regions,
    balance_within_regions=True
)
```

## Running the Demo

```bash
cd non_iid_partitioner
python examples/demo.py
```

This demonstrates:
1. All partition strategies with different parameters
2. Visualizations of partition statistics
3. Comparison of heterogeneity metrics
4. Reproducibility verification

## Reproducibility

**Critical for research:** Set `random_state` for reproducible partitions.

```python
# Same random_state = identical partitions
partitioner1 = NonIIDPartitioner(n_clients=10, random_state=42)
partitioner2 = NonIIDPartitioner(n_clients=10, random_state=42)

partitions1 = partitioner1.partition_label_skew(X, y, alpha=0.5)
partitions2 = partitioner2.partition_label_skew(X, y, alpha=0.5)

# partitions1 and partitions2 are identical
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_strategies.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

The test suite validates:
- ✅ Correctness of each partition strategy
- ✅ Reproducibility with random_state
- ✅ Coverage of all samples (no data loss/gain)
- ✅ Parameter validation (alpha, exponent, etc.)
- ✅ Integration with main partitioner class

## Integration with FL Frameworks

### PyTorch Example

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Get partitions
partitioner = NonIIDPartitioner(n_clients=10, random_state=42)
partitions = partitioner.partition_label_skew(X, y, alpha=0.5)

# Create DataLoaders for each client
client_loaders = {}
for client_id, (X_client, y_client) in partitions.items():
    dataset = TensorDataset(
        torch.FloatTensor(X_client),
        torch.LongTensor(y_client)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    client_loaders[client_id] = loader

# Use in FL training
for client_id in client_loaders:
    local_model = train_local_model(global_model, client_loaders[client_id])
```

### TensorFlow Example

```python
import tensorflow as tf

# Get partitions
partitions = partitioner.partition_label_skew(X, y, alpha=0.5)

# Create tf.data.Datasets
client_datasets = {}
for client_id, (X_client, y_client) in partitions.items():
    dataset = tf.data.Dataset.from_tensor_slices((X_client, y_client))
    dataset = dataset.shuffle(1000).batch(32)
    client_datasets[client_id] = dataset
```

## Theory & References

### Dirichlet Distribution for Label Skew

The Dirichlet distribution is used because:

1. **Flexible**: Can represent uniform (α→∞) to sparse (α→0) distributions
2. **Normalized**: Automatically sums to 1 for proportions
3. **Theoretically grounded**: Well-studied in Bayesian statistics

**Key Papers:**
- [McMahan et al., 2017] Communication-Efficient Learning of Deep Networks from Decentralized Data
- [Hsieh et al., 2020] Can You Really Backpropagate Through Federated Averaging?

### Power Law for Quantity Skew

Real-world data naturally follows power law distributions:
- User activity (social networks, mobile apps)
- File sizes (internet traffic)
- Wealth distribution (economic data)

**Key Insight:** Most FL systems have a "long tail" of inactive users.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{non_iid_partitioner,
  title={Non-IID Data Partitioner for Federated Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/non-iid-partitioner}
}
```

## License

MIT License - feel free to use in your research and projects.

## Contributing

Contributions welcome! Areas for improvement:
- Additional partition strategies (e.g., concept drift)
- Support for time-series data
- Integration with more FL frameworks
- Extended visualization options

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].

---

**Happy Federated Learning! 🚀**
