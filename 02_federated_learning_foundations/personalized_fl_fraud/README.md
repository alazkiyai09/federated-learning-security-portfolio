# Personalized Federated Learning for Fraud Detection

Adapting global fraud detection models to individual bank patterns through personalized federated learning.

## Overview

Standard federated learning (FedAvg) produces a "one-size-fits-all" global model that underperforms on non-IID data where different banks (clients) have varying fraud patterns. This project implements and compares **four personalization techniques** to adapt global models to local client data while preserving knowledge sharing.

## Motivation

**Problem**: In real-world fraud detection, different banks serve different customer demographics and geographic regions, leading to highly non-IID data distributions. A global model trained via FedAvg may:
- Perform poorly on clients with underrepresented fraud patterns
- Fail to capture local fraud characteristics
- Result in unfair performance variance across clients

**Solution**: Personalized FL techniques that:
1. Learn shared representations across all clients
2. Adapt to local data distributions
3. Balance personalization vs. generalization

## Methods Implemented

| Method | Personalization Approach | Pros | Cons | Best For |
|--------|-------------------------|------|------|----------|
| **FedAvg** | None (baseline) | Simple, reproducible | No personalization | Homogeneous data, baseline comparison |
| **Local Fine-Tuning** | Post-hoc local training | Simple, no server changes | Risk of overfitting, catastrophic forgetting | Clients with sufficient local data |
| **FedPer** | Personalized classification layer | Clear separation, low communication | Requires model split | Architectures with clear feature/classifier split |
| **Ditto** | Local + global models with regularization | Robust, explicit local objective | 2x memory, more complex | Highly non-IID data, fairness concerns |
| **Per-FedAvg** | MAML-inspired meta-learning | Fast adaptation, strong theory | Complex, computationally expensive | Rapidly changing fraud patterns |

## Project Structure

```
personalized_fl_fraud/
├── config/
│   ├── experiments.yaml       # Experiment configurations
│   └── methods.yaml           # Method-specific hyperparameters
├── data/
│   ├── raw/                   # Raw data (use synthetic)
│   └── processed/             # Partitioned data cache
├── src/
│   ├── models/                # Model architectures
│   │   ├── base.py            # FraudDetectionModel with split
│   │   └── utils.py           # Model utilities
│   ├── methods/               # Personalization methods
│   │   ├── base.py            # Abstract base class
│   │   ├── local_finetuning.py
│   │   ├── fedper.py
│   │   ├── ditto.py
│   │   └── per_fedavg.py
│   ├── clients/               # FL client implementations
│   │   ├── personalized_client.py
│   │   └── wrappers.py        # Client factory functions
│   ├── servers/               # FL server strategies
│   │   └── personalized_server.py
│   ├── experiments/           # Experiment orchestration
│   │   ├── runner.py          # Main experiment runner
│   │   └── comparison.py      # Method comparison
│   ├── metrics/               # Metrics and visualization
│   │   ├── personalized_metrics.py
│   │   └── visualization.py
│   └── utils/                 # Utilities
│       ├── partitioning.py    # Data partitioning (uses Day 9)
│       ├── reproducibility.py # Checkpointing, seeds
│       ├── compute_tracking.py # FLOPs, communication tracking
│       └── metrics.py         # Fraud detection metrics
├── tests/
│   └── test_methods.py        # Unit tests
├── examples/
│   ├── demo_single_method.py
│   └── demo_comparison.py
├── results/
│   ├── checkpoints/           # Model checkpoints
│   ├── metrics/               # Per-client metrics (JSON)
│   └── figures/               # Plots
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Navigate to project directory
cd /home/ubuntu/30Days_Project/personalized_fl_fraud

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Run Demo

```bash
# Run single method demo
python examples/demo_single_method.py

# Run method comparison demo
python examples/demo_comparison.py
```

### 2. Run Unit Tests

```bash
# Run all tests
pytest tests/test_methods.py -v

# Run specific test class
pytest tests/test_methods.py::TestLocalFineTuning -v
```

### 3. Run Full Experiment

```python
from src.utils import set_random_seed, DataPartitioner, create_synthetic_fraud_data
from src.experiments.runner import ExperimentRunner
from src.methods.local_finetuning import LocalFineTuning
from src.methods.fedper import FedPer
from src.methods.ditto import Ditto
from src.methods.per_fedavg import PerFedAvg
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config/experiments.yaml")

# Set random seed
set_random_seed(42)

# Create data
X, y = create_synthetic_fraud_data(n_samples=10000, n_features=20, fraud_ratio=0.05)

# Partition with multiple alpha levels
partitioner = DataPartitioner(n_clients=10, test_size=0.2, val_size=0.1)
partitions = partitioner.create_partitions_at_multiple_alphas(
    X, y, alpha_values=[0.1, 0.5, 1.0, 10.0]
)

# Initialize methods
methods = [
    LocalFineTuning(name="Local Fine-Tuning", config=config),
    FedPer(name="FedPer", config=config),
    Ditto(name="Ditto", config=config),
    PerFedAvg(name="Per-FedAvg", config=config),
]

# Run comparison
runner = ExperimentRunner(config, methods, random_state=42)
results = runner.run_comparison(alpha_values=[0.1, 0.5, 1.0, 10.0], n_rounds=100)
```

## Configuration

### experiments.yaml

Controls experiment parameters:
- `data.n_clients`: Number of banks/clients
- `partitioning.alpha_values`: Non-IID levels to test
- `federated.n_rounds`: Communication rounds
- `model.hidden_dims`: MLP architecture

### methods.yaml

Controls method-specific hyperparameters:
- `local_finetuning.finetuning_epochs`: Local fine-tuning rounds
- `fedper.personal_layers`: Which layers to personalize
- `ditto.lambda_regularization`: Local model regularization strength
- `per_fedavg.beta`: Moreau envelope strength

## Key Features

### 1. Non-IID Data Partitioning

Uses Day 9's `NonIIDPartitioner` with Dirichlet-based label skew:
- α → 0: Extreme non-IID (1-2 dominant classes per client)
- α = 1: Moderate heterogeneity
- α → ∞: Approaches IID

### 2. Fair Comparison

All methods evaluated with:
- **Same compute budget**: Tracked FLOPs and communication cost
- **Same random seeds**: Reproducible results
- **Per-client metrics**: Not just averages, but full distributions
- **Multiple α levels**: Test across non-IID spectrum

### 3. Comprehensive Metrics

- **Performance**: AUC, PR-AUC, F1, Recall@1%FPR
- **Fairness**: Performance variance, Gini coefficient, worst-client
- **Personalization Benefit**: Δ(personalized - global)
- **Efficiency**: FLOPs, communication bytes, training time

### 4. Visualization

- Violin plots of per-client performance
- Personalization vs generalization trade-offs
- Alpha sensitivity curves
- Training curves across rounds

## Expected Results

Based on literature and non-IID fraud detection characteristics:

| α (Non-IID Level) | Best Method | Reason |
|-------------------|-------------|--------|
| 0.1 (High non-IID) | Ditto, FedPer | Explicit local modeling, personalized layers |
| 0.5 (Medium) | FedPer, Per-FedAvg | Balance of sharing and adaptation |
| 1.0 (Low) | All methods converge | More data sharing possible |
| 10.0 (Near IID) | FedAvg sufficient | Personalization provides little benefit |

**Key Findings to Expect**:
1. Personalization benefit **inversely correlated** with α
2. FedPer provides **best communication efficiency**
3. Ditto shows **best worst-client performance** (fairness)
4. Per-FedAvg **fastest to adapt** but highest compute cost

## Reproducibility

All experiments use:
- Fixed random seeds (`random_state=42`)
- Deterministic PyTorch operations
- Checkpointing for all rounds
- Full configuration logging

To reproduce results:
```bash
python examples/run_full_study.py \
    --config config/experiments.yaml \
    --methods fedavg,local_finetuning,fedper,ditto,per_fedavg \
    --alpha 0.1,0.5,1.0,10.0 \
    --rounds 100 \
    --seed 42
```

## Citation

If you use this code, please cite:

```bibtex
@software{personalized_fl_fraud,
  title={Personalized Federated Learning for Fraud Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/personalized-fl-fraud}
}
```

## References

1. **FedPer**: "Federated Learning with Personalization Layers" (Arivazhagan et al., ICLR 2020)
2. **Ditto**: "Ditto: Fair and Robust Federated Learning" (Li et al., ICLR 2022)
3. **Per-FedAvg**: "Personalized Federated Learning with Moreau Envelopes" (Dinh et al., NeurIPS 2020)
4. **MAML**: "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., ICML 2017)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Areas for improvement:
- Additional personalization methods (e.g., FedHybrid, pFedMe)
- Real-world fraud datasets
- Distributed training across multiple machines
- Adaptive hyperparameter tuning

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Status**: ✅ Implementation complete, ready for experiments

**Next Steps**:
1. Run full experiment suite across all α values
2. Generate comparison plots
3. Write up results for PhD application portfolio
