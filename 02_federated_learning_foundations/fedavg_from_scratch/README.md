# FedAvg from Scratch

**Pure PyTorch Implementation of Federated Averaging**

A clean, well-documented implementation of the FedAvg algorithm from McMahan et al., 2017. Built for deep understanding of federated learning mechanics and research extension in trustworthy ML.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Testing](#testing)
- [Reference](#reference)

## 🎯 Overview

This project implements **Federated Averaging (FedAvg)** from scratch using only PyTorch - no external federated learning frameworks. The implementation is designed for:

- **Educational purposes**: Deep understanding of FL mechanics
- **Research extension**: Clean codebase for PhD research in trustworthy ML
- **Fraud detection**: Applying FL to privacy-sensitive financial data

### Key Features

- ✅ Pure PyTorch implementation (no Flower, PySyft, or other FL frameworks)
- ✅ Weight serialization from scratch
- ✅ Heterogeneous client data handling
- ✅ IID and non-IID data partitioning
- ✅ Weighted averaging by sample count
- ✅ Comprehensive unit tests
- ✅ MNIST sanity check
- ✅ Fraud detection application

## 🧮 Algorithm

### Federated Averaging (FedAvg)

FedAvg enables collaborative training across decentralized clients while preserving data privacy. The algorithm proceeds in rounds:

**Notation:**
- $K$: Total number of clients
- $C$: Fraction of clients selected per round
- $E$: Number of local training epochs
- $B$: Local batch size
- $\eta$: Learning rate
- $w_t$: Global model weights at round $t$
- $n_k$: Number of training samples on client $k$
- $n = \sum_{k} n_k$: Total samples

**Algorithm:**

```
Initialize global model weights w_0

for each round t = 1, 2, ..., T do
    # Step 1: Client Selection
    S_t ← random subset of clients (size = max(C·K, 1))

    # Step 2: Local Training
    for each client k ∈ S_t in parallel do
        w_t^k ← local SGD(w_t, E, B, η) on client k's data
    end for

    # Step 3: Weighted Aggregation
    w_{t+1} ← Σ_{k∈S_t} (n_k / n) · w_t^k

end for
```

**Key Insight:** Clients train locally for multiple epochs, reducing communication rounds while converging to a good global model.

### Implementation Details

Our implementation follows the algorithm exactly:

1. **Client Selection** (`src/server.py:select_clients`):
   - Randomly samples fraction $C$ of clients each round
   - Ensures at least 1 client is selected

2. **Local Training** (`src/client.py:local_train`):
   - Loads global weights
   - Runs $E$ epochs of local SGD
   - Returns updated weights and sample count

3. **Weighted Aggregation** (`src/server.py:aggregate_weights`):
   - Computes weighted average: $w_{new} = \sum_k \frac{n_k}{n_{total}} w_k$
   - Ensures exact implementation of FedAvg formula

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Setup

```bash
# Clone repository
git clone <repository-url>
cd fedavg_from_scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
fedavg_from_scratch/
├── README.md                 # This file
├── requirements.txt          # Dependencies
│
├── src/                      # Core implementation
│   ├── __init__.py
│   ├── client.py            # FederatedClient class
│   ├── server.py            # FederatedServer class
│   ├── models.py            # Neural network architectures
│   ├── data.py              # Data loading and partitioning
│   ├── utils.py             # Utilities (serialization, seeds)
│   └── metrics.py           # Convergence tracking and visualization
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_serialization.py    # Weight save/load tests
│   ├── test_client.py           # Local training tests
│   └── test_aggregation.py      # FedAvg weighted averaging tests
│
├── experiments/              # Run experiments
│   ├── mnist_sanity_check.py    # MNIST baseline
│   └── fraud_detection.py       # Fraud detection application
│
└── results/                  # Experimental results
    ├── mnist/                # MNIST outputs
    └── fraud/                # Fraud detection outputs
```

## 💻 Usage

### Quick Start: MNIST Sanity Check

```bash
# Run MNIST experiment (verifies correct implementation)
python experiments/mnist_sanity_check.py

# Results will be saved to results/mnist/
# - Convergence plots
# - Metrics JSON
# - Trained model
```

### Fraud Detection Application

```bash
# Download Credit Card Fraud dataset from:
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# Place data.csv in ./data/ directory

# Run fraud detection experiment
python experiments/fraud_detection.py

# Results will be saved to results/fraud/
```

### Python API

```python
from src.models import SimpleCNN
from src.data import load_mnist, partition_data, create_test_loader
from src.client import FederatedClient
from src.server import FederatedServer
from src.metrics import ConvergenceTracker
from src.utils import set_seed, get_device

# Setup
set_seed(42)
device = get_device()

# Load data
train_dataset, test_dataset = load_mnist()
client_loaders, sample_counts = partition_data(
    train_dataset, num_clients=10, distribution='iid'
)
test_loader = create_test_loader(test_dataset)

# Create global model
global_model = SimpleCNN(num_classes=10)

# Create clients
clients = []
for i, loader in enumerate(client_loaders):
    client = FederatedClient(
        client_id=i,
        model=SimpleCNN(num_classes=10),
        train_loader=loader,
        config={'local_epochs': 5, 'learning_rate': 0.01}
    )
    clients.append(client)

# Create server
server = FederatedServer(
    model=global_model,
    config={'num_rounds': 20, 'client_fraction': 0.5}
)

# Training
tracker = ConvergenceTracker()
for round_num in range(20):
    selected_clients = server.select_clients(clients, fraction=0.5)
    metrics = server.federated_round(selected_clients, round_num, test_loader)
    tracker.update(round_num, metrics)

# Save results
tracker.plot_convergence('results/convergence.png')
```

## 🧪 Experiments

### 1. MNIST Sanity Check

**Purpose:** Verify correct implementation by achieving >95% test accuracy on MNIST.

**Configuration:**
- 10 clients
- 20 communication rounds
- 50% client participation per round
- 5 local epochs per client
- SGD with momentum (lr=0.01)

**Results (IID):**
- Final Test Accuracy: ~98%
- Convergence in 15-20 rounds

**Results (Non-IID):**
- Final Test Accuracy: ~96%
- Slower convergence due to data heterogeneity

### 2. Fraud Detection

**Purpose:** Apply FedAvg to real-world imbalanced classification.

**Configuration:**
- 5 clients (simulating banks)
- 30 communication rounds
- 80% client participation
- 10 local epochs (Adam optimizer)
- MLP with batch normalization

**Metrics:**
- AUC-PR (primary metric for imbalanced data)
- F1 score
- Precision, Recall

## 📊 Results

### MNIST Convergence

![MNIST Convergence](results/mnist/mnist_convergence_iid.png)

### Fraud Detection Performance

![Fraud Detection Convergence](results/fraud/fraud_convergence_non-iid.png)

## 🧪 Testing

Run all unit tests:

```bash
# Test weight serialization
pytest tests/test_serialization.py -v

# Test client local training
pytest tests/test_client.py -v

# Test FedAvg aggregation (weighted averaging math)
pytest tests/test_aggregation.py -v

# Run all tests
pytest tests/ -v
```

### Test Coverage

- **test_serialization.py**: Verifies weight preservation during save/load
- **test_client.py**: Tests local training logic and weight updates
- **test_aggregation.py**: Validates weighted averaging formula

## 📚 Reference

**Paper:**
> McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017).
> Communication-efficient learning of deep networks from decentralized data.
> *Artificial Intelligence and Statistics*, 1273-1282.

**Key Contributions:**
1. FedAvg algorithm for communication-efficient FL
2. Theoretical analysis of convergence
3. Empirical results on MNIST and Shakespeare dataset

## 🔬 Research Extensions

This implementation provides a foundation for research in:

1. **Privacy attacks**: Membership inference on FedAvg
2. **Fairness**: Ensuring equitable performance across clients
3. **Robustness**: Byzantine-resistant aggregation
4. **Efficiency**: Communication compression techniques
5. **Personalization**: Local adaptation strategies

## 📝 License

This project is for research and educational purposes.

## 👤 Author

Built for PhD applications in trustworthy machine learning.

---

**Note:** This implementation prioritizes clarity and correctness over optimization. For production use, consider frameworks like Flower (flwr) that provide additional features and optimizations.
