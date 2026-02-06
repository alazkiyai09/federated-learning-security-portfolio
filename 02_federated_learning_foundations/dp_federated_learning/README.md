# DP Federated Learning

Differentially Private Federated Learning implementation using DP-SGD with Renyi Differential Privacy (RDP) accounting.

## Overview

This project implements federated learning with formal differential privacy guarantees using:
- **DP-SGD**: Differentially Private Stochastic Gradient Descent
- **RDP Accounting**: Renyi Differential Privacy for tight privacy bounds
- **Per-Client Privacy**: Each client's local training is privacy-protected
- **Privacy Budget Tracking**: Monitor and enforce privacy spending

## Mathematical Background

### Differential Privacy
A mechanism M satisfies (ε, δ)-DP if for any adjacent datasets D, D' and any output S:
```
P[M(D) ∈ S] ≤ e^ε * P[M(D'] ∈ S] + δ
```

### DP-SGD Algorithm
1. **Gradient Clipping**: Limit per-sample gradient L2 norm
2. **Noise Addition**: Add Gaussian noise to summed gradients
3. **Privacy Accounting**: Track cumulative privacy loss

### Renyi DP
For α-order Renyi divergence, RDP provides tight composition bounds:
```
ε_α = min_{α > 1} [ ε_α + log(1/δ) / (α - 1) ]
```

## Installation

```bash
cd 02_federated_learning_foundations/dp_federated_learning
pip install -r requirements.txt
```

## Project Structure

```
dp_federated_learning/
├── src/
│   ├── dp_mechanisms/
│   │   ├── privacy_accountant.py  # RDP accounting
│   │   ├── gradient_clipper.py    # Gradient clipping
│   │   └── noise_addition.py      # Gaussian noise mechanisms
│   ├── dp_strategies/
│   │   └── client_dp.py           # Client-level DP strategies
│   ├── utils/
│   │   └── privacy_calibration.py # Privacy parameter tuning
│   └── experiments/
│       └── dp_experiments.py      # Experiment runners
├── tests/
│   └── test_dp_sgd.py             # DP-SGD tests
└── config/
    └── dp_config.yaml             # Privacy parameters
```

## Usage

### Basic DP-SGD Training

```python
from src.models.dp_sgd_custom import DPSGDOptimizer

# Create DP-SGD optimizer
optimizer = DPSGDOptimizer(
    model=model,
    noise_multiplier=1.5,
    clipping_bound=1.0,
    batch_size=32,
    lr=0.01,
    dataset_size=1000
)

# Train with privacy guarantee
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        loss = model(batch_x).loss(batch_y)
        loss.backward()  # Gradients are automatically clipped
        optimizer.step()  # Noise is automatically added

# Check privacy spent
print(optimizer.privacy_spent())
```

### Privacy Accounting

```python
from src.dp_mechanisms.privacy_accountant import RDPAccountant, PrivacyBudget

# Set privacy budget
budget = PrivacyBudget(epsilon=3.0, delta=1e-5)

# Create accountant
accountant = RDPAccountant(
    noise_multiplier=1.5,
    sampling_rate=0.01,
    target_delta=1e-5
)

# Track privacy per step
for step in range(num_steps):
    accountant.step()

# Check remaining budget
epsilon_spent = accountant.get_epsilon(num_steps)
print(f"ε spent: {epsilon_spent:.2f} / {budget.epsilon}")
```

## Configuration

Privacy parameters in `config/dp_config.yaml`:

```yaml
privacy:
  target_epsilon: 3.0        # Total privacy budget
  target_delta: 1e-5         # Failure probability
  noise_multiplier: 1.5      # Per-round noise scale
  max_grad_norm: 1.0         # Gradient clipping bound

training:
  num_rounds: 100
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
```

## Privacy Parameters Guide

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `noise_multiplier` | More noise = better privacy | Reduces model accuracy |
| `max_grad_norm` | Lower clipping = better privacy | May affect convergence |
| `batch_size` | Larger batch = tighter bounds | Increases computation |
| `target_epsilon` | Lower ε = stronger privacy | May limit training rounds |

### Recommended Settings

| Privacy Level | ε | δ | noise_multiplier |
|--------------|---|---|------------------|
| Strong | 1.0 | 1e-5 | 2.0 |
| Moderate | 3.0 | 1e-5 | 1.5 |
| Weak | 8.0 | 1e-5 | 1.0 |

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_dp_sgd.py::TestDPSGDOptimizerInit
```

## Theoretical Background

This implementation is based on:
- Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
- Mironov "Renyi Differential Privacy" (IEEE S&P 2017)

### Privacy Composition
For T rounds of DP-SGD with noise multiplier σ and sampling rate q:
```
ε_T ≈ sqrt(T * log(1/δ)) * q * σ
```

## Citation

If you use this code, please cite:
```bibtex
@article{dp_fl_fraud_detection,
  title={Differentially Private Federated Learning for Fraud Detection},
  author={...},
  journal={...},
  year={2025}
}
```

## License

See LICENSE file for details.
