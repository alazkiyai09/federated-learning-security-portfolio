# Vertical Federated Learning for Fraud Detection

A PyTorch implementation of Vertical Federated Learning (VFL) for fraud detection using split learning architecture. In VFL, different parties hold **different features** for the same users, enabling cross-institution collaboration without sharing raw data.

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Party A       │         │   Party B       │
│  (Bank A)       │         │  (Bank B)       │
│                 │         │                 │
│  Transaction    │         │   Credit        │
│  Features (7)   │         │   Features (3)  │
│        ↓        │         │        ↓        │
│  ┌──────────┐   │         │   ┌──────────┐ │
│  │  Bottom  │   │         │   │  Bottom  │ │
│  │  Model A │   │         │   │  Model B │ │
│  └────┬─────┘   │         │   └────┬─────┘ │
│       │          │         │        │       │
│       │ z_a (16) │         │  z_b (8)│      │
└───────┼──────────┘         └────────┼───────┘
        │                             │
        │    EMBEDDINGS ONLY          │
        └──────────┬──────────────────┘
                   ↓
          ┌─────────────────┐
          │     Server      │
          │                 │
          │  Concatenate    │
          │  [z_a, z_b]     │
          │        ↓        │
          │  ┌──────────┐   │
          │  │   Top    │   │
          │  │  Model   │   │
          │  └────┬─────┘   │
          │       │          │
          │       ↓          │
          │  Prediction     │
          └─────────────────┘
```

## 📋 Features

- **Split Learning Architecture**: Bottom models at parties, top model at server
- **Private Set Intersection (PSI)**: Secure ID alignment without revealing non-intersecting users
- **Secure Forward/Backward Pass**: Only embeddings and gradients transmitted
- **Gradient Leakage Analysis**: Quantify privacy risks from gradient transmission
- **Baseline Comparisons**: Single-party, centralized, and Horizontal FL
- **Unit Tests**: Verify gradient flow correctness

## 🔒 Privacy Guarantees

### What IS Shared
| Data | Party A → Server | Party B → Server | Server → Parties |
|------|------------------|------------------|------------------|
| **Forward** | Embeddings z_a | Embeddings z_b | Predictions |
| **Backward** | Gradients ∂L/∂z_a | Gradients ∂L/∂z_b | None |

### What is NOT Shared
- ✅ Raw transaction features (Party A) - **STAY LOCAL**
- ✅ Raw credit features (Party B) - **STAY LOCAL**
- ✅ Bottom model parameters - **KEPT SECRET**
- ✅ Raw parameter gradients (∂L/θ) - **NEVER TRANSMITTED**

## 📁 Directory Structure

```
vertical_fraud_detection/
├── config/                    # Configuration files
│   ├── model_config.yaml     # Model architectures
│   └── experiment_config.yaml # Experiment settings
├── data/                      # Data directory
│   ├── raw/                  # Raw data files
│   ├── processed/            # Aligned and split data
│   └── psi_intersection.json # PSI results
├── src/
│   ├── psi/                  # Private Set Intersection
│   ├── models/               # Neural network models
│   ├── training/             # Training protocols
│   ├── experiments/          # Experiment runners
│   ├── privacy/              # Privacy analysis
│   └── utils/                # Utilities
├── tests/                     # Unit tests
├── results/                   # Experiment results
└── run_experiments.py        # Main entry point
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Setup: Generate data and run PSI
python run_experiments.py --mode setup

# Run Vertical FL only
python run_experiments.py --mode vfl

# Run baseline experiments only
python run_experiments.py --mode baseline

# Run full comparison (VFL + all baselines)
python run_experiments.py --mode all

# Run unit tests
python run_experiments.py --mode test
```

### Run Unit Tests

```bash
# Run all tests
python tests/test_gradient_flow.py
python tests/test_psi.py
python tests/test_split_nn.py
```

## 📊 Results

After running experiments, results are saved in `results/`:

- **Training History**: Loss, accuracy, AUC over epochs
- **Gradient Leakage**: Privacy risk analysis over training
- **Model Comparisons**: VFL vs single-party vs Horizontal FL
- **Confusion Matrices**: Classification performance

## 🧪 Demo Scenario

### Party A (Transaction Features)
- `transaction_amount`: Transaction amount
- `transaction_count_7d`: Transaction count in 7 days
- `transaction_count_30d`: Transaction count in 30 days
- `avg_amount_7d`: Average amount in 7 days
- `time_since_last`: Hours since last transaction
- `hour_of_day`: Transaction hour (0-23)
- `day_of_week`: Day of week (0-6)

### Party B (Credit Features)
- `credit_score`: Credit score (300-850)
- `account_age_days`: Account age in days
- `income_bracket`: Income category (1-5)

### Output
- Binary classification: Fraud (1) vs Legitimate (0)

## 🔬 Experimental Results (Expected)

| Method | AUC-ROC | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| Vertical FL | ~0.92 | ~0.85 | ~0.78 | ~0.81 |
| Combined (Centralized) | ~0.94 | ~0.87 | ~0.80 | ~0.83 |
| Party A Only | ~0.78 | ~0.65 | ~0.55 | ~0.60 |
| Party B Only | ~0.72 | ~0.58 | ~0.50 | ~0.54 |
| Horizontal FL | ~0.88 | ~0.80 | ~0.72 | ~0.76 |

## 📈 Privacy Analysis

### Gradient Leakage Risk

The system monitors gradient-embedding correlation during training:

- **Low Risk**: < 15% correlation
- **Medium Risk**: 15-30% correlation
- **High Risk**: > 30% correlation

### Mitigation Strategies

1. **Gradient Noise**: Add Gaussian noise (DP-SGD)
2. **Gradient Clipping**: Limit gradient magnitude
3. **Secure Aggregation**: Encrypt gradients
4. **Embedding Dimension**: Higher dimensions = lower leakage

See `docs/privacy_analysis.md` for detailed threat model.

## 🧩 Key Components

### Private Set Intersection (PSI)

```python
from src.psi import execute_psi

intersection, metadata = execute_psi(
    party_a_ids={'user_1', 'user_2', ...},
    party_b_ids={'user_2', 'user_3', ...},
    save_path='data/psi_intersection.json'
)
```

### Vertical FL Training

```python
from src.training import VerticalFLTrainer, TrainingConfig

config = TrainingConfig(
    num_epochs=50,
    batch_size=256,
    learning_rate=0.001
)

trainer = VerticalFLTrainer(config)
history = trainer.train(X_a_train, X_b_train, y_train,
                        X_a_val, X_b_val, y_val)
```

### Gradient Leakage Analysis

```python
from src.privacy import GradientLeakageAnalyzer

analyzer = GradientLeakageAnalyzer()
report = analyzer.analyze(embeddings, embedding_gradients)

print(f"Leakage risk: {report.leakage_risk_percent:.1f}%")
print(f"Risk level: {report.risk_level}")
```

## 📚 References

1. **Romanini, et al.** "Private federated learning on vertically partitioned data" (2023)
2. **Vepakomma, et al.** "Split Learning for Collaborative Deep Learning" (2018)
3. **Zhu, et al.** "Leakage of Gradient in Vertical Federated Learning" (2022)

## 🛠️ Technical Details

### Model Architecture

**Bottom Model A (Party A)**:
- Input: 7 transaction features
- Hidden: [32, 24]
- Output: 16-dim embedding
- Parameters: ~1,800

**Bottom Model B (Party B)**:
- Input: 3 credit features
- Hidden: [16, 12]
- Output: 8-dim embedding
- Parameters: ~400

**Top Model (Server)**:
- Input: 24-dim concatenated embedding
- Hidden: [32, 16]
- Output: 2 classes (fraud/legitimate)
- Parameters: ~1,500

**Total Parameters**: ~3,700 (very lightweight!)

### Training Protocol

1. **Forward Pass**:
   - Parties compute embeddings locally (raw features stay local)
   - Embeddings sent to server
   - Server computes prediction

2. **Backward Pass**:
   - Server computes gradients wrt embeddings (∂L/∂z)
   - Embedding gradients sent to parties
   - Parties compute parameter gradients via chain rule (∂L/∂θ)
   - All models updated locally

## 🐛 Troubleshooting

**Issue**: Low AUC on test set
- **Solution**: Increase model capacity or training epochs

**Issue**: High gradient leakage risk
- **Solution**: Add gradient noise or increase embedding dimension

**Issue**: Training instability
- **Solution**: Reduce learning rate or increase gradient clipping

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

This is part of a PhD application portfolio. Suggestions and improvements welcome!

---

**Built for PhD application in Federated Learning for Fraud Detection**
