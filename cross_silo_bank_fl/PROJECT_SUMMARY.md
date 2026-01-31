# Cross-Silo Bank Federated Learning Simulation - Project Summary

## Project Overview

This project simulates realistic federated learning across 5 banks for fraud detection, demonstrating the benefits of cross-silo collaboration while preserving data privacy. It's designed as a PhD research portfolio piece demonstrating trustworthy FL in financial services.

## Project Structure

```
cross_silo_bank_fl/
├── config/
│   ├── bank_profiles.yaml          # Realistic bank configurations
│   └── simulation_config.yaml      # FL hyperparameters
│
├── data/
│   ├── raw/                        # Generated synthetic bank data
│   └── processed/                  # Partitioned and preprocessed data
│
├── src/
│   ├── data_generation/            # Phase 1
│   │   ├── bank_profile.py         # BankProfile dataclass
│   │   ├── transaction_generator.py # Transaction pattern generation
│   │   └── fraud_generator.py      # Fraud pattern injection
│   │
│   ├── preprocessing/              # Phase 2
│   │   ├── feature_engineering.py  # Feature creation
│   │   └── partitioner.py          # Data splitting
│   │
│   ├── models/                     # Phase 3
│   │   ├── fraud_nn.py             # Neural network architecture
│   │   └── training_utils.py       # Training utilities
│   │
│   ├── experiments/                # Phase 4 & 5
│   │   ├── local_baseline.py       # Independent local training
│   │   ├── federated_training.py   # FL with Flower
│   │   └── centralized_baseline.py # Pooled data baseline
│   │
│   ├── federation/                 # Phase 5
│   │   ├── flower_client.py        # Flower client implementation
│   │   ├── strategy.py             # Custom FedAvg with per-bank tracking
│   │   └── secure_aggregation.py   # Additive masking simulation
│   │
│   ├── evaluation/                 # Phase 6
│   │   ├── metrics.py              # Metrics calculation
│   │   └── visualization.py        # Plot generation
│   │
│   └── utils/
│       └── helpers.py              # Utility functions
│
├── tests/                          # Phase 7
│   ├── test_bank_profiles.py       # Validate realistic bank profiles
│   ├── test_data_generation.py     # Test data quality
│   └── test_federation.py          # Test FL logic
│
├── results/
│   ├── metrics/                    # CSV files with metrics
│   └── figures/                    # PNG visualizations
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration
│
├── main.py                         # Entry point
├── requirements.txt
├── pytest.ini
└── .gitignore
```

## Implementation Phases

### Phase 1: Data Generation (Complete)
- **bank_profile.py**: BankProfile dataclass with 5 realistic bank profiles
- **transaction_generator.py**: Generates transaction amounts, timestamps, merchants
- **fraud_generator.py**: Injects realistic fraud patterns per bank type

### Phase 2: Preprocessing (Complete)
- **feature_engineering.py**: Temporal, behavioral, and contextual features
- **partitioner.py**: Non-IID partitioning and train/val/test splits

### Phase 3: Model Architecture (Complete)
- **fraud_nn.py**: PyTorch neural network with embeddings
- **training_utils.py**: Trainer class, metrics, early stopping

### Phase 4: Baselines (Complete)
- **local_baseline.py**: Train 5 independent local models
- **centralized_baseline.py**: Train on pooled data (privacy upper bound)

### Phase 5: Federation (Complete)
- **flower_client.py**: Flower Client with fit() and evaluate()
- **strategy.py**: Custom FedAvg with per-bank metric tracking
- **secure_aggregation.py**: Additive masking simulation
- **federated_training.py**: Full FL simulation orchestration

### Phase 6: Evaluation (Complete)
- **metrics.py**: Per-bank and aggregate metrics
- **visualization.py**: 5 visualization functions for results

### Phase 7: Integration (Complete)
- **main.py**: Orchestrates all experiments and generates README
- **tests/**: Unit tests for validation

## Bank Profiles

| Bank | Type | Clients | Daily Tx | Fraud Rate | Key Characteristics |
|------|------|---------|----------|------------|---------------------|
| **Bank A** | Retail | 500K | 150K | 0.25% | Large retail bank, diverse fraud |
| **Bank B** | Regional | 80K | 25K | 0.18% | Regional bank, local patterns |
| **Bank C** | Digital | 120K | 45K | 0.80% | Neobank, synthetic identity fraud |
| **Bank D** | Credit Union | 35K | 8K | 0.10% | Member-focused, lowest fraud |
| **Bank E** | International | 300K | 90K | 0.35% | Global bank, cross-border fraud |

## Key Features

### 1. Realistic Bank Profiles
- Based on real-world banking characteristics
- Different customer demographics (age, income)
- Varying fraud patterns and rates
- Different transaction volumes and merchant distributions

### 2. Non-IID Data Distribution
- Each bank has unique data characteristics
- Realistic fraud type distributions per bank
- Geographic and transactional differences

### 3. Fair Comparison Framework
- Same model architecture across all approaches
- Same hyperparameters for fair comparison
- Three approaches: Local, Federated, Centralized

### 4. Per-Bank Performance Tracking
- Custom Flower strategy tracks metrics per bank
- Shows which banks benefit most from federation
- Demonstrates heterogeneous benefits

### 5. Secure Aggregation Simulation
- Additive masking for privacy
- Pairwise masking demonstration
- Shows how privacy is preserved

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Simulation
```bash
# Run all experiments (local, federated, centralized)
python main.py

# With custom parameters
python main.py --n-rounds 20 --local-epochs 5 --seed 123
```

### Run Specific Components
```bash
# Skip local baseline
python main.py --skip-local

# Skip federated learning
python main.py --skip-fl

# Skip centralized baseline
python main.py --skip-centralized
```

### Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bank_profiles.py -v

# Run with coverage
pytest --cov=src tests/
```

## Key Metrics Tracked

### Primary Metrics
- **AUC-ROC**: Area under ROC curve (main metric)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Per-Bank Analysis
- FL vs Local improvement (%)
- Centralized gap (how close FL is to upper bound)
- Performance progression over communication rounds

## Expected Results

Based on bank profiles:

1. **Bank C (Digital)**: Highest fraud rate, benefits most from federation
2. **Bank D (Credit Union)**: Smallest bank, large improvement from collaboration
3. **Bank A (Retail)**: Moderate improvement
4. **Bank E (International)**: Unique patterns, gains from diverse data
5. **Bank B (Regional)**: Local patterns, moderate benefit

## Output Files

### Metrics (results/metrics/)
- `comparison_metrics.csv`: Per-bank comparison
- `fl_per_bank_metrics.csv`: FL metrics per bank
- `improvements.json`: Improvement statistics
- `aggregates.json`: Aggregate metrics

### Figures (results/figures/)
- `per_bank_comparison.png`: Bar chart comparison
- `learning_curves.png`: FL progression
- `fraud_analysis.png`: Bank characteristics
- `improvement_analysis.png`: FL vs Local improvement
- `summary.png`: Combined summary figure

## Technical Details

### Model Architecture
- Input: 25+ engineered features
- Hidden layers: [128, 64, 32]
- Dropout: 0.3
- Output: Binary classification (logits)
- Parameters: ~50K trainable parameters

### Training
- Optimizer: Adam
- Learning rate: 0.001
- Loss: BCEWithLogitsLoss
- Batch size: 256
- Early stopping: 5 epochs patience

### Federated Learning
- Framework: Flower 1.8+
- Strategy: FedAvg with per-bank tracking
- Rounds: 15
- Local epochs: 3
- Client participation: 100% (all banks)

## Research Contributions

This project demonstrates:

1. **Realistic FL Simulation**: Bank profiles based on industry knowledge
2. **Per-Bank Analysis**: Shows heterogeneous benefits of federation
3. **Privacy-Preserving**: Performance close to centralized without data sharing
4. **Reproducible**: Complete code with unit tests
5. **PhD-Ready**: Suitable for research portfolio

## Future Extensions

- Add more banks (10-20 banks)
- Implement differential privacy
- Add more sophisticated fraud patterns
- Experiment with different FL strategies (FedProx, Scaffold)
- Real-time fraud detection simulation
- Economic analysis of FL benefits

## Contact

For questions about this project, please refer to the generated README.md in the results directory.
