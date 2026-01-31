#!/usr/bin/env python3
"""
Simple test runner without pytest dependency
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run basic import tests
print("="*60)
print("Running Import Tests")
print("="*60)

try:
    print("\n[1/6] Importing src.model...")
    from src.model import FraudDetectionModel, FraudDetectionLoss, create_model
    print("  OK")

    print("\n[2/6] Importing src.client...")
    from src.client import FlClient, create_client
    print("  OK")

    print("\n[3/6] Importing src.server...")
    from src.server import get_strategy, get_initial_parameters, start_server_with_strategy
    print("  OK")

    print("\n[4/6] Importing src.strategy...")
    from src.strategy import FedAvgCustom, FedProxCustom, FedAdamCustom
    print("  OK")

    print("\n[5/6] Importing src.data...")
    from src.data import (
        load_synthetic_fraud_data,
        create_data_loaders,
        partition_data_iid,
        partition_data_dirichlet,
        prepare_federated_data,
    )
    print("  OK")

    print("\n[6/6] Importing src.utils...")
    from src.utils import (
        weighted_average,
        aggregate_q,
        compute_fraud_metrics,
        TensorBoardLogger,
        set_seed,
    )
    print("  OK")

    print("\n" + "="*60)
    print("All imports successful!")
    print("="*60)

    # Run simple functional tests
    print("\n" + "="*60)
    print("Running Functional Tests")
    print("="*60)

    print("\n[Test 1] Create model...")
    model = FraudDetectionModel(input_dim=10, hidden_dims=[8, 4])
    num_params = model.get_num_parameters()
    print(f"  Model created with {num_params} parameters")
    assert num_params > 0, "Model should have parameters"

    print("\n[Test 2] Generate synthetic data...")
    X, y = load_synthetic_fraud_data(n_samples=100, n_features=10, seed=42)
    print(f"  Generated {X.shape} features, {y.shape} labels")
    assert X.shape == (100, 10), "Wrong X shape"
    assert y.shape == (100,), "Wrong y shape"

    print("\n[Test 3] Weighted average aggregation...")
    metrics = [
        (100, {"accuracy": 0.8, "loss": 0.5}),
        (200, {"accuracy": 0.9, "loss": 0.4}),
    ]
    total_samples, aggregated = weighted_average(metrics)
    print(f"  Total samples: {total_samples}")
    print(f"  Aggregated accuracy: {aggregated['accuracy']:.4f}")
    print(f"  Aggregated loss: {aggregated['loss']:.4f}")
    assert total_samples == 300, "Wrong total samples"
    assert abs(aggregated["accuracy"] - 0.8667) < 0.001, "Wrong accuracy aggregation"

    print("\n[Test 4] Fraud metrics computation...")
    import numpy as np
    predictions = np.array([0, 1, 0, 1, 0, 1])
    targets = np.array([0, 1, 0, 1, 0, 1])
    metrics = compute_fraud_metrics(predictions, targets)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    assert metrics["accuracy"] == 1.0, "Perfect predictions should give 100% accuracy"

    print("\n[Test 5] Strategy initialization...")
    fedavg = FedAvgCustom(
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )
    print(f"  FedAvg created: {fedavg}")

    fedprox = FedProxCustom(
        proximal_mu=0.01,
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )
    print(f"  FedProx created (mu={fedprox.proximal_mu}): {fedprox}")

    fedadam = FedAdamCustom(
        tau=0.9,
        eta=0.01,
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )
    print(f"  FedAdam created (tau={fedadam.tau}, eta={fedadam.eta}): {fedadam}")

    print("\n[Test 6] Quantile aggregation...")
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    median = aggregate_q(values, 0.5)
    print(f"  Median of {values}: {median}")
    assert median == 3.0, "Wrong median"

    print("\n[Test 7] Random seed reproducibility...")
    import random
    set_seed(42)
    val1 = random.random()
    set_seed(42)
    val2 = random.random()
    print(f"  First draw: {val1:.6f}")
    print(f"  Second draw (same seed): {val2:.6f}")
    assert val1 == val2, "Seed should produce same results"

    print("\n" + "="*60)
    print("All functional tests PASSED!")
    print("="*60)

    # Check for missing dependencies
    print("\n" + "="*60)
    print("Dependency Check")
    print("="*60)

    missing = []
    try:
        import torch
        print("  torch: OK")
    except ImportError:
        print("  torch: MISSING")
        missing.append("torch")

    try:
        import numpy as np
        print("  numpy: OK")
    except ImportError:
        print("  numpy: MISSING")
        missing.append("numpy")

    try:
        from omegaconf import OmegaConf
        print("  omegaconf: OK")
    except ImportError:
        print("  omegaconf: MISSING")
        missing.append("omegaconf")

    try:
        from hydra import compose, initialize
        print("  hydra-core: OK")
    except ImportError:
        print("  hydra-core: MISSING")
        missing.append("hydra-core")

    try:
        from flwr.server import ServerConfig
        print("  flwr (Flower): OK")
    except ImportError:
        print("  flwr (Flower): MISSING")
        missing.append("flwr")

    try:
        from sklearn.model_selection import train_test_split
        print("  scikit-learn: OK")
    except ImportError:
        print("  scikit-learn: MISSING")
        missing.append("scikit-learn")

    try:
        from scipy.stats import dirichlet
        print("  scipy: OK")
    except ImportError:
        print("  scipy: MISSING")
        missing.append("scipy")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\nAll dependencies installed!")

    print("\n" + "="*60)
    print("TEST SUMMARY: ALL TESTS PASSED")
    print("="*60)
    sys.exit(0)

except Exception as e:
    print(f"\n{'='*60}")
    print("TEST FAILED")
    print('='*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
