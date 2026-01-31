#!/usr/bin/env python3
"""
Main entry point for Vertical Federated Learning experiments.

Usage:
    python run_experiments.py --mode all
    python run_experiments.py --mode vfl
    python run_experiments.py --mode baseline
    python run_experiments.py --mode test
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_loader import generate_synthetic_data, create_data_splits
from src.psi.private_set_intersection import execute_psi
from src.experiments.vertical_fl import run_full_comparison, VerticalFLExperiment
from src.training.vertical_fl_trainer import TrainingConfig


def setup_experiment(args):
    """Setup experiment: generate data, run PSI, create splits."""
    print("\n" + "="*80)
    print("VERTICAL FEDERATED LEARNING EXPERIMENT SETUP")
    print("="*80)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    raw_dir = os.path.join(data_dir, 'raw')

    # Check if data exists
    if os.path.exists(os.path.join(raw_dir, 'party_a_transactions.csv')):
        print("\nData already exists. Skipping generation.")
        print("To regenerate data, delete the data/ directory and run again.")
    else:
        print("\nGenerating synthetic fraud detection data...")

        # Load config
        with open('config/experiment_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Generate data
        party_a, party_b, labels = generate_synthetic_data(
            num_samples=config['data']['num_samples'],
            fraud_ratio=config['data']['fraud_ratio'],
            random_seed=config['data']['random_seed'],
            save_path=raw_dir
        )

        print(f"\nData generated:")
        print(f"  Party A (Transaction): {party_a.shape}")
        print(f"  Party B (Credit): {party_b.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Fraud rate: {labels['is_fraud'].mean():.2%}")

    # Run PSI for ID alignment
    print("\n" + "-"*80)
    print("Running Private Set Intersection (PSI) for ID alignment...")

    party_a_data = pd.read_csv(os.path.join(raw_dir, 'party_a_transactions.csv'))
    party_b_data = pd.read_csv(os.path.join(raw_dir, 'party_b_credit.csv'))

    party_a_ids = set(party_a_data['user_id'].astype(str).tolist())
    party_b_ids = set(party_b_data['user_id'].astype(str).tolist())

    intersection, metadata = execute_psi(
        party_a_ids,
        party_b_ids,
        save_path=os.path.join(data_dir, 'psi_intersection.json')
    )

    print(f"\nPSI Results:")
    print(f"  Party A users: {metadata['client_set_size']:,}")
    print(f"  Party B users: {metadata['server_set_size']:,}")
    print(f"  Intersection: {metadata['intersection_size']:,}")
    print(f"  Privacy: {metadata['privacy_guarantee']}")

    # Create train/val/test splits
    print("\n" + "-"*80)
    print("Creating train/val/test splits...")

    create_data_splits(data_dir)

    print("\n✓ Setup complete!")


def load_data_splits(data_dir='data'):
    """Load train/val/test splits."""
    import json
    from src.utils.data_loader import load_aligned_data

    # Load split info
    with open(os.path.join(data_dir, 'processed', 'split_info.json'), 'r') as f:
        split_info = json.load(f)

    # Load data
    train = load_aligned_data(os.path.join(data_dir, 'processed'), split='train')
    val = load_aligned_data(os.path.join(data_dir, 'processed'), split='val')
    test = load_aligned_data(os.path.join(data_dir, 'processed'), split='test')

    return train, val, test, split_info


def run_vfl_experiment(train, val, test, args):
    """Run Vertical FL experiment only."""
    print("\n" + "="*80)
    print("VERTICAL FEDERATED LEARNING EXPERIMENT")
    print("="*80)

    # Load config
    with open('config/experiment_config.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)

    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)

    # Setup training config
    config = TrainingConfig(
        num_epochs=exp_config['training']['num_epochs'],
        batch_size=exp_config['training']['batch_size'],
        learning_rate=exp_config['training']['learning_rate'],
        gradient_clip=exp_config['training']['gradient_clip'],
        early_stopping_patience=10,
        analyze_gradient_leakage=exp_config['privacy']['analyze_gradient_leakage']
    )

    # Create experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    exp = VerticalFLExperiment(
        config=config,
        model_config=model_config,
        device=device
    )

    # Run experiment
    exp.run_vertical_fl(
        train['X_a'], train['X_b'], train['y'],
        val['X_a'], val['X_b'], val['y'],
        test['X_a'], test['X_b'], test['y']
    )

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'vfl')
    exp.save_results(results_dir)
    exp.print_summary()

    return exp


def run_baselines(train, val, test, args):
    """Run baseline experiments only."""
    print("\n" + "="*80)
    print("BASELINE EXPERIMENTS")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    from src.experiments.single_party_baseline import run_single_party_baselines
    from src.experiments.horizontal_fl_baseline import run_horizontal_fl_baseline

    # Single-party baselines
    print("\n" + "-"*80)
    print("Single-Party Baselines")
    print("-"*80)

    baseline_results = run_single_party_baselines(
        train['X_a'], test['X_a'],
        train['X_b'], test['X_b'],
        train['y'], test['y'],
        device=device
    )

    # Horizontal FL
    print("\n" + "-"*80)
    print("Horizontal FL Baseline")
    print("-"*80)

    hfl_results = run_horizontal_fl_baseline(
        train['X_a'], test['X_a'],
        train['X_b'], test['X_b'],
        train['y'], test['y'],
        num_clients=3,
        num_rounds=20,
        device=device
    )

    return {**baseline_results, **hfl_results}


def run_full_comparison_experiment(train, val, test, args):
    """Run full comparison: VFL vs all baselines."""
    print("\n" + "="*80)
    print("FULL COMPARISON: VFL vs BASELINES")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Run full comparison
    all_results = run_full_comparison(
        train['X_a'], test['X_a'],
        train['X_b'], test['X_b'],
        train['y'], test['y'],
        device=device
    )

    # Save results
    import json
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Convert to serializable format
    serializable_results = {}
    for name, metrics in all_results.items():
        if isinstance(metrics, dict):
            serializable_results[name] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                          for k, v in metrics.items()}

    with open(os.path.join(results_dir, 'comparison_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return all_results


def run_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)

    import subprocess

    tests = [
        ('PSI Tests', 'tests/test_psi.py'),
        ('Gradient Flow Tests', 'tests/test_gradient_flow.py'),
        ('SplitNN Tests', 'tests/test_split_nn.py'),
    ]

    results = []

    for name, test_path in tests:
        print(f"\nRunning {name}...")
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True
        )

        passed = result.returncode == 0
        results.append((name, passed))

        if passed:
            print(f"✓ {name} PASSED")
        else:
            print(f"✗ {name} FAILED")
            print(result.stdout)
            print(result.stderr)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Vertical FL for Fraud Detection')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['setup', 'vfl', 'baseline', 'all', 'test'],
        default='all',
        help='Experiment mode'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory'
    )

    args = parser.parse_args()

    # Import pandas here (only needed for setup)
    import pandas as pd

    if args.mode == 'test':
        return run_tests()

    # Setup data if needed
    if args.mode in ['setup', 'all', 'vfl', 'baseline']:
        setup_experiment(args)

        # Load data
        train, val, test, split_info = load_data_splits(args.data_dir)

        print(f"\nData splits:")
        print(f"  Train: {len(train['y']):,}")
        print(f"  Val: {len(val['y']):,}")
        print(f"  Test: {len(test['y']):,}")

    # Run experiments
    if args.mode == 'vfl':
        run_vfl_experiment(train, val, test, args)
    elif args.mode == 'baseline':
        run_baselines(train, val, test, args)
    elif args.mode == 'all':
        run_full_comparison_experiment(train, val, test, args)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
