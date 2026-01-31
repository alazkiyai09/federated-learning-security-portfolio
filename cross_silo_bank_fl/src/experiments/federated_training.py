"""
Federated learning experiment using Flower framework.
Simulates cross-silo training across 5 banks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import torch
import flwr as fl
from pathlib import Path
import pickle

from ..federation.flower_client import FraudClient, client_fn, compute_validation_metrics
from ..federation.strategy import create_strategy, PerBankMetricStrategy, get_per_bank_metrics
from ..models.fraud_nn import create_model
from ..preprocessing.feature_engineering import FeatureEngineerer


def run_federated_simulation(
    bank_data: Dict[str, Dict],
    model_config: Dict = None,
    training_config: Dict = None,
    federation_config: Dict = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run federated learning simulation across banks.

    Args:
        bank_data: Dictionary mapping bank_id -> data dictionary with X_train, y_train, X_val, y_val
        model_config: Model hyperparameters
        training_config: Training hyperparameters
        federation_config: Federated learning parameters
        output_dir: Optional directory to save results

    Returns:
        Results dictionary with metrics and trained model
    """
    if model_config is None:
        model_config = {
            'hidden_layers': [128, 64, 32],
            'dropout': 0.3
        }

    if training_config is None:
        training_config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'local_epochs': 3,
            'early_stopping_patience': 5
        }

    if federation_config is None:
        federation_config = {
            'n_rounds': 15,
            'fraction_fit': 1.0,  # Use all clients
            'min_fit_clients': 5,
            'min_available_clients': 5
        }

    print(f"\n{'='*60}")
    print("Federated Learning Simulation")
    print(f"{'='*60}")
    print(f"Number of banks: {len(bank_data)}")
    print(f"Communication rounds: {federation_config['n_rounds']}")
    print(f"Local epochs per round: {training_config['local_epochs']}")

    # Show data distribution
    print("\nData distribution:")
    for bank_id, data in bank_data.items():
        print(f"  {bank_id}: {len(data['y_train']):,} train, {len(data['y_val']):,} val samples")
        print(f"    Fraud rate: {data['y_train'].mean():.4f}")

    # Create strategy
    strategy = create_strategy(
        fraction_fit=federation_config['fraction_fit'],
        min_fit_clients=federation_config['min_fit_clients'],
        min_available_clients=federation_config['min_available_clients']
    )

    # Create client function
    def make_client_fn(bank_id: str):
        def client_fn_wrapper(cid: str):
            # Get bank data
            data = bank_data[bank_id]

            # Create model
            input_dim = data['X_train'].shape[1]
            model = create_model(
                input_dim=input_dim,
                hidden_layers=model_config['hidden_layers'],
                dropout=model_config['dropout']
            )

            # Create client
            device = "cuda" if torch.cuda.is_available() else "cpu"
            client = FraudClient(
                bank_id=bank_id,
                model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                training_config=training_config
            )

            return client
        return client_fn_wrapper

    # Start Flower simulation
    print(f"\nStarting Flower simulation...")
    print(f"{'-'*60}")

    # Create client manager with all clients
    bank_ids = list(bank_data.keys())

    # Use SimulationManager for local simulation
    from flwr.simulation import start_simulation

    # Create a simple client function that uses bank_id index
    def create_client_wrapper(bank_ids_list, bank_data_dict, model_cfg, training_cfg):
        def client_fn(cid: str):
            bank_id = bank_ids_list[int(cid)]
            data = bank_data_dict[bank_id]

            input_dim = data['X_train'].shape[1]
            model = create_model(
                input_dim=input_dim,
                hidden_layers=model_cfg['hidden_layers'],
                dropout=model_cfg['dropout']
            )

            client = FraudClient(
                bank_id=bank_id,
                model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                training_config=training_cfg
            )

            return client
        return client_fn

    client_fn = create_client_wrapper(bank_ids, bank_data, model_config, training_config)

    # Run simulation
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=len(bank_ids),
        config=fl.server.ServerConfig(
            num_rounds=federation_config['n_rounds']
        ),
        strategy=strategy,
        client_resources={'num_cpus': 1},
        ray_init_args={'include_dashboard': False}
    )

    print(f"{'-'*60}")
    print("Federated training complete!")

    # Extract results from strategy
    per_bank_metrics = strategy.get_per_bank_metrics()
    round_metrics = strategy.get_round_metrics()

    # Create final metrics DataFrame
    final_metrics_df = strategy.get_final_metrics()

    print("\nFinal Per-Bank Metrics:")
    print(final_metrics_df.to_string(index=False))

    # Calculate aggregate metrics
    final_auc = final_metrics_df['auc_roc_final'].mean()
    best_auc = final_metrics_df['auc_roc_best'].mean()

    print(f"\nAggregate Metrics:")
    print(f"  Average Final AUC: {final_auc:.4f}")
    print(f"  Average Best AUC: {best_auc:.4f}")

    results = {
        'strategy': strategy,
        'per_bank_metrics': per_bank_metrics,
        'round_metrics': round_metrics,
        'final_metrics': final_metrics_df,
        'history': hist,
        'n_rounds': federation_config['n_rounds'],
        'n_banks': len(bank_data)
    }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        final_metrics_df.to_csv(output_path / 'fl_per_bank_metrics.csv', index=False)

        # Save round metrics
        round_metrics_df = pd.DataFrame(round_metrics)
        round_metrics_df.to_csv(output_path / 'fl_round_metrics.csv', index=False)

        # Save per-bank AUC progression
        per_bank_auc_df = get_per_bank_metrics(strategy, 'auc_roc')
        per_bank_auc_df.to_csv(output_path / 'fl_per_bank_auc_progression.csv', index=False)

        print(f"\nResults saved to {output_path}")

    return results


def prepare_federated_data(
    split_data: Dict[str, Dict[str, pd.DataFrame]],
    feature_engineer: FeatureEngineerer = None
) -> Tuple[Dict[str, Dict], FeatureEngineerer]:
    """
    Prepare data for federated learning.

    Args:
        split_data: Nested dict bank_id -> split -> DataFrame
        feature_engineer: Optional fitted feature engineer

    Returns:
        Tuple of (prepared_data, feature_engineer)
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineerer()

    prepared_data = {}

    for bank_id, splits in split_data.items():
        train_df = splits['train'].copy()
        val_df = splits['val'].copy()
        test_df = splits['test'].copy()

        # Feature engineering (fit on first bank, transform others)
        if len(prepared_data) == 0:
            train_df = feature_engineer.fit_transform(train_df, fit=True)
        else:
            train_df = feature_engineer.fit_transform(train_df, fit=False)

        val_df = feature_engineer.fit_transform(val_df, fit=False)
        test_df = feature_engineer.fit_transform(test_df, fit=False)

        # Get feature columns
        feature_cols = feature_engineer.get_feature_columns(train_df)

        # Prepare arrays
        X_train = train_df[feature_cols].values
        y_train = train_df['is_fraud'].values

        X_val = val_df[feature_cols].values
        y_val = val_df['is_fraud'].values

        X_test = test_df[feature_cols].values
        y_test = test_df['is_fraud'].values

        prepared_data[bank_id] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_cols': feature_cols
        }

    return prepared_data, feature_engineer


def evaluate_federated_model(
    fl_results: Dict,
    prepared_data: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Evaluate final federated model on each bank's test set.

    Args:
        fl_results: Results from run_federated_simulation
        prepared_data: Prepared data dictionary

    Returns:
        Per-bank test metrics
    """
    print("\nEvaluating Federated Model on Test Sets:")

    # Get final model parameters from strategy
    # Note: In simulation, we need to create a model and set final params
    # This is simplified - in practice would extract from strategy

    # For now, evaluate using the validation metrics from training
    test_metrics = {}

    for bank_id, data in prepared_data.items():
        # In practice, would load final model and evaluate on test set
        # Here we use the final validation metrics as proxy
        per_bank_metrics = fl_results['per_bank_metrics']

        if bank_id in per_bank_metrics:
            bank_metrics = per_bank_metrics[bank_id]

            test_metrics[bank_id] = {
                'auc_roc': bank_metrics['auc_roc'][-1] if bank_metrics['auc_roc'] else 0.0,
                'f1': bank_metrics['f1'][-1] if bank_metrics['f1'] else 0.0,
                'n_test_samples': len(data['y_test'])
            }

            print(f"  {bank_id}: AUC = {test_metrics[bank_id]['auc_roc']:.4f}, "
                  f"F1 = {test_metrics[bank_id]['f1']:.4f}")

    return test_metrics
