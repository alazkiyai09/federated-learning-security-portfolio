"""
Train local models (baseline comparison).
Each bank trains independently without sharing data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import torch
from pathlib import Path
import pickle

from ..models.fraud_nn import create_model
from ..models.training_utils import Trainer, compute_metrics
from ..preprocessing.feature_engineering import FeatureEngineerer


def train_local_models(
    bank_data: Dict[str, pd.DataFrame],
    model_config: Dict = None,
    training_config: Dict = None,
    output_dir: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Train independent local model for each bank.

    Args:
        bank_data: Dictionary mapping bank_id -> DataFrame
        model_config: Model hyperparameters
        training_config: Training hyperparameters
        output_dir: Optional directory to save models

    Returns:
        Dictionary mapping bank_id -> results dictionary
    """
    if model_config is None:
        model_config = {
            'hidden_layers': [128, 64, 32],
            'dropout': 0.3
        }

    if training_config is None:
        training_config = {
            'n_epochs': 10,
            'batch_size': 256,
            'learning_rate': 0.001,
            'early_stopping_patience': 5
        }

    results = {}

    for bank_id, df in bank_data.items():
        print(f"\n{'='*60}")
        print(f"Training local model for {bank_id}")
        print(f"{'='*60}")

        # Prepare data
        feature_engineer = FeatureEngineerer()

        # Split train/val/test
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()

        # Feature engineering
        train_df = feature_engineer.fit_transform(train_df, fit=True)
        val_df = feature_engineer.fit_transform(val_df, fit=False)
        test_df = feature_engineer.fit_transform(test_df, fit=False)

        # Get feature columns
        feature_cols = feature_engineer.get_feature_columns(train_df)

        # Prepare data arrays
        X_train = train_df[feature_cols].values
        y_train = train_df['is_fraud'].values

        X_val = val_df[feature_cols].values
        y_val = val_df['is_fraud'].values

        X_test = test_df[feature_cols].values
        y_test = test_df['is_fraud'].values

        print(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"Fraud rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

        # Create model
        input_dim = X_train.shape[1]
        model = create_model(
            input_dim=input_dim,
            hidden_layers=model_config['hidden_layers'],
            dropout=model_config['dropout']
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        trainer = Trainer(
            model=model,
            learning_rate=training_config['learning_rate'],
            verbose=True
        )

        history = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_epochs=training_config['n_epochs'],
            batch_size=training_config['batch_size'],
            early_stopping_patience=training_config['early_stopping_patience']
        )

        # Evaluate
        test_predictions = trainer.predict(X_test)
        test_metrics = compute_metrics(y_test, test_predictions)

        print(f"\nTest Results for {bank_id}:")
        print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")

        # Store results
        results[bank_id] = {
            'model': model,
            'feature_engineer': feature_engineer,
            'feature_cols': feature_cols,
            'history': history,
            'test_metrics': test_metrics,
            'test_predictions': test_predictions,
            'y_test': y_test,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

        # Save model if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            model_path = output_path / f"{bank_id}_local_model.pt"
            torch.save(model.state_dict(), model_path)

            meta_path = output_path / f"{bank_id}_local_metadata.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump({
                    'feature_cols': feature_cols,
                    'test_metrics': test_metrics,
                    'history': history
                }, f)

    return results


def evaluate_local_models(
    local_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create summary DataFrame of local model results.

    Args:
        local_results: Results dictionary from train_local_models

    Returns:
        DataFrame with per-bank metrics
    """
    summary = []

    for bank_id, result in local_results.items():
        metrics = result['test_metrics']
        summary.append({
            'bank_id': bank_id,
            'approach': 'local',
            'auc_roc': metrics['auc_roc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'n_train_samples': result['n_train_samples']
        })

    df = pd.DataFrame(summary)
    df = df.sort_values('bank_id')

    return df


def get_aggregate_local_metrics(
    local_results: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Calculate aggregate metrics across all local models.

    Args:
        local_results: Results dictionary from train_local_models

    Returns:
        Dictionary with aggregate metrics
    """
    # Average metrics weighted by sample size
    total_samples = sum(r['n_test_samples'] for r in local_results.values())

    weighted_auc = sum(
        r['test_metrics']['auc_roc'] * r['n_test_samples']
        for r in local_results.values()
    ) / total_samples

    weighted_f1 = sum(
        r['test_metrics']['f1'] * r['n_test_samples']
        for r in local_results.values()
    ) / total_samples

    return {
        'weighted_avg_auc_roc': weighted_auc,
        'weighted_avg_f1': weighted_f1,
        'n_banks': len(local_results),
        'total_test_samples': total_samples
    }
