"""
Train centralized model (upper bound baseline).
Uses all data pooled together - privacy-invasive but best possible performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import torch
from pathlib import Path
import pickle

from ..models.fraud_nn import create_model
from ..models.training_utils import Trainer, compute_metrics
from ..preprocessing.feature_engineering import FeatureEngineerer


def train_centralized_model(
    centralized_data: Dict[str, pd.DataFrame],
    model_config: Dict = None,
    training_config: Dict = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Train centralized model on pooled data from all banks.

    This represents the upper bound on performance (privacy-invasive).

    Args:
        centralized_data: Dict with 'train', 'val', 'test' DataFrames
        model_config: Model hyperparameters
        training_config: Training hyperparameters
        output_dir: Optional directory to save model

    Returns:
        Results dictionary
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

    print(f"\n{'='*60}")
    print("Training Centralized Model (All Data Pooled)")
    print(f"{'='*60}")

    # Prepare data
    feature_engineer = FeatureEngineerer()

    train_df = centralized_data['train'].copy()
    val_df = centralized_data['val'].copy()
    test_df = centralized_data['test'].copy()

    # Feature engineering
    train_df = feature_engineer.fit_transform(train_df, fit=True)
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

    print(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Fraud rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

    # Show per-bank distribution in training data
    if 'bank_id' in train_df.columns:
        bank_dist = train_df['bank_id'].value_counts()
        print("\nTraining data distribution:")
        for bank_id, count in bank_dist.items():
            print(f"  {bank_id}: {count:,} samples")

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

    print(f"\nCentralized Model Test Results:")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

    # Evaluate per bank
    per_bank_metrics = {}
    if 'bank_id' in test_df.columns:
        print("\nPer-Bank Performance (Centralized Model):")
        for bank_id in test_df['bank_id'].unique():
            bank_mask = test_df['bank_id'] == bank_id
            bank_y = y_test[bank_mask]
            bank_preds = test_predictions[bank_mask]

            bank_metrics = compute_metrics(bank_y, bank_preds)
            per_bank_metrics[bank_id] = bank_metrics

            print(f"  {bank_id}: AUC = {bank_metrics['auc_roc']:.4f}, "
                  f"F1 = {bank_metrics['f1']:.4f}")

    results = {
        'model': model,
        'feature_engineer': feature_engineer,
        'feature_cols': feature_cols,
        'history': history,
        'test_metrics': test_metrics,
        'per_bank_metrics': per_bank_metrics,
        'test_predictions': test_predictions,
        'y_test': y_test,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    # Save model
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / "centralized_model.pt"
        torch.save(model.state_dict(), model_path)

        meta_path = output_path / "centralized_metadata.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_cols': feature_cols,
                'test_metrics': test_metrics,
                'per_bank_metrics': per_bank_metrics,
                'history': history
            }, f)

    return results
