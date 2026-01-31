"""
Single-party baseline experiments.

Trains models on:
1. Party A data only (transaction features)
2. Party B data only (credit features)
3. Combined data (centralized, non-FL)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score

from ..utils.metrics import compute_metrics, print_metrics_table


class SinglePartyBaseline:
    """Train and evaluate single-party baselines."""

    def __init__(self, input_dim: int, device: str = 'cpu'):
        """
        Initialize baseline experiment.

        Args:
            input_dim: Number of input features
            device: Device to train on
        """
        self.input_dim = input_dim
        self.device = device
        self.model = None

    def _build_model(self, hidden_dims: list = [64, 32, 16]) -> nn.Module:
        """Build simple neural network."""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        return nn.Sequential(*layers)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train single-party model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dictionary with training history and metrics
        """
        self.model = self._build_model().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

        num_samples = len(X_train)
        num_batches = num_samples // batch_size

        best_val_auc = 0.0

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_x = torch.FloatTensor(X_train[start_idx:end_idx]).to(self.device)
                batch_y = torch.LongTensor(y_train[start_idx:end_idx]).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(X_val).to(self.device)
                val_outputs = self.model(val_x)
                val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

            val_loss = nn.CrossEntropyLoss()(
                val_outputs,
                torch.LongTensor(y_val).to(self.device)
            ).item()

            val_auc = roc_auc_score(y_val, val_probs)

            history['train_loss'].append(epoch_loss / num_batches)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            best_val_auc = max(best_val_auc, val_auc)

            if epoch % 10 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}], "
                      f"Loss: {epoch_loss/num_batches:.4f}, "
                      f"Val AUC: {val_auc:.4f}")

        return {
            'history': history,
            'best_val_auc': best_val_auc
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        with torch.no_grad():
            test_x = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(test_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.cpu().numpy(), axis=1)

        metrics = compute_metrics(y_test, preds, probs)
        metrics['auc_roc'] = roc_auc_score(y_test, probs)
        metrics['auc_pr'] = compute_metrics(y_test, preds, probs).get('auc_pr', 0.0)

        return metrics


def run_single_party_baselines(
    X_a_train: np.ndarray,
    X_a_test: np.ndarray,
    X_b_train: np.ndarray,
    X_b_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Run all single-party baseline experiments.

    Args:
        X_a_train: Party A training features
        X_a_test: Party A test features
        X_b_train: Party B training features
        X_b_test: Party B test features
        y_train: Training labels
        y_test: Test labels
        device: Device to train on

    Returns:
        Dictionary of {method_name: metrics}
    """
    results = {}

    print("\n=== Single-Party Baselines ===\n")

    # Party A only
    print("Training Party A only (transaction features)...")
    baseline_a = SinglePartyBaseline(input_dim=X_a_train.shape[1], device=device)
    baseline_a.train(X_a_train, y_train, X_a_test, y_test)
    metrics_a = baseline_a.evaluate(X_a_test, y_test)
    results['Party A Only'] = metrics_a

    # Party B only
    print("\nTraining Party B only (credit features)...")
    baseline_b = SinglePartyBaseline(input_dim=X_b_train.shape[1], device=device)
    baseline_b.train(X_b_train, y_train, X_b_test, y_test)
    metrics_b = baseline_b.evaluate(X_b_test, y_test)
    results['Party B Only'] = metrics_b

    # Combined (centralized)
    print("\nTraining Combined (centralized, all features)...")
    X_combined_train = np.concatenate([X_a_train, X_b_train], axis=1)
    X_combined_test = np.concatenate([X_a_test, X_b_test], axis=1)

    baseline_combined = SinglePartyBaseline(input_dim=X_combined_train.shape[1], device=device)
    baseline_combined.train(X_combined_train, y_train, X_combined_test, y_test)
    metrics_combined = baseline_combined.evaluate(X_combined_test, y_test)
    results['Combined (Centralized)'] = metrics_combined

    # Print comparison
    print_metrics_table(results, "Single-Party Baseline Results")

    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Single-Party Baselines...")

    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 1000, 500

    X_a_train = np.random.randn(n_train, 7)
    X_b_train = np.random.randn(n_train, 3)
    y_train = np.random.randint(0, 2, n_train)

    X_a_test = np.random.randn(n_test, 7)
    X_b_test = np.random.randn(n_test, 3)
    y_test = np.random.randint(0, 2, n_test)

    results = run_single_party_baselines(
        X_a_train, X_a_test, X_b_train, X_b_test, y_train, y_test
    )

    print("\n✓ Single-party baselines working correctly")
