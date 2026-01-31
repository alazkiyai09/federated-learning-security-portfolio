"""
Training utilities for fraud detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time


class Trainer:
    """
    Trainer for fraud detection models.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on
            verbose: Whether to print progress
        """
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function (binary classification with class imbalance)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': []
        }

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Prepare data for training/evaluation.

        Args:
            X: Feature array
            y: Label array
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if self.verbose:
            print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation data.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        all_predictions = []
        all_labels = []

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions
            probs = torch.sigmoid(logits)
            all_predictions.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        metrics = compute_metrics(labels, predictions)

        avg_loss = total_loss / n_batches

        return avg_loss, metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_epochs: int = 10,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history
        """
        self.verbose = verbose

        # Prepare data loaders
        train_loader = self.prepare_data(X_train, y_train, batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self.prepare_data(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_auc'].append(val_metrics['auc_roc'])
                self.history['val_f1'].append(val_metrics['f1'])

                if verbose:
                    print(f"  Val Loss = {val_loss:.4f}, AUC = {val_metrics['auc_roc']:.4f}, "
                          f"F1 = {val_metrics['f1']:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if val_loader is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)

        return self.history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature array

        Returns:
            Probability predictions
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        probs = torch.sigmoid(self.model(X_tensor))
        predictions = probs.cpu().numpy().flatten()

        return predictions


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc_roc = 0.0

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'auc_roc': auc_roc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def calculate_positive_weight(y: np.ndarray) -> float:
    """
    Calculate positive class weight for imbalanced data.

    Args:
        y: Label array

    Returns:
        Weight for positive class
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        """
        Check if should stop training.

        Args:
            val_score: Current validation score (higher is better)

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop
