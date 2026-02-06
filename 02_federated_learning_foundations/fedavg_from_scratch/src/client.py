"""
Federated Client implementation.

Handles local training on client data partitions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np

from src.utils import serialize_weights, deserialize_weights, create_optimizer


StateDict = Dict[str, torch.Tensor]


class FederatedClient:
    """
    Federated learning client for local training.

    Implements local SGD training on client's private data partition.
    Returns weight updates for server aggregation.

    Attributes:
        client_id: Unique client identifier
        model: Local neural network model
        train_loader: DataLoader for client's training data
        config: Training configuration dictionary

    Example:
        >>> client = FederatedClient(
        ...     client_id=0,
        ...     model=SimpleCNN(),
        ...     train_loader=train_loader,
        ...     config={'local_epochs': 5, 'learning_rate': 0.01}
        ... )
        >>> weights, num_samples = client.local_train(global_weights)
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        config: Dict
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model to train
            train_loader: DataLoader with client's training data
            config: Training configuration with keys:
                - local_epochs: Number of local training epochs
                - learning_rate: Learning rate for optimizer
                - optimizer_type: Type of optimizer ('sgd' or 'adam')
                - momentum: Momentum for SGD (default: 0.9)
                - weight_decay: Weight decay (default: 0.0)
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config

        # Get training configuration
        self.local_epochs = config.get('local_epochs', 5)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.optimizer_type = config.get('optimizer_type', 'sgd')

        # Get device
        self.device = config.get('device', torch.device('cpu'))
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            self.optimizer_type,
            self.learning_rate,
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training statistics
        self.num_samples = len(train_loader.dataset)
        self.loss_history = []

    def local_train(
        self,
        global_weights: StateDict,
        verbose: bool = False
    ) -> Tuple[StateDict, int]:
        """
        Perform local training starting from global weights.

        Implements FedAvg local training:
        1. Load global weights
        2. Train for E local epochs on client data
        3. Return updated weights

        Args:
            global_weights: Current global model weights
            verbose: Whether to show training progress

        Returns:
            Tuple[StateDict, int]:
                - Updated local model weights
                - Number of training samples (for weighted averaging)
        """
        # Load global weights
        deserialize_weights(self.model, global_weights)

        # Set model to training mode
        self.model.train()

        # Local training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            if verbose:
                pbar = tqdm(self.train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}/{self.local_epochs}")
            else:
                pbar = self.train_loader

            for batch_idx, (data, target) in enumerate(pbar):
                # Move to device
                data, target = data.to(self.device), target.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)

            if verbose:
                print(f"Client {self.client_id} | Epoch {epoch+1}/{self.local_epochs} | Loss: {avg_loss:.4f}")

        # Return updated weights and sample count
        local_weights = serialize_weights(self.model)
        return local_weights, self.num_samples

    def evaluate(
        self,
        test_loader: DataLoader,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_loader: DataLoader with test data
            verbose: Whether to print results

        Returns:
            Dict[str, float]: Dictionary with 'loss', 'accuracy', and 'num_samples'
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        }

        if verbose:
            print(f"Client {self.client_id} Evaluation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        return metrics

    def get_loss_history(self) -> list:
        """
        Get training loss history.

        Returns:
            list: Loss values per local epoch
        """
        return self.loss_history


class FederatedClientBinary(FederatedClient):
    """
    Federated client for binary classification tasks (e.g., fraud detection).

    Uses BCELoss instead of CrossEntropyLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override loss function for binary classification
        self.criterion = nn.BCELoss()

    def evaluate(self, test_loader: DataLoader, verbose: bool = False) -> Dict[str, float]:
        """
        Evaluate binary classification model.

        Returns additional metrics: precision, recall, f1, auc_pr
        """
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = test_loss / len(test_loader)

        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        # Binary predictions at 0.5 threshold
        binary_preds = (all_preds >= 0.5).astype(int)

        # Calculate metrics
        accuracy = (binary_preds == all_targets).mean()

        # Handle edge case where no positive predictions
        if binary_preds.sum() == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
            precision = precision_score(all_targets, binary_preds, zero_division=0)
            recall = recall_score(all_targets, binary_preds, zero_division=0)
            f1 = f1_score(all_targets, binary_preds, zero_division=0)

        # AUC-PR (Area Under Precision-Recall Curve)
        try:
            from sklearn.metrics import average_precision_score
            auc_pr = average_precision_score(all_targets, all_preds)
        except (ImportError, ValueError) as e:
            # Log warning but don't fail - auc_pr is optional metric
            auc_pr = 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_pr': auc_pr,
            'num_samples': len(all_targets)
        }

        if verbose:
            print(f"Client {self.client_id} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | "
                  f"F1: {f1:.4f} | AUC-PR: {auc_pr:.4f}")

        return metrics
