"""
Flower Client for Fraud Detection

Implements FlClient extending fl.client.NumPyClient with
parameter handling, training, and evaluation.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import FitRes, Parameters, Scalar, arrays_to_ndarrays, ndarrays_to_arrays
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import FraudDetectionLoss, FraudDetectionModel
from src.utils import compute_fraud_metrics


class FlClient(NumPyClient):
    """
    Flower Client for federated fraud detection.

    Extends NumPyClient to implement local training and evaluation
    with support for FedProx proximal term.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        device: str = "cpu",
    ) -> None:
        """
        Initialize Flower client.

        Args:
            model: PyTorch model for fraud detection
            train_loader: Training data loader
            test_loader: Test/validation data loader
            config: Configuration dictionary
            device: Device to use for computation (cpu or cuda)
        """
        super().__init__()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Initialize loss function
        self.criterion = FraudDetectionLoss(pos_weight=10.0)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.

        Returns:
            PyTorch optimizer
        """
        optimizer_name = self.config.get("optimizer", "adam").lower()
        learning_rate = self.config.get("learning_rate", 0.01)

        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Server configuration (not used directly)

        Returns:
            List of numpy arrays containing model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of numpy arrays containing model parameters
        """
        # Convert parameters to state dict format
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(v) for k, v in params_dict}

        # Load state dict
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train the model on local data.

        Args:
            parameters: Global model parameters from server
            config: Training configuration (local_epochs, proximal_mu, etc.)

        Returns:
            (updated_parameters, num_samples, metrics) tuple
        """
        # Set global parameters
        self.set_parameters(parameters)

        # Get training config
        local_epochs = int(config.get("local_epochs", self.config.get("local_epochs", 5)))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Store initial global parameters for proximal term
        if proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        # Training loop
        self.model.train()
        epoch_losses = []

        for epoch in range(local_epochs):
            batch_losses = []
            for batch_idx, (features, labels) in enumerate(self.train_loader):
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(features)

                # Compute loss
                loss = self.criterion(predictions, labels)

                # Add proximal term if using FedProx
                if proximal_mu > 0:
                    proximal_term = 0.0
                    for p, g_p in zip(self.model.parameters(), global_params):
                        proximal_term += torch.norm(p - g_p) ** 2
                    loss = loss + (proximal_mu / 2) * proximal_term

                # Backward pass
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            # Compute epoch loss
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)

        # Get training metrics
        train_metrics = self._evaluate_on_train()

        # Return updated parameters and metrics
        updated_parameters = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "loss": float(np.mean(epoch_losses)),
            **train_metrics,
        }

        return updated_parameters, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local test data.

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration

        Returns:
            (loss, num_samples, metrics) tuple
        """
        # Set global parameters
        self.set_parameters(parameters)

        # Evaluate
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                test_loss += loss.item() * len(features)

                # Collect predictions and targets
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Compute average loss
        test_loss /= len(self.test_loader.dataset)

        # Compute metrics
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        metrics = compute_fraud_metrics(predictions_array, targets_array)

        metrics["loss"] = float(test_loss)

        num_samples = len(self.test_loader.dataset)

        return float(test_loss), num_samples, metrics

    def _evaluate_on_train(self) -> Dict[str, Scalar]:
        """
        Evaluate on training data for metrics collection.

        Returns:
            Dictionary of training metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(features)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        return compute_fraud_metrics(predictions_array, targets_array)


def create_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: DictConfig,
    device: str = "cpu",
) -> FlClient:
    """
    Factory function to create a Flower client.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration
        device: Device to use

    Returns:
        FlClient instance
    """
    return FlClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
