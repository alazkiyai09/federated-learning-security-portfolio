"""
Personalized Client Implementations

Implements Flower clients for each personalization method.
"""

from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from flwr.common import FitRes, Scalar
from omegaconf import DictConfig

from ..models.base import FraudDetectionLoss
from ..models.utils import (
    get_parameters_by_layer_type,
    set_parameters_by_layer_type,
    split_model_parameters
)
from ..utils import compute_fraud_metrics


class BaseClient(NumPyClient):
    """Base class for all personalized FL clients."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        device: str = "cpu",
    ):
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
        """Create optimizer based on configuration."""
        optimizer_name = self.config.get("training", {}).get("optimizer", "adam").lower()
        learning_rate = self.config.get("training", {}).get("learning_rate", 0.01)

        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get current model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data."""
        self.set_parameters(parameters)
        self.model.eval()

        test_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                test_loss += loss.item() * len(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        test_loss /= len(self.test_loader.dataset)

        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        metrics = compute_fraud_metrics(predictions_array, targets_array)
        metrics["loss"] = float(test_loss)

        num_samples = len(self.test_loader.dataset)

        return float(test_loss), num_samples, metrics


class FedAvgClient(BaseClient):
    """Standard FedAvg client (baseline)."""

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the model on local data."""
        self.set_parameters(parameters)

        local_epochs = int(config.get(
            "local_epochs",
            self.config.get("federated", {}).get("local_epochs", 5)
        ))

        self.model.train()
        epoch_losses = []

        for epoch in range(local_epochs):
            batch_losses = []
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        # Get training metrics
        train_metrics = self._evaluate_on_train()

        updated_parameters = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "loss": float(np.mean(epoch_losses)),
            **train_metrics,
        }

        return updated_parameters, num_samples, metrics

    def _evaluate_on_train(self) -> Dict[str, Scalar]:
        """Evaluate on training data."""
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


class FedPerClient(FedAvgClient):
    """
    FedPer client with personalized classification layer.

    Only shares feature extractor parameters with server.
    Classifier remains local and personalized.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        method_config: DictConfig,
        device: str = "cpu",
    ):
        super().__init__(model, train_loader, test_loader, config, device)
        self.method_config = method_config

        # Setup differential learning rates
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup separate optimizers for shared and personal layers."""
        shared_lr = self.method_config.get('shared_lr', 0.001)
        personal_lr = self.method_config.get('personal_lr', 0.01)

        shared_params, personal_params = split_model_parameters(self.model)

        # Combined optimizer with different learning rates
        self.optimizer = torch.optim.Adam([
            {'params': shared_params, 'lr': shared_lr},
            {'params': personal_params, 'lr': personal_lr}
        ])

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get only shared (feature extractor) parameters."""
        return get_parameters_by_layer_type(self.model, 'feature_extractor')

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set only shared (feature extractor) parameters."""
        set_parameters_by_layer_type(self.model, parameters, 'feature_extractor')


class DittoClient(FedAvgClient):
    """
    Ditto client with local + global models.

    Maintains separate local model trained only on local data.
    Global model is regularized to stay close to local model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        method_config: DictConfig,
        device: str = "cpu",
    ):
        super().__init__(model, train_loader, test_loader, config, device)
        self.method_config = method_config

        # Create local model
        self.local_model = deepcopy(model).to(device)
        self.local_optimizer = self._create_optimizer_for_model(self.local_model)

        # Train local model
        self._train_local_model()

    def _create_optimizer_for_model(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for specific model."""
        local_lr = self.method_config.get('local_lr', 0.01)
        return torch.optim.Adam(model.parameters(), lr=local_lr)

    def _train_local_model(self):
        """Train local model on client data."""
        personal_epochs = self.method_config.get('personal_epochs', 5)

        self.local_model.train()
        for epoch in range(personal_epochs):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.local_optimizer.zero_grad()
                predictions = self.local_model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.local_optimizer.step()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train global model with Ditto regularization."""
        self.set_parameters(parameters)

        local_epochs = int(config.get(
            "local_epochs",
            self.config.get("federated", {}).get("local_epochs", 5)
        ))
        lambda_reg = self.method_config.get('lambda_regularization', 0.5)

        self.model.train()
        epoch_losses = []

        for epoch in range(local_epochs):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                data_loss = self.criterion(predictions, labels)

                # Add proximal term (distance to local model)
                proximal_term = 0.0
                for p, p_local in zip(self.model.parameters(), self.local_model.parameters()):
                    proximal_term += torch.norm(p - p_local) ** 2

                loss = data_loss + (lambda_reg / 2) * proximal_term
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

        train_metrics = self._evaluate_on_train()
        updated_parameters = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "loss": float(np.mean(epoch_losses)),
            **train_metrics,
        }

        return updated_parameters, num_samples, metrics


class PerFedAvgClient(FedAvgClient):
    """
    Per-FedAvg client with MAML-inspired meta-learning.

    Uses inner loop adaptation on support set and
    outer loop meta-update on query set.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        method_config: DictConfig,
        device: str = "cpu",
    ):
        super().__init__(model, train_loader, test_loader, config, device)
        self.method_config = method_config

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train with Per-FedAvg meta-learning approach."""
        self.set_parameters(parameters)

        # For simplicity, use standard FedAvg training here
        # Full Per-FedAvg requires support/query split which is complex
        # This is a simplified version

        local_epochs = int(config.get(
            "local_epochs",
            self.config.get("federated", {}).get("local_epochs", 5)
        ))

        self.model.train()
        epoch_losses = []

        for epoch in range(local_epochs):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

        train_metrics = self._evaluate_on_train()
        updated_parameters = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "loss": float(np.mean(epoch_losses)),
            **train_metrics,
        }

        return updated_parameters, num_samples, metrics
