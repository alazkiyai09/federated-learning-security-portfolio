"""
Ditto Personalization Method

Method: Maintain local + global models with proximal regularization.

Pros:
- Explicitly models local vs global objectives
- Regularization prevents catastrophic forgetting
- Strong empirical performance on non-IID data

Cons:
- Doubles memory requirement (two models per client)
- More complex training logic
- Hyperparameter sensitivity (lambda)

Reference:
- "Ditto: Fair and Robust Federated Learning" (Li et al., ICLR 2022)
- https://arxiv.org/abs/2012.04221
"""

from typing import Dict, List, Tuple
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.server.strategy import FedAvg

from .base import PersonalizationMethod, PersonalizationResult
from ..models.base import FraudDetectionLoss
from ..models.utils import compute_param_distance
from ..utils import compute_fraud_metrics


class Ditto(PersonalizationMethod):
    """
    Ditto: Local + global models with proximal regularization.

    Objective:
        L(w) = L_global(w) + lambda * ||w - w_local||^2

    Where:
    - L_global(w): Standard FL loss on global model
    - w_local: Local model parameters trained on local data only
    - lambda: Regularization strength

    Process:
    1. Each client maintains TWO models: global and local
    2. Global model participates in FedAvg
    3. Local model trains only on local data
    4. Global model is regularized to stay close to local model
    5. Final model: Blend of global and local

    Hyperparameters:
    - lambda_regularization: Strength of local model regularization
    - personal_epochs: Local model training epochs
    - local_lr: Learning rate for local model
    - global_lr: Learning rate for global model updates
    """

    def _get_config_key(self) -> str:
        return "ditto"

    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """Return Ditto client with dual-model training."""
        from ..clients.wrappers import create_ditto_client
        return create_ditto_client(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=self.config,
            method_config=self.method_config,
            device=device
        )

    def get_server_strategy(
        self,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int
    ):
        """
        Return standard FedAvg server.

        Ditto uses standard FedAvg on server side.
        Client handles local model and regularization.
        """
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def compute_ditto_loss(
        self,
        model: nn.Module,
        local_model: nn.Module,
        lambda_reg: float,
        data_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Ditto loss with local model regularization.

        L_ditto(w) = L_data(w) + (lambda/2) * ||w - w_local||^2

        Args:
            model: Global model
            local_model: Local model
            lambda_reg: Regularization strength
            data_loss: Data fitting loss (e.g., BCE)

        Returns:
            Total Ditto loss
        """
        # Compute proximal term
        proximal_term = compute_param_distance(
            model, local_model, layer_type="all"
        )

        # Total loss
        total_loss = data_loss + (lambda_reg / 2) * proximal_term

        return total_loss

    def train_local_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: str
    ) -> nn.Module:
        """
        Train local model on client data (no regularization).

        Args:
            model: Base model architecture
            train_loader: Client's training data
            device: Device to use

        Returns:
            Trained local model
        """
        local_lr = self.method_config.get('local_lr', 0.01)
        personal_epochs = self.method_config.get('personal_epochs', 5)

        # Clone model for local training
        local_model = deepcopy(model)
        local_model.to(device)

        criterion = FraudDetectionLoss(pos_weight=10.0)
        optimizer = optim.Adam(local_model.parameters(), lr=local_lr)

        # Train local model
        local_model.train()
        for epoch in range(personal_epochs):
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                predictions = local_model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

        return local_model

    def train_global_with_regularization(
        self,
        model: nn.Module,
        local_model: nn.Module,
        train_loader: DataLoader,
        device: str
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train global model with Ditto regularization.

        Args:
            model: Global model to train
            local_model: Local model for regularization
            train_loader: Training data
            device: Device to use

        Returns:
            Tuple of (trained_model, metrics)
        """
        start_time = time()

        lambda_reg = self.method_config.get('lambda_regularization', 0.5)
        global_lr = self.method_config.get('global_lr', 0.001)
        local_epochs = self.config.get('federated', {}).get('local_epochs', 5)

        criterion = FraudDetectionLoss(pos_weight=10.0)
        optimizer = optim.Adam(model.parameters(), lr=global_lr)

        model.train()
        all_losses = []

        for epoch in range(local_epochs):
            epoch_losses = []

            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                data_loss = criterion(predictions, labels)

                # Add Ditto regularization
                total_loss = self.compute_ditto_loss(
                    model, local_model, lambda_reg, data_loss
                )

                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())

            all_losses.extend(epoch_losses)

        training_time = time() - start_time
        avg_loss = np.mean(all_losses)

        metrics = {
            'ditto_loss': avg_loss,
            'training_time': training_time
        }

        return model, metrics

    def create_personalized_model(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
        alpha: float = 0.5
    ) -> nn.Module:
        """
        Create personalized model by blending global and local.

        w_personalized = alpha * w_global + (1 - alpha) * w_local

        Args:
            global_model: Global model
            local_model: Local model
            alpha: Blending factor (0.5 = equal weight)

        Returns:
            Personalized model
        """
        personalized = deepcopy(global_model)

        # Blend parameters
        with torch.no_grad():
            for (name_p, param_p), (name_l, param_l) in zip(
                global_model.named_parameters(),
                local_model.named_parameters()
            ):
                if name_p == name_l:
                    param_p.data = (
                        alpha * param_p.data +
                        (1 - alpha) * param_l.data
                    )

        return personalized

    def evaluate_ditto_model(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
        test_loader: DataLoader,
        device: str,
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate Ditto personalized model.

        Args:
            global_model: Global model
            local_model: Local model
            test_loader: Test data
            device: Device to use
            alpha: Blending factor

        Returns:
            Metrics dictionary
        """
        # Create personalized model
        personalized_model = self.create_personalized_model(
            global_model, local_model, alpha
        )

        # Evaluate
        personalized_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)

                predictions = personalized_model(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)

        return compute_fraud_metrics(predictions_array, targets_array)

    def compute_personalization_benefit(
        self,
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute personalization benefit for Ditto.

        For Ditto, benefit comes from:
        1. Local model capturing client-specific patterns
        2. Global model maintaining general knowledge
        3. Blending both for optimal performance
        """
        return super().compute_personalization_benefit(
            global_metrics,
            personalized_metrics
        )
