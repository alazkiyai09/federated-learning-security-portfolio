"""
Local Fine-Tuning Personalization Method

Method: Train global model via FedAvg, then fine-tune locally on each client.

Pros:
- Simple to implement and understand
- No changes to server-side aggregation
- Works with any model architecture

Cons:
- Risk of overfitting to small local datasets
- No personalization during federated training phase
- May forget global knowledge (catastrophic forgetting)

Reference:
- Standard transfer learning approach applied to FL
"""

from typing import Dict, List, Tuple
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .base import PersonalizationMethod, PersonalizationResult
from ..models.base import FraudDetectionLoss
from ..models.utils import freeze_model_layers, unfreeze_model_layers
from ..utils import compute_fraud_metrics


class LocalFineTuning(PersonalizationMethod):
    """
    Local Fine-Tuning: Train global model, then fine-tune locally.

    Process:
    1. Run standard FedAvg to get global model
    2. Each client fine-tunes global model on local data
    3. Track performance improvement

    Hyperparameters:
    - finetuning_epochs: Number of local fine-tuning epochs
    - finetuning_lr: Learning rate for fine-tuning
    - freeze_layers: List of layer prefixes to freeze during fine-tuning
    """

    def _get_config_key(self) -> str:
        return "local_finetuning"

    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """Return standard FedAvg client (fine-tuning is post-hoc)."""
        from ..clients.wrappers import create_fedavg_client
        return create_fedavg_client(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=self.config,
            device=device
        )

    def get_server_strategy(
        self,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int
    ):
        """Return standard FedAvg server strategy."""
        from flwr.server.strategy import FedAvg
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def fine_tune_client_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ) -> PersonalizationResult:
        """
        Fine-tune global model on client data.

        Args:
            model: Global model to fine-tune
            train_loader: Client's training data
            test_loader: Client's test data
            device: Device to use

        Returns:
            PersonalizationResult with metrics and compute info
        """
        start_time = time()

        # Get config
        finetuning_epochs = self.method_config.get('finetuning_epochs', 10)
        finetuning_lr = self.method_config.get('finetuning_lr', 0.001)
        freeze_layers = self.method_config.get('freeze_layers', [])

        # Evaluate global model before fine-tuning
        global_metrics = self._evaluate_model(model, test_loader, device)

        # Freeze specified layers
        if freeze_layers:
            freeze_model_layers(model, freeze_layers)

        # Setup fine-tuning
        model.train()
        criterion = FraudDetectionLoss(pos_weight=10.0)
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=finetuning_lr
        )

        # Fine-tuning loop
        flops = 0
        for epoch in range(finetuning_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Estimate FLOPs
                batch_size = len(features)
                input_dim = features.shape[1]
                hidden_dims = self.config.get('model', {}).get('hidden_dims', [64, 32, 16])
                flops += self._estimate_flops(input_dim, hidden_dims, batch_size) * 2

        # Unfreeze layers
        if freeze_layers:
            unfreeze_model_layers(model, freeze_layers)

        # Evaluate fine-tuned model
        personalized_metrics = self._evaluate_model(model, test_loader, device)

        # Compute personalization benefit
        personalization_delta = self.compute_personalization_benefit(
            global_metrics,
            personalized_metrics
        )

        training_time = time() - start_time

        # No additional communication cost (fine-tuning is local)
        communication_cost = 0

        return PersonalizationResult(
            global_metrics=global_metrics,
            personalized_metrics=personalized_metrics,
            personalization_delta=personalization_delta,
            training_time=training_time,
            flops=flops,
            communication_cost=communication_cost
        )

    def _evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str
    ) -> Dict[str, float]:
        """Evaluate model and compute metrics."""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)

                predictions = model(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)

        return compute_fraud_metrics(predictions_array, targets_array)

    def _estimate_flops(
        self,
        input_dim: int,
        hidden_dims: List[int],
        batch_size: int
    ) -> int:
        """Estimate FLOs for forward pass."""
        from ..models.utils import compute_flops_per_forward
        return compute_flops_per_forward(input_dim, hidden_dims, batch_size)
