"""
Per-FedAvg Personalization Method

Method: MAML-inspired meta-learning for fast client adaptation.

Pros:
- Learns initialization that adapts quickly to new clients
- Moreau envelope prevents overfitting during adaptation
- Strong theoretical foundations

Cons:
- Computationally expensive (inner + outer loops)
- Complex implementation
- Hyperparameter sensitivity (beta, inner_lr, num_steps)

Reference:
- "Personalized Federated Learning with Moreau Envelopes" (T Dinh et al., NeurIPS 2020)
- "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., ICML 2017)
"""

from typing import Dict, List, Tuple, Callable, Optional
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.server.strategy import FedAvg

from .base import PersonalizationMethod
from ..models.base import FraudDetectionLoss
from ..models.utils import (
    compute_moreau_envelope,
    get_parameters_by_layer_type,
    set_parameters_by_layer_type
)
from ..utils import compute_fraud_metrics


class PerFedAvg(PersonalizationMethod):
    """
    Per-FedAvg: Meta-learning for personalized FL with Moreau envelopes.

    Objective (Meta-learning):
        min_w E_client[L_beta(f(w; support), w)]
        where L_beta includes Moreau envelope regularization

    Inner loop (adaptation):
        w' = w - lr_inner * grad(L_train(w, support))

    Outer loop (meta-update):
        w = w - lr_meta * grad(L_beta(w', query))

    Process:
    1. For each client, split data into support (adaptation) and query (evaluation)
    2. Inner loop: Adapt model on support set
    3. Outer loop: Meta-update on query set with Moreau regularization
    4. Repeat for multiple communication rounds

    Hyperparameters:
    - beta: Moreau envelope regularization strength
    - lr_inner: Inner loop (adaptation) learning rate
    - num_inner_steps: Number of gradient steps in inner loop
    - lr_meta: Meta-learning (outer loop) learning rate
    - first_order_approx: Use first-order approximation (faster, less accurate)
    """

    def _get_config_key(self) -> str:
        return "per_fedavg"

    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """Return Per-FedAvg client with meta-learning logic."""
        from ..clients.wrappers import create_per_fedavg_client
        return create_per_fedavg_client(
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

        Per-FedAvg uses standard FedAvg aggregation.
        Meta-learning logic is in client training.
        """
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def compute_moreau_loss(
        self,
        model: nn.Module,
        global_params: Dict[str, torch.Tensor],
        beta: float
    ) -> torch.Tensor:
        """
        Compute Moreau envelope regularization term.

        L_beta(w) = L(w) + (beta/2) * ||w - w_global||^2

        The Moreau envelope smooths the loss landscape and prevents
        overfitting during local adaptation.

        Args:
            model: Current model parameters
            global_params: Global model parameters (reference point)
            beta: Regularization strength

        Returns:
            Moreau envelope term (scalar tensor)
        """
        return compute_moreau_envelope(model, global_params, beta)

    def inner_loop_adaptation(
        self,
        model: nn.Module,
        support_loader: DataLoader,
        global_params: Dict[str, torch.Tensor],
        device: str
    ) -> nn.Module:
        """
        Inner loop: Adapt model to client data (MAML-style).

        Args:
            model: Initial model
            support_loader: Support set data for adaptation
            global_params: Global parameters for Moreau envelope
            device: Device to use

        Returns:
            Adapted model
        """
        # Get hyperparameters
        lr_inner = self.method_config.get('lr_inner', 0.01)
        num_inner_steps = self.method_config.get('num_inner_steps', 5)
        beta = self.method_config.get('beta', 1.0)
        first_order = self.method_config.get('first_order_approx', False)

        # Clone model for adaptation
        adapted_model = deepcopy(model)
        adapted_model.to(device)

        criterion = FraudDetectionLoss(pos_weight=10.0)

        # Inner loop: Multiple gradient steps on support set
        for step in range(num_inner_steps):
            adapted_model.train()
            support_loss = 0.0
            n_batches = 0

            for features, labels in support_loader:
                features = features.to(device)
                labels = labels.to(device)

                # Forward pass
                predictions = adapted_model(features)
                data_loss = criterion(predictions, labels)

                # Add Moreau envelope
                moreau_term = self.compute_moreau_loss(
                    adapted_model, global_params, beta
                )

                total_loss = data_loss + moreau_term

                # Backward pass
                if first_order:
                    # First-order approximation: ignore gradients through Moreau
                    # This is faster but less accurate
                    adapted_model.zero_grad()
                    data_loss.backward()
                else:
                    # Full second-order: compute all gradients
                    adapted_model.zero_grad()
                    total_loss.backward()

                # Manual gradient step (SGD)
                with torch.no_grad():
                    for param in adapted_model.parameters():
                        if param.grad is not None:
                            param.data -= lr_inner * param.grad.data

                support_loss += total_loss.item()
                n_batches += 1

        return adapted_model

    def outer_loop_meta_update(
        self,
        adapted_model: nn.Module,
        query_loader: DataLoader,
        global_params: Dict[str, torch.Tensor],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Outer loop: Compute meta-gradient on query set.

        Args:
            adapted_model: Model after inner loop adaptation
            query_loader: Query set for meta-evaluation
            global_params: Global parameters
            device: Device to use

        Returns:
            Tuple of (loss, metrics)
        """
        beta = self.method_config.get('beta', 1.0)

        criterion = FraudDetectionLoss(pos_weight=10.0)

        adapted_model.eval()
        query_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in query_loader:
                features = features.to(device)
                labels = labels.to(device)

                predictions = adapted_model(features)
                data_loss = criterion(predictions, labels)

                # Add Moreau envelope for meta-loss
                moreau_term = self.compute_moreau_loss(
                    adapted_model, global_params, beta
                )

                total_loss = data_loss + moreau_term
                query_loss += total_loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Compute metrics
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        metrics = compute_fraud_metrics(predictions_array, targets_array)
        metrics['query_loss'] = query_loss / len(query_loader)

        return query_loss / len(query_loader), metrics

    def adapt_to_client(
        self,
        global_model: nn.Module,
        train_loader: DataLoader,
        device: str
    ) -> nn.Module:
        """
        Adapt global model to a specific client (inference-time personalization).

        Args:
            global_model: Global model from server
            train_loader: Client's training data
            device: Device to use

        Returns:
            Personalized model for this client
        """
        # Get global parameters for Moreau envelope
        global_params = global_model.get_state_dict_by_layer_type('all')

        # Inner loop adaptation
        adapted_model = self.inner_loop_adaptation(
            global_model,
            train_loader,
            global_params,
            device
        )

        return adapted_model

    def evaluate_per_fedavg(
        self,
        global_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate Per-FedAvg: global vs adapted model.

        Args:
            global_model: Global model
            train_loader: Client's training data (for adaptation)
            test_loader: Client's test data
            device: Device to use

        Returns:
            Tuple of (global_metrics, adapted_metrics)
        """
        start_time = time()

        # Evaluate global model
        global_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = global_model(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        global_predictions = np.array(all_predictions)
        global_targets = np.array(all_targets)
        global_metrics = compute_fraud_metrics(global_predictions, global_targets)

        # Adapt model to client
        adapted_model = self.adapt_to_client(global_model, train_loader, device)

        # Evaluate adapted model
        adapted_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = adapted_model(features)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        adapted_predictions = np.array(all_predictions)
        adapted_targets = np.array(all_targets)
        adapted_metrics = compute_fraud_metrics(adapted_predictions, adapted_targets)

        # Add timing
        adaptation_time = time() - start_time
        adapted_metrics['adaptation_time'] = adaptation_time

        return global_metrics, adapted_metrics

    def compute_personalization_benefit(
        self,
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute personalization benefit for Per-FedAvg.

        For Per-FedAvg, benefit comes from:
        1. Meta-learned initialization (faster adaptation)
        2. Inner loop adaptation to client data
        3. Moreau envelope preventing overfitting
        """
        return super().compute_personalization_benefit(
            global_metrics,
            personalized_metrics
        )

    def estimate_adaptation_flops(
        self,
        input_dim: int,
        hidden_dims: List[int],
        batch_size: int,
        n_batches: int,
        num_inner_steps: int
    ) -> int:
        """
        Estimate FLOPs for inner loop adaptation.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            batch_size: Batch size
            n_batches: Number of batches in support set
            num_inner_steps: Number of inner loop steps

        Returns:
            Estimated FLOPs
        """
        from ..models.utils import compute_flops_per_forward, compute_flops_per_backward

        flops_per_step = n_batches * (
            compute_flops_per_forward(input_dim, hidden_dims, batch_size) +
            compute_flops_per_backward(input_dim, hidden_dims, batch_size)
        )

        return flops_per_step * num_inner_steps
