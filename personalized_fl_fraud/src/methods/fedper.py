"""
FedPer Personalization Method

Method: Shared feature extractor, personalized classification layer.

Pros:
- Minimal communication overhead (only share feature extractor)
- Clear separation between shared knowledge and personalization
- No risk of catastrophic forgetting for shared knowledge

Cons:
- Requires model architecture with clear feature/classifier split
- May underperform if fraud patterns are in feature space
- Limited personalization capacity (only final layer)

Reference:
- "Federated Learning with Personalization Layers" (Arivazhagan et al., ICLR 2020)
"""

from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.server.strategy import FedAvg

from .base import PersonalizationMethod
from ..models.utils import (
    split_model_parameters,
    get_parameters_by_layer_type,
    set_parameters_by_layer_type,
    freeze_model_layers,
    unfreeze_model_layers
)


class FedPer(PersonalizationMethod):
    """
    FedPer: Shared feature extractor, personalized classifier.

    Architecture:
    - Feature extractor: Shared across all clients, aggregated via FedAvg
    - Classifier: Personalized per client, not aggregated

    Process:
    1. Server aggregates only feature extractor parameters
    2. Clients train feature extractor + classifier locally
    3. Classifier remains personalized (not shared)

    Hyperparameters:
    - personal_layers: Layer prefixes to personalize (default: ["classifier"])
    - freeze_feature_extractor: Whether to freeze feature extractor during personalization
    - personal_lr: Learning rate for personalized layers
    - shared_lr: Learning rate for shared layers
    """

    def _get_config_key(self) -> str:
        return "fedper"

    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """Return FedPer client with differential learning rates."""
        from ..clients.wrappers import create_fedper_client
        return create_fedper_client(
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
        Return FedAvg server that only aggregates feature extractor.

        FedPer uses standard FedAvg but only for shared parameters.
        The client handles filtering out personalized parameters.
        """
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def split_model_parameters(
        self,
        model: nn.Module
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Split model parameters into shared and personalizable.

        Args:
            model: FraudDetectionModel

        Returns:
            Tuple of (shared_params, personal_params)
        """
        personal_layers = self.method_config.get('personal_layers', ['classifier'])

        shared_params = []
        personal_params = []

        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in personal_layers):
                personal_params.append(param)
            else:
                shared_params.append(param)

        return shared_params, personal_params

    def create_personalized_classifier(
        self,
        model: nn.Module,
        client_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create personalized classifier for a client.

        Args:
            model: Current model
            client_id: Client identifier

        Returns:
            State dict of personalized classifier
        """
        if not hasattr(model, 'classifier'):
            raise ValueError("Model must have 'classifier' attribute for FedPer")

        # Clone classifier
        classifier_state = deepcopy(model.classifier.state_dict())

        # Add client ID prefix for identification
        personalized_state = {
            f'client_{client_id}.{k}': v
            for k, v in classifier_state.items()
        }

        return personalized_state

    def set_personlized_classifier(
        self,
        model: nn.Module,
        client_classifier: Dict[str, torch.Tensor]
    ) -> None:
        """
        Set personalized classifier for a client.

        Args:
            model: Model to update
            client_classifier: Personalized classifier state dict
        """
        if not hasattr(model, 'classifier'):
            raise ValueError("Model must have 'classifier' attribute for FedPer")

        # Remove client ID prefix
        classifier_state = {}
        for key, value in client_classifier.items():
            # Extract original key (remove "client_X." prefix)
            original_key = '.'.join(key.split('.')[1:])
            classifier_state[original_key] = value

        model.classifier.load_state_dict(classifier_state)

    def freeze_feature_extractor(self, model: nn.Module) -> None:
        """Freeze feature extractor parameters."""
        if hasattr(model, 'feature_extractor'):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            # Fallback: freeze all except classifier
            freeze_model_layers(model, ['fc', 'bn', 'relu', 'dropout'])

    def unfreeze_feature_extractor(self, model: nn.Module) -> None:
        """Unfreeze feature extractor parameters."""
        if hasattr(model, 'feature_extractor'):
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
        else:
            # Fallback: unfreeze all except classifier
            unfreeze_model_layers(model, ['fc', 'bn', 'relu', 'dropout'])

    def get_shared_parameters(
        self,
        model: nn.Module
    ) -> List[np.ndarray]:
        """
        Get shared (feature extractor) parameters.

        Args:
            model: PyTorch model

        Returns:
            List of numpy arrays
        """
        if hasattr(model, 'feature_extractor'):
            return get_parameters_by_layer_type(model, 'feature_extractor')
        else:
            # Fallback: return all except last layer
            all_params = get_parameters_by_layer_type(model, 'all')
            return all_params[:-1]

    def get_personal_parameters(
        self,
        model: nn.Module
    ) -> List[np.ndarray]:
        """
        Get personal (classifier) parameters.

        Args:
            model: PyTorch model

        Returns:
            List of numpy arrays
        """
        if hasattr(model, 'classifier'):
            return get_parameters_by_layer_type(model, 'classifier')
        else:
            # Fallback: return only last layer
            all_params = get_parameters_by_layer_type(model, 'all')
            return [all_params[-1]]

    def set_shared_parameters(
        self,
        model: nn.Module,
        parameters: List[np.ndarray]
    ) -> None:
        """
        Set shared (feature extractor) parameters.

        Args:
            model: PyTorch model
            parameters: List of numpy arrays
        """
        if hasattr(model, 'feature_extractor'):
            set_parameters_by_layer_type(model, parameters, 'feature_extractor')
        else:
            # Fallback: set all except last layer
            # Get current state
            current_state = model.state_dict()
            # Update all but last layer
            # This is complex - better to require proper model architecture
            raise NotImplementedError(
                "FedPer requires model with feature_extractor attribute"
            )

    def get_num_shared_params(self, model: nn.Module) -> int:
        """Count shared parameters."""
        shared_params, _ = self.split_model_parameters(model)
        return sum(p.numel() for p in shared_params if p.requires_grad)

    def get_num_personal_params(self, model: nn.Module) -> int:
        """Count personal parameters."""
        _, personal_params = self.split_model_parameters(model)
        return sum(p.numel() for p in personal_params if p.requires_grad)

    def compute_personalization_benefit(
        self,
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute personalization benefit for FedPer.

        For FedPer, personalization comes from:
        1. Localized classifier (personal layer)
        2. Potentially fine-tuned feature extractor
        """
        return super().compute_personalization_benefit(
            global_metrics,
            personalized_metrics
        )
