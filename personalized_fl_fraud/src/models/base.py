"""
Extended Fraud Detection Model for Personalized Federated Learning

Extends the base FraudDetectionModel to support:
1. Feature extractor + classifier split (for FedPer)
2. Layer-wise parameter access (for personalization)
3. Model cloning (for Ditto local models)
"""

from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import torch
import torch.nn as nn


class FraudDetectionModel(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection with modular architecture.

    Designed to support personalized federated learning methods requiring:
    - Feature extractor / classifier separation (FedPer)
    - Layer-wise freezing/unfreezing
    - Model cloning for local models (Ditto)

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes
        dropout: Dropout probability (default: 0.2)
        batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        self.use_batch_norm = batch_norm

        # Build feature extractor (all layers except final classifier)
        self.feature_extractor = self._build_feature_extractor()

        # Build classifier (final layer)
        self.classifier = self._build_classifier()

        # Combined network for forward pass
        self.network = nn.Sequential(
            self.feature_extractor,
            self.classifier
        )

    def _build_feature_extractor(self) -> nn.Module:
        """
        Build feature extractor network.

        Returns:
            nn.Module: Feature extractor (all layers except output)
        """
        layers = []
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))

            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_classifier(self) -> nn.Module:
        """
        Build classification head.

        Returns:
            nn.Module: Binary classifier with sigmoid activation
        """
        final_dim = self.hidden_dims[-1] if self.hidden_dims else self.input_dim
        return nn.Sequential(
            nn.Linear(final_dim, 1),
            nn.Sigmoid()
        )

    def _get_layer_name(self, prefix: str, value: int) -> str:
        """Helper for layer naming (for debugging)."""
        return f"{prefix}_{value}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through full network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with fraud probabilities
        """
        return self.network(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor only.

        Used in FedPer for separating feature extraction from classification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Feature tensor of shape (batch_size, final_hidden_dim)
        """
        return self.feature_extractor(x)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier only.

        Used in FedPer for personalized classification layer.

        Args:
            features: Feature tensor from feature extractor

        Returns:
            Output tensor of shape (batch_size, 1) with fraud probabilities
        """
        return self.classifier(features)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters_per_layer(self) -> Dict[str, int]:
        """
        Return number of parameters per named layer.

        Returns:
            Dictionary mapping layer names to parameter counts
        """
        return {
            name: p.numel()
            for name, p in self.named_parameters()
            if p.requires_grad
        }

    def freeze_feature_extractor(self) -> None:
        """Freeze all parameters in the feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze all parameters in the feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def freeze_classifier(self) -> None:
        """Freeze all parameters in the classifier."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self) -> None:
        """Unfreeze all parameters in the classifier."""
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_feature_extractor_params(self) -> List[nn.Parameter]:
        """Get list of feature extractor parameters."""
        return list(self.feature_extractor.parameters())

    def get_classifier_params(self) -> List[nn.Parameter]:
        """Get list of classifier parameters."""
        return list(self.classifier.parameters())

    def clone(self) -> 'FraudDetectionModel':
        """
        Create a deep copy of the model.

        Used in Ditto for maintaining separate local and global models.

        Returns:
            Cloned FraudDetectionModel with same parameters
        """
        return deepcopy(self)

    def get_state_dict_by_layer_type(
        self,
        layer_type: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Get state dict filtered by layer type.

        Args:
            layer_type: One of 'all', 'feature_extractor', 'classifier'

        Returns:
            State dict with only specified layer parameters
        """
        full_state = self.state_dict()

        if layer_type == "all":
            return full_state
        elif layer_type == "feature_extractor":
            prefix = "feature_extractor."
            return {
                k: v for k, v in full_state.items()
                if k.startswith(prefix)
            }
        elif layer_type == "classifier":
            prefix = "classifier."
            return {
                k: v for k, v in full_state.items()
                if k.startswith(prefix)
            }
        else:
            raise ValueError(
                f"Unknown layer_type: {layer_type}. "
                "Must be 'all', 'feature_extractor', or 'classifier'"
            )

    def load_state_dict_by_layer_type(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer_type: str = "all",
        strict: bool = False
    ) -> None:
        """
        Load state dict for specific layer type only.

        Args:
            state_dict: State dictionary to load
            layer_type: One of 'all', 'feature_extractor', 'classifier'
            strict: Whether to strictly enforce that all keys match
        """
        current_state = self.state_dict()

        if layer_type == "all":
            self.load_state_dict(state_dict, strict=strict)
        elif layer_type == "feature_extractor":
            prefix = "feature_extractor."
            filtered_state = {
                k: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            current_state.update(filtered_state)
            self.load_state_dict(current_state, strict=False)
        elif layer_type == "classifier":
            prefix = "classifier."
            filtered_state = {
                k: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            current_state.update(filtered_state)
            self.load_state_dict(current_state, strict=False)
        else:
            raise ValueError(
                f"Unknown layer_type: {layer_type}. "
                "Must be 'all', 'feature_extractor', or 'classifier'"
            )


class FraudDetectionLoss(nn.Module):
    """
    Combined loss function for imbalanced fraud detection.

    Combines binary cross-entropy with class weighting.
    """

    def __init__(self, pos_weight: float = 10.0) -> None:
        """
        Args:
            pos_weight: Weight for positive class (fraud) to handle imbalance
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy loss.

        Args:
            predictions: Predicted probabilities (batch_size, 1)
            targets: Ground truth labels (batch_size, 1)

        Returns:
            Scalar loss value
        """
        # Apply class weighting
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight

        # Compute weighted BCE loss
        loss = self.bce(predictions, targets)
        weighted_loss = (loss * weights).mean()

        return weighted_loss


def create_model(
    input_dim: int,
    hidden_dims: List[int],
    device: str = "cpu",
) -> FraudDetectionModel:
    """
    Factory function to create and initialize a FraudDetectionModel.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes
        device: Device to place model on

    Returns:
        Initialized model on specified device
    """
    model = FraudDetectionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
    )
    return model.to(device)
