"""
PyTorch Model Definition for Fraud Detection

Implements a flexible MLP architecture for binary fraud classification.
"""

from typing import List

import torch
import torch.nn as nn


class FraudDetectionModel(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection on tabular data.

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

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with fraud probabilities
        """
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
