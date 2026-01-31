"""
Neural network architectures for federated learning experiments.

Includes:
- SimpleCNN: For MNIST sanity checks
- MLP: For fraud detection on tabular data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.

    Architecture:
    - Conv2d(1->32) -> ReLU -> MaxPool
    - Conv2d(32->64) -> ReLU -> MaxPool
    - Flatten -> Linear(1600->128) -> ReLU -> Dropout
    - Linear(128->10)

    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two pooling layers: 28 -> 14 -> 7
        # 64 channels * 7 * 7 = 3136, but let's compute properly
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7, 7)

        x = x.view(x.size(0), -1)  # (B, 64*7*7) = (B, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection on tabular data.

    Suitable for datasets like Credit Card Fraud Detection with 30 features.

    Architecture:
    - Input(30) -> Linear(64) -> ReLU -> BatchNorm -> Dropout
    - Linear(64) -> ReLU -> BatchNorm -> Dropout
    - Linear(64) -> ReLU -> BatchNorm -> Dropout
    - Linear(1) -> Sigmoid

    Args:
        input_dim: Number of input features (default: 30)
        hidden_dim: Hidden layer dimension (default: 64)
        num_layers: Number of hidden layers (default: 3)
        dropout: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Build hidden layers
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer for binary classification
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Probabilities of shape (batch_size, 1)
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models by type.

    Args:
        model_type: Type of model ('cnn' or 'mlp')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        nn.Module: Instantiated model
    """
    if model_type.lower() == 'cnn':
        return SimpleCNN(**kwargs)
    elif model_type.lower() == 'mlp':
        return MLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
