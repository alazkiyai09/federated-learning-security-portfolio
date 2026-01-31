"""
Neural network architecture for fraud detection.
Uses embedding layers for categorical features and dense layers for numerical.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class FraudNN(nn.Module):
    """
    Neural network for fraud detection.

    Architecture:
    - Embedding layers for categorical features (merchant, region, etc.)
    - Dense layers for processed numerical features
    - Combined feature representation
    - Output layer with sigmoid activation
    """

    def __init__(
        self,
        numerical_features: int,
        categorical_features: Dict[str, int],  # feature_name -> cardinality
        embedding_dims: Dict[str, int],  # feature_name -> embedding_dim
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize fraud detection neural network.

        Args:
            numerical_features: Number of numerical input features
            categorical_features: Dictionary mapping feature names to cardinality
            embedding_dims: Dictionary mapping feature names to embedding dimensions
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(FraudNN, self).__init__()

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.hidden_layers = hidden_layers

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feature, cardinality in categorical_features.items():
            embed_dim = embedding_dims.get(feature, min(8, (cardinality + 1) // 2))
            self.embeddings[feature] = nn.Embedding(cardinality + 1, embed_dim)  # +1 for unseen
            total_embedding_dim += embed_dim

        # Calculate input dimension for first hidden layer
        input_dim = numerical_features + total_embedding_dim

        # Create hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def forward(
        self,
        x_numerical: torch.Tensor,
        x_categorical: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_numerical: Tensor of shape (batch_size, numerical_features)
            x_categorical: Optional dict of feature_name -> tensor of shape (batch_size,)

        Returns:
            Tensor of shape (batch_size, 1) with logits
        """
        batch_size = x_numerical.size(0)

        # Process categorical features through embeddings
        if x_categorical is not None and len(self.embeddings) > 0:
            embeddings_list = []

            for feature_name, embedding_layer in self.embeddings.items():
                if feature_name in x_categorical:
                    # Get embedding for this feature
                    embedded = embedding_layer(x_categorical[feature_name])
                    embeddings_list.append(embedded)

            # Concatenate all embeddings
            if embeddings_list:
                x_embedded = torch.cat(embeddings_list, dim=1)
            else:
                x_embedded = torch.zeros(batch_size, 0, device=x_numerical.device)
        else:
            x_embedded = torch.zeros(batch_size, 0, device=x_numerical.device)

        # Concatenate numerical and embedded features
        x_combined = torch.cat([x_numerical, x_embedded], dim=1)

        # Pass through hidden layers
        x_hidden = self.hidden(x_combined)

        # Output layer (logits, not probabilities)
        logits = self.output(x_hidden)

        return logits

    def predict_proba(
        self,
        x_numerical: torch.Tensor,
        x_categorical: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x_numerical: Tensor of shape (batch_size, numerical_features)
            x_categorical: Optional dict of categorical tensors

        Returns:
            Tensor of shape (batch_size, 1) with probabilities
        """
        logits = self.forward(x_numerical, x_categorical)
        return torch.sigmoid(logits)


class SimplifiedFraudNN(nn.Module):
    """
    Simplified neural network for fraud detection.

    Uses only numerical features (simpler data preparation).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize simplified fraud detection network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(SimplifiedFraudNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers

        # Create layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_hidden = self.network(x)
        logits = self.output(x_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def create_model(
    input_dim: int,
    hidden_layers: List[int] = None,
    dropout: float = 0.3,
    simplified: bool = True
) -> nn.Module:
    """
    Create a fraud detection model.

    Args:
        input_dim: Number of input features
        hidden_layers: Hidden layer sizes
        dropout: Dropout probability
        simplified: If True, use SimplifiedFraudNN

    Returns:
        Initialized model
    """
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]

    if simplified:
        model = SimplifiedFraudNN(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
    else:
        # Would need categorical feature info for full model
        raise NotImplementedError("Full model requires categorical feature configuration")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
