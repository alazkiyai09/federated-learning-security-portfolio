"""
Bottom Model for Split Learning in Vertical Federated Learning.

Each party (Party A and Party B) has a bottom model that maps their
raw features to an embedding space. These embeddings are sent to
the server's top model for final prediction.
"""

import torch
import torch.nn as nn
from typing import List, Literal


class BottomModel(nn.Module):
    """
    Bottom model for a party in Vertical Federated Learning.

    Maps raw features to an embedding representation that is shared
    with the server's top model. No raw features are transmitted.

    Architecture:
        Input -> Hidden Layers -> Embedding

    Args:
        input_dim: Number of input features
        embedding_dim: Dimension of output embedding
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('ReLU', 'LeakyReLU', 'Tanh')
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: List[int] = None,
        activation: str = 'ReLU',
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = []

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation layer by name."""
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(0.1),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'ELU': nn.ELU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottom model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.network(x)

    def get_embedding_dim(self) -> int:
        """Return output embedding dimension."""
        return self.embedding_dim


class PartyABottomModel(BottomModel):
    """Bottom model for Party A (transaction features)."""

    def __init__(
        self,
        input_dim: int = 7,
        embedding_dim: int = 16,
        hidden_dims: List[int] = None,
        activation: str = 'ReLU',
        dropout: float = 0.2
    ):
        super().__init__(input_dim, embedding_dim, hidden_dims, activation, dropout)


class PartyBBottomModel(BottomModel):
    """Bottom model for Party B (credit features)."""

    def __init__(
        self,
        input_dim: int = 3,
        embedding_dim: int = 8,
        hidden_dims: List[int] = None,
        activation: str = 'ReLU',
        dropout: float = 0.2
    ):
        super().__init__(input_dim, embedding_dim, hidden_dims, activation, dropout)


if __name__ == "__main__":
    # Test bottom models
    print("Testing Bottom Models...")

    # Party A model
    model_a = PartyABottomModel(
        input_dim=7,
        embedding_dim=16,
        hidden_dims=[32, 24]
    )
    print(f"\nParty A Model:")
    print(f"Parameters: {sum(p.numel() for p in model_a.parameters()):,}")

    x_a = torch.randn(64, 7)
    emb_a = model_a(x_a)
    print(f"Input shape: {x_a.shape}")
    print(f"Embedding shape: {emb_a.shape}")

    # Party B model
    model_b = PartyBBottomModel(
        input_dim=3,
        embedding_dim=8,
        hidden_dims=[16, 12]
    )
    print(f"\nParty B Model:")
    print(f"Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    x_b = torch.randn(64, 3)
    emb_b = model_b(x_b)
    print(f"Input shape: {x_b.shape}")
    print(f"Embedding shape: {emb_b.shape}")

    print("\n✓ Bottom models working correctly")
