"""
Top Model for Split Learning in Vertical Federated Learning.

The server holds the top model that receives embeddings from all parties'
bottom models and produces the final prediction.
"""

import torch
import torch.nn as nn
from typing import List, Literal


class TopModel(nn.Module):
    """
    Top model for the server in Vertical Federated Learning.

    Receives concatenated embeddings from all parties' bottom models
    and produces fraud prediction.

    Architecture:
        Concatenated Embeddings -> Hidden Layers -> Classification

    Args:
        embedding_dim_total: Total dimension of concatenated embeddings
        output_dim: Number of output classes (2 for binary classification)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers
        output_activation: Activation for output layer
        dropout: Dropout rate
    """

    def __init__(
        self,
        embedding_dim_total: int,
        output_dim: int = 2,
        hidden_dims: List[int] = None,
        activation: str = 'ReLU',
        output_activation: str = 'Softmax',
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding_dim_total = embedding_dim_total
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [32, 16]

        # Build layers
        layers = []
        prev_dim = embedding_dim_total

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Output activation
        self.output_activation_fn = self._get_output_activation(output_activation)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation layer by name."""
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(0.1),
            'Tanh': nn.Tanh(),
            'ELU': nn.ELU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def _get_output_activation(self, activation: str) -> nn.Module:
        """Get output activation layer by name."""
        if activation == 'Softmax':
            return nn.Softmax(dim=1)
        elif activation == 'Sigmoid':
            return nn.Sigmoid()
        elif activation == 'None':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {activation}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through top model.

        Args:
            embeddings: Concatenated embeddings of shape
                       (batch_size, embedding_dim_total)

        Returns:
            Logits or probabilities of shape (batch_size, output_dim)
        """
        logits = self.network(embeddings)
        return self.output_activation_fn(logits)

    def forward_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without output activation (returns logits).

        Useful for computing loss with nn.CrossEntropyLoss.

        Args:
            embeddings: Concatenated embeddings

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(embeddings)


if __name__ == "__main__":
    # Test top model
    print("Testing Top Model...")

    model = TopModel(
        embedding_dim_total=24,  # 16 (Party A) + 8 (Party B)
        output_dim=2,
        hidden_dims=[32, 16],
        dropout=0.3
    )

    print(f"Top Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    embeddings = torch.randn(64, 24)
    output = model(embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be 1.0): {output[0].sum().item():.6f}")

    print("\n✓ Top model working correctly")
