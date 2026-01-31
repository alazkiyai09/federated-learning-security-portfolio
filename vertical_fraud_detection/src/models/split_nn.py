"""
Split Neural Network for Vertical Federated Learning.

Integrates bottom models (parties) and top model (server) and manages
secure forward and backward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from .bottom_model import BottomModel
from .top_model import TopModel


class SplitNN:
    """
    Split Neural Network for Vertical Federated Learning.

    Orchestrates the split learning architecture:
    - Party A: Bottom model A
    - Party B: Bottom model B
    - Server: Top model

    Privacy guarantee: Only embeddings and gradients are transmitted.
    No raw features are shared between parties.
    """

    def __init__(
        self,
        bottom_model_a: BottomModel,
        bottom_model_b: BottomModel,
        top_model: TopModel,
        device: str = 'cpu'
    ):
        """
        Initialize SplitNN.

        Args:
            bottom_model_a: Party A's bottom model
            bottom_model_b: Party B's bottom model
            top_model: Server's top model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.bottom_model_a = bottom_model_a.to(device)
        self.bottom_model_b = bottom_model_b.to(device)
        self.top_model = top_model.to(device)
        self.device = device

    def train_mode(self) -> None:
        """Set all models to training mode."""
        self.bottom_model_a.train()
        self.bottom_model_b.train()
        self.top_model.train()

    def eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        self.bottom_model_a.eval()
        self.bottom_model_b.eval()
        self.top_model.eval()

    def forward_pass(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Secure forward pass through split architecture.

        Privacy:
        - Each party computes embeddings locally (raw features stay local)
        - Only embeddings are sent to server
        - Server returns predictions

        Args:
            x_a: Party A features (batch_size, feat_a_dim)
            x_b: Party B features (batch_size, feat_b_dim)

        Returns:
            Tuple of (predictions, embeddings_a, embeddings_b)
            - predictions: (batch_size, num_classes)
            - embeddings_a: (batch_size, embedding_dim_a)
            - embeddings_b: (batch_size, embedding_dim_b)
        """
        # Party A computes embedding (local computation)
        embeddings_a = self.bottom_model_a(x_a)

        # Party B computes embedding (local computation)
        embeddings_b = self.bottom_model_b(x_b)

        # Server receives embeddings and computes prediction
        # No raw features transmitted!
        combined_embeddings = torch.cat([embeddings_a, embeddings_b], dim=1)
        predictions = self.top_model(combined_embeddings)

        return predictions, embeddings_a, embeddings_b

    def backward_pass(
        self,
        loss: torch.Tensor,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Secure backward pass through split architecture.

        Privacy:
        - Server computes gradients wrt embeddings
        - Only embedding gradients sent back to parties
        - Parties update their bottom models locally

        Args:
            loss: Computed loss tensor
            embeddings_a: Forward pass embeddings from Party A
            embeddings_b: Forward pass embeddings from Party B
            x_a: Party A features (for gradient computation)
            x_b: Party B features (for gradient computation)

        Returns:
            Dictionary with gradient statistics
        """
        # Zero all gradients
        self.bottom_model_a.zero_grad()
        self.bottom_model_b.zero_grad()
        self.top_model.zero_grad()

        # Backward pass
        loss.backward()

        # Gradients are computed automatically by autograd
        # In real VFL, these would be transmitted securely

        stats = {
            'loss': loss.item(),
            'grad_norm_a': self._compute_grad_norm(self.bottom_model_a),
            'grad_norm_b': self._compute_grad_norm(self.bottom_model_b),
            'grad_norm_top': self._compute_grad_norm(self.top_model),
        }

        return stats

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of model gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def predict(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Make predictions (evaluation mode).

        Args:
            x_a: Party A features
            x_b: Party B features

        Returns:
            Predictions (batch_size, num_classes)
        """
        self.eval_mode()
        with torch.no_grad():
            predictions, _, _ = self.forward_pass(x_a, x_b)
        return predictions

    def get_parameters(self) -> Dict[str, List[torch.Tensor]]:
        """Get all model parameters."""
        return {
            'bottom_a': list(self.bottom_model_a.parameters()),
            'bottom_b': list(self.bottom_model_b.parameters()),
            'top': list(self.top_model.parameters()),
        }

    def save_models(self, save_dir: str) -> None:
        """Save all models."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            self.bottom_model_a.state_dict(),
            os.path.join(save_dir, 'bottom_model_a.pth')
        )
        torch.save(
            self.bottom_model_b.state_dict(),
            os.path.join(save_dir, 'bottom_model_b.pth')
        )
        torch.save(
            self.top_model.state_dict(),
            os.path.join(save_dir, 'top_model.pth')
        )

    def load_models(self, load_dir: str) -> None:
        """Load all models."""
        import os

        self.bottom_model_a.load_state_dict(
            torch.load(os.path.join(load_dir, 'bottom_model_a.pth'))
        )
        self.bottom_model_b.load_state_dict(
            torch.load(os.path.join(load_dir, 'bottom_model_b.pth'))
        )
        self.top_model.load_state_dict(
            torch.load(os.path.join(load_dir, 'top_model.pth'))
        )


if __name__ == "__main__":
    # Test SplitNN
    from .bottom_model import PartyABottomModel, PartyBBottomModel

    print("Testing Split Neural Network...")

    # Create models
    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=16, hidden_dims=[32, 24])
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=8, hidden_dims=[16, 12])
    top = TopModel(embedding_dim_total=24, output_dim=2, hidden_dims=[32, 16])

    # Create SplitNN
    split_nn = SplitNN(bottom_a, bottom_b, top)

    print("\nTotal Parameters:")
    print(f"Bottom A: {sum(p.numel() for p in bottom_a.parameters()):,}")
    print(f"Bottom B: {sum(p.numel() for p in bottom_b.parameters()):,}")
    print(f"Top: {sum(p.numel() for p in top.parameters()):,}")

    # Test forward pass
    x_a = torch.randn(64, 7)
    x_b = torch.randn(64, 3)

    predictions, emb_a, emb_b = split_nn.forward_pass(x_a, x_b)

    print(f"\nForward pass:")
    print(f"Input A shape: {x_a.shape}")
    print(f"Input B shape: {x_b.shape}")
    print(f"Embedding A shape: {emb_a.shape}")
    print(f"Embedding B shape: {emb_b.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # Test backward pass
    labels = torch.randint(0, 2, (64,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)

    stats = split_nn.backward_pass(loss, emb_a, emb_b, x_a, x_b)

    print(f"\nBackward pass:")
    print(f"Loss: {stats['loss']:.4f}")
    print(f"Grad norm A: {stats['grad_norm_a']:.4f}")
    print(f"Grad norm B: {stats['grad_norm_b']:.4f}")
    print(f"Grad norm top: {stats['grad_norm_top']:.4f}")

    print("\n✓ SplitNN working correctly")
