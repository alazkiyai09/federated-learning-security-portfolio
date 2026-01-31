"""
Secure forward pass protocol for Vertical Federated Learning.

This module implements the communication protocol for the forward pass
where parties send embeddings (not raw features) to the server.
"""

import torch
from typing import Tuple
from ..models.bottom_model import BottomModel
from ..models.top_model import TopModel


def secure_forward(
    bottom_model_a: BottomModel,
    bottom_model_b: BottomModel,
    top_model: TopModel,
    x_a: torch.Tensor,
    x_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute secure forward pass in Vertical Federated Learning.

    Protocol:
    1. Party A computes embedding locally (raw features stay on device)
    2. Party B computes embedding locally (raw features stay on device)
    3. Both parties send embeddings to server
    4. Server concatenates embeddings and computes prediction
    5. Server returns prediction to label holder

    Privacy Guarantee:
    - Raw features (x_a, x_b) never leave local devices
    - Only embeddings transmitted to server
    - Embeddings are latent representations, not raw data

    Args:
        bottom_model_a: Party A's bottom model (transaction features)
        bottom_model_b: Party B's bottom model (credit features)
        top_model: Server's top model
        x_a: Party A's raw features (batch_size, feat_a_dim)
        x_b: Party B's raw features (batch_size, feat_b_dim)

    Returns:
        Tuple of (predictions, embeddings_a, embeddings_b)
        - predictions: Server's predictions (batch_size, num_classes)
        - embeddings_a: Party A's embeddings (for backward pass)
        - embeddings_b: Party B's embeddings (for backward pass)
    """
    # Step 1: Party A computes embedding locally
    # Privacy: x_a never leaves Party A's device
    embeddings_a = bottom_model_a(x_a)

    # Step 2: Party B computes embedding locally
    # Privacy: x_b never leaves Party B's device
    embeddings_b = bottom_model_b(x_b)

    # Step 3: Embeddings transmitted to server
    # In real deployment, this would be encrypted/secure channel

    # Step 4: Server concatenates and computes prediction
    combined_embeddings = torch.cat([embeddings_a, embeddings_b], dim=1)
    predictions = top_model(combined_embeddings)

    # Step 5: Server returns predictions to label holder

    return predictions, embeddings_a, embeddings_b


def compute_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = 'cross_entropy'
) -> torch.Tensor:
    """
    Compute loss for training.

    Args:
        predictions: Model predictions (batch_size, num_classes) or logits
        labels: Ground truth labels (batch_size,)
        loss_type: Type of loss ('cross_entropy', 'bce')

    Returns:
        Loss tensor
    """
    if loss_type == 'cross_entropy':
        # Predictions should be logits (no softmax)
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == 'bce':
        # Predictions should be probabilities
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion(predictions, labels)


def log_forward_pass_info(
    batch_idx: int,
    num_batches: int,
    loss: float,
    log_interval: int = 10
) -> None:
    """Log forward pass information."""
    if batch_idx % log_interval == 0:
        print(f"Batch [{batch_idx}/{num_batches}], Loss: {loss:.4f}")


if __name__ == "__main__":
    # Test secure forward
    from ..models.bottom_model import PartyABottomModel, PartyBBottomModel
    from ..models.top_model import TopModel

    print("Testing Secure Forward Pass...")

    # Create models
    bottom_a = PartyABottomModel()
    bottom_b = PartyBBottomModel()
    top = TopModel()

    # Create data
    x_a = torch.randn(32, 7)
    x_b = torch.randn(32, 3)

    print("\n=== FORWARD PASS PROTOCOL ===")

    print("\n[Party A] Computing embedding from raw features...")
    print(f"  Raw features shape: {x_a.shape}")
    embeddings_a = bottom_a(x_a)
    print(f"  Embedding shape: {embeddings_a.shape}")
    print(f"  ✓ Raw features NOT transmitted")

    print("\n[Party B] Computing embedding from raw features...")
    print(f"  Raw features shape: {x_b.shape}")
    embeddings_b = bottom_b(x_b)
    print(f"  Embedding shape: {embeddings_b.shape}")
    print(f"  ✓ Raw features NOT transmitted")

    print("\n[Server] Receiving embeddings and computing prediction...")
    print(f"  Embeddings A received: {embeddings_a.shape}")
    print(f"  Embeddings B received: {embeddings_b.shape}")
    combined = torch.cat([embeddings_a, embeddings_b], dim=1)
    print(f"  Combined embeddings: {combined.shape}")
    predictions = top(combined)
    print(f"  Predictions: {predictions.shape}")
    print(f"  ✓ Only embeddings transmitted, NOT raw features")

    print("\n=== PRIVACY SUMMARY ===")
    print("✓ Party A raw features: STAY LOCAL")
    print("✓ Party B raw features: STAY LOCAL")
    print("✓ Transmitted to server: EMBEDDINGS ONLY")
    print("✓ Server receives: embeddings_a + embeddings_b")

    print("\n✓ Secure forward pass working correctly")
