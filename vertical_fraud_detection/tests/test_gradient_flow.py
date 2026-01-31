"""
Unit tests for gradient flow correctness in Vertical FL.

Verifies that gradients are computed correctly through the split
architecture using numerical gradient checking.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from models.bottom_model import PartyABottomModel, PartyBBottomModel
from models.top_model import TopModel
from models.split_nn import SplitNN
from training.forward_pass import secure_forward
from training.backward_pass import secure_backward


def test_forward_pass_shapes():
    """Test that forward pass produces correct output shapes."""
    print("\n=== Testing Forward Pass Shapes ===")

    # Create models
    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    # Create data
    batch_size = 32
    x_a = torch.randn(batch_size, 7)
    x_b = torch.randn(batch_size, 3)

    # Forward pass
    predictions, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)

    # Check shapes
    assert emb_a.shape == (batch_size, 8), f"Expected emb_a shape {(batch_size, 8)}, got {emb_a.shape}"
    assert emb_b.shape == (batch_size, 4), f"Expected emb_b shape {(batch_size, 4)}, got {emb_b.shape}"
    assert predictions.shape == (batch_size, 2), f"Expected predictions shape {(batch_size, 2)}, got {predictions.shape}"

    print("✓ Forward pass shapes correct")
    return True


def test_backward_pass_gradients_exist():
    """Test that backward pass computes gradients for all parameters."""
    print("\n=== Testing Backward Pass Gradient Computation ===")

    # Create models
    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    # Create data
    batch_size = 16
    x_a = torch.randn(batch_size, 7)
    x_b = torch.randn(batch_size, 3)
    labels = torch.randint(0, 2, (batch_size,))

    # Forward pass
    predictions, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    logits = top.forward_logits(torch.cat([emb_a, emb_b], dim=1))
    loss = criterion(logits, labels)

    # Backward pass
    stats = secure_backward(top, emb_a, emb_b, loss, bottom_a, bottom_b, x_a, x_b)

    # Check that all parameters have gradients
    for name, param in bottom_a.named_parameters():
        assert param.grad is not None, f"Bottom A parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Bottom A parameter {name} has NaN gradient"

    for name, param in bottom_b.named_parameters():
        assert param.grad is not None, f"Bottom B parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Bottom B parameter {name} has NaN gradient"

    for name, param in top.named_parameters():
        assert param.grad is not None, f"Top model parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Top model parameter {name} has NaN gradient"

    print("✓ All parameters have gradients")
    print(f"  Grad norm A: {stats['grad_norm_a']:.4f}")
    print(f"  Grad norm B: {stats['grad_norm_b']:.4f}")
    print(f"  Grad norm top: {stats['grad_norm_top']:.4f}")

    return True


def numerical_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    loss_fn: callable,
    epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Compute numerical gradient using finite differences.

    Args:
        model: Model to compute gradient for
        input_tensor: Input tensor
        loss_fn: Function that takes input and returns loss
        epsilon: Perturbation size

    Returns:
        Numerical gradient with same shape as input_tensor
    """
    numerical_grad = torch.zeros_like(input_tensor)

    for i in range(input_tensor.numel()):
        # Create perturbation
        perturbation = torch.zeros_like(input_tensor)
        perturbation.view(-1)[i] = epsilon

        # Compute loss with +epsilon
        loss_plus = loss_fn(input_tensor + perturbation)

        # Compute loss with -epsilon
        loss_minus = loss_fn(input_tensor - perturbation)

        # Finite difference
        numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return numerical_grad


def test_gradient_correctness():
    """
    Test gradient correctness using numerical gradient checking.

    Verifies: dL/dθ = dL/dz × dz/dθ via chain rule
    """
    print("\n=== Testing Gradient Correctness (Numerical) ===")

    # Create small models for faster numerical gradient computation
    bottom_a = PartyABottomModel(input_dim=3, embedding_dim=4, hidden_dims=[8])
    bottom_b = PartyBBottomModel(input_dim=2, embedding_dim=2, hidden_dims=[])
    top = TopModel(embedding_dim_total=6, output_dim=2, hidden_dims=[])

    # Create small batch
    x_a = torch.randn(4, 3)
    x_b = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))

    # Forward pass
    def compute_loss():
        pred, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)
        logits = top.forward_logits(torch.cat([emb_a, emb_b], dim=1))
        return nn.CrossEntropyLoss()(logits, labels)

    loss = compute_loss()

    # Backward pass
    _, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)
    secure_backward(top, emb_a, emb_b, loss, bottom_a, bottom_b, x_a, x_b)

    # Check analytical vs numerical gradients for one parameter
    param_to_check = list(bottom_a.parameters())[0]  # First layer weights

    analytical_grad = param_to_check.grad.clone()

    # Numerical gradient
    numerical_grad = numerical_gradient(
        param_to_check,
        param_to_check.data,
        lambda p: compute_loss(),
        epsilon=1e-4
    )

    # Compare
    grad_diff = (analytical_grad - numerical_grad).abs().max().item()
    grad_norm = analytical_grad.abs().max().item()

    relative_error = grad_diff / (grad_norm + 1e-8)

    print(f"  Max absolute difference: {grad_diff:.6e}")
    print(f"  Max analytical gradient: {grad_norm:.6e}")
    print(f"  Relative error: {relative_error:.6e}")

    # Tolerance: 1e-3 for numerical gradient checking
    assert relative_error < 1e-3, f"Gradient check failed: relative error {relative_error}"

    print("✓ Gradient correctness verified (relative error < 1e-3)")
    return True


def test_embedding_gradients_flow():
    """Test that embedding gradients flow correctly through chain rule."""
    print("\n=== Testing Embedding Gradient Flow ===")

    # Create models
    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    # Create data
    x_a = torch.randn(8, 7)
    x_b = torch.randn(8, 3)
    labels = torch.randint(0, 2, (8,))

    # Forward pass
    predictions, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)

    # Compute loss
    logits = top.forward_logits(torch.cat([emb_a, emb_b], dim=1))
    loss = nn.CrossEntropyLoss()(logits, labels)

    # Backward pass
    secure_backward(top, emb_a, emb_b, loss, bottom_a, bottom_b, x_a, x_b)

    # Verify embedding gradients exist
    assert emb_a.grad is not None, "Embedding A has no gradient"
    assert emb_b.grad is not None, "Embedding B has no gradient"

    # Check shapes
    assert emb_a.grad.shape == emb_a.shape, f"Embedding A grad shape mismatch: {emb_a.grad.shape} vs {emb_a.shape}"
    assert emb_b.grad.shape == emb_b.shape, f"Embedding B grad shape mismatch: {emb_b.grad.shape} vs {emb_b.shape}"

    # Check that gradients are non-zero (loss should depend on embeddings)
    assert emb_a.grad.abs().sum() > 0, "Embedding A gradients are zero"
    assert emb_b.grad.abs().sum() > 0, "Embedding B gradients are zero"

    print("✓ Embedding gradients flow correctly")
    print(f"  Embedding A grad norm: {emb_a.grad.norm().item():.4f}")
    print(f"  Embedding B grad norm: {emb_b.grad.norm().item():.4f}")

    return True


def test_gradient_step_updates_parameters():
    """Test that gradient step actually updates parameters."""
    print("\n=== Testing Parameter Updates ===")

    # Create models
    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    # Store initial parameters
    initial_params = {
        'bottom_a': [p.clone() for p in bottom_a.parameters()],
        'bottom_b': [p.clone() for p in bottom_b.parameters()],
        'top': [p.clone() for p in top.parameters()],
    }

    # Training step
    x_a = torch.randn(16, 7)
    x_b = torch.randn(16, 3)
    labels = torch.randint(0, 2, (16,))

    predictions, emb_a, emb_b = secure_forward(bottom_a, bottom_b, top, x_a, x_b)
    logits = top.forward_logits(torch.cat([emb_a, emb_b], dim=1))
    loss = nn.CrossEntropyLoss()(logits, labels)

    secure_backward(top, emb_a, emb_b, loss, bottom_a, bottom_b, x_a, x_b)

    # Optimizer step
    optimizer = torch.optim.Adam([
        *bottom_a.parameters(),
        *bottom_b.parameters(),
        *top.parameters()
    ], lr=0.001)

    optimizer.step()

    # Check that parameters changed
    for name, model in [('bottom_a', bottom_a), ('bottom_b', bottom_b), ('top', top)]:
        for i, (initial, current) in enumerate(zip(initial_params[name], model.parameters())):
            param_diff = (initial - current).abs().max().item()
            assert param_diff > 0, f"{name} parameter {i} did not update"

    print("✓ Parameters updated after gradient step")

    return True


def run_all_tests():
    """Run all gradient flow tests."""
    print("\n" + "="*80)
    print("GRADIENT FLOW UNIT TESTS")
    print("="*80)

    tests = [
        test_forward_pass_shapes,
        test_backward_pass_gradients_exist,
        test_gradient_correctness,
        test_embedding_gradients_flow,
        test_gradient_step_updates_parameters,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except AssertionError as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
        except Exception as e:
            print(f"\n✗ {test.__name__} ERROR: {e}")
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
