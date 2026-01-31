"""
Gradient Clipping for Differential Privacy
===========================================

Implements per-sample gradient computation and L2 norm clipping
as required for DP-SGD (Abadi et al. 2016).

Mathematical Background:
    For each sample i in batch:
    1. Compute gradient g_i = ∇_θ L(f_θ(x_i), y_i)
    2. Clip to L2 norm bound C: g_ĩ = g_i / max(1, ||g_i||_2 / C)

    This ensures ||g_ĩ||_2 ≤ C for all samples, enabling the
    sensitivity analysis required for DP guarantees.
"""

import torch
import torch.nn as nn


def compute_per_sample_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module
) -> torch.Tensor:
    """Compute gradients for each sample in the batch.

    This is a critical step for DP-SGD: we need per-sample gradients
    to clip them individually before aggregation.

    Algorithm:
        1. For each sample (x_i, y_i) in batch:
        2. Compute loss l_i = loss_fn(model(x_i), y_i)
        3. Backward pass to get gradient g_i = ∇_θ l_i
        4. Store g_i in output tensor

    Args:
        model: PyTorch model with parameters to compute gradients for
        inputs: Batch of input data (batch_size, ...)
        targets: Batch of target labels (batch_size, ...)
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss(reduction='none'))

    Returns:
        per_sample_grads: Tensor of shape (batch_size, num_params)
            where per_sample_grads[i] is the flattened gradient for sample i

    Example:
        >>> model = nn.Linear(10, 2)
        >>> inputs = torch.randn(32, 10)  # batch_size=32
        >>> targets = torch.randint(0, 2, (32,))
        >>> loss_fn = nn.CrossEntropyLoss(reduction='none')
        >>> grads = compute_per_sample_gradients(model, inputs, targets, loss_fn)
        >>> grads.shape
        torch.Size([32, 22])  # 32 samples, 22 params (10*2 + 2 bias)
    """
    batch_size = inputs.shape[0]
    device = inputs.device

    # Get all model parameters
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    # Initialize output tensor
    per_sample_grads = torch.zeros(batch_size, num_params, device=device)

    # Store original parameters and gradients
    original_params = [p.data.clone() for p in params]
    original_grads = [p.grad.clone() if p.grad is not None else None
                      for p in params]

    try:
        # Compute gradient for each sample independently
        for i in range(batch_size):
            # Zero out all gradients
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            # Compute loss for single sample
            single_input = inputs[i:i+1]  # Keep batch dimension
            single_target = targets[i:i+1]
            output = model(single_input)
            loss = loss_fn(output, single_target)

            # Backward pass
            loss.backward()

            # Extract and flatten gradients
            grad_list = []
            for p in params:
                if p.grad is not None:
                    grad_list.append(p.grad.view(-1).clone())
                else:
                    grad_list.append(torch.zeros(p.numel(), device=device))

            per_sample_grads[i] = torch.cat(grad_list)

    finally:
        # Restore original parameters and gradients
        for p, orig_param, orig_grad in zip(params, original_params, original_grads):
            p.data = orig_param
            if orig_grad is not None:
                p.grad = orig_grad
            else:
                p.grad = None

    return per_sample_grads


def clip_gradients_l2(
    per_sample_grads: torch.Tensor,
    clipping_bound: float
) -> tuple:
    """Clip per-sample gradients to L2 norm bound C.

    This implements the clipping operation from Algorithm 1 of
    Abadi et al. 2016:
        g_ĩ = g_i / max(1, ||g_i||_2 / C)

    The clipping ensures each gradient has L2 norm at most C,
    which bounds the sensitivity for noise addition.

    Args:
        per_sample_grads: Tensor of shape (batch_size, num_params)
            containing gradients for each sample
        clipping_bound: Maximum L2 norm C (clipping threshold)

    Returns:
        clipped_grads: Clipped gradients, same shape as input
        clip_norms: Per-sample L2 norms BEFORE clipping (for statistics)

    Mathematical Details:
        Let g_i be the gradient for sample i.
        Define the clipping factor:
            α_i = max(1, ||g_i||_2 / C)
        Then the clipped gradient is:
            g_ĩ = g_i / α_i

        This ensures ||g_ĩ||_2 = min(||g_i||_2, C) ≤ C.

    Example:
        >>> grads = torch.randn(32, 100)  # 32 samples, 100 params
        >>> clipped, norms = clip_gradients_l2(grads, clipping_bound=1.0)
        >>> clipped.norm(dim=1).max().item()  # All norms ≤ 1.0
        1.0
        >>> norms.max().item()  # Some original norms > 1.0
        2.37
    """
    if clipping_bound <= 0:
        raise ValueError(f"Clipping bound must be positive, got {clipping_bound}")

    # Compute L2 norm for each sample's gradient
    # Shape: (batch_size,)
    clip_norms = per_sample_grads.norm(dim=1, p=2)

    # Compute scaling factors: α_i = max(1, ||g_i||_2 / C)
    # Shape: (batch_size, 1) for broadcasting
    scale_factors = torch.clamp(
        clip_norms / clipping_bound,
        min=1.0
    ).unsqueeze(1)

    # Apply clipping: g_ĩ = g_i / α_i
    clipped_grads = per_sample_grads / scale_factors

    return clipped_grads, clip_norms


def flat_clip_gradients_l2(
    model: nn.Module,
    clipping_bound: float,
    batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternative implementation: clip aggregated gradient after backward.

    This computes gradients for the entire batch at once (standard backward),
    then estimates per-sample clipping by scaling the aggregated gradient.

    WARNING: This is an approximation and does NOT provide true per-sample
    guarantees. Use compute_per_sample_gradients() + clip_gradients_l2()
    for correct DP implementation.

    This method is included for educational purposes and to contrast with
    the correct per-sample clipping approach.

    Args:
        model: PyTorch model
        clipping_bound: L2 norm bound C
        batch_size: Batch size used for scaling

    Returns:
        flat_grads: Flattened clipped gradients
        clip_norm: Estimated clip norm (less accurate)
    """
    # Get all gradients
    params = list(model.parameters())
    grad_list = []
    for p in params:
        if p.grad is not None:
            grad_list.append(p.grad.view(-1))
        else:
            grad_list.append(torch.zeros(p.numel(), device=p.device))

    flat_grads = torch.cat(grad_list)

    # Compute L2 norm
    grad_norm = flat_grads.norm(p=2)

    # Scale (this is NOT per-sample clipping!)
    scale_factor = max(1.0, grad_norm / clipping_bound)
    flat_grads = flat_grads / scale_factor

    return flat_grads, grad_norm


if __name__ == "__main__":
    # Test the clipping implementation
    print("Testing gradient clipping...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    # Create dummy data
    batch_size = 8
    inputs = torch.randn(batch_size, 10)
    targets = torch.randint(0, 2, (batch_size,))
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Compute per-sample gradients
    grads = compute_per_sample_gradients(model, inputs, targets, loss_fn)
    print(f"Per-sample gradients shape: {grads.shape}")

    # Check clipping
    C = 1.0
    clipped_grads, norms = clip_gradients_l2(grads, C)

    print(f"\nClipping bound C = {C}")
    print(f"Original norms: {norms}")
    print(f"Clipped norms: {clipped_grads.norm(dim=1)}")
    print(f"Max clipped norm: {clipped_grads.norm(dim=1).max().item():.4f}")

    # Verify all clipped norms ≤ C
    assert (clipped_grads.norm(dim=1) <= C + 1e-6).all(), "Clipping failed!"
    print("\n✓ Clipping test passed!")
