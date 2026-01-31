"""
DP-SGD Optimizer (Custom Implementation)
=========================================

From-scratch implementation of DP-SGD as described in:
Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)

Algorithm 1: DP-SGD
    For each batch {(x_i, y_i)}:
        1. Compute per-sample gradients: g_i = ∇_θ L(f_θ(x_i), y_i)
        2. Clip gradients: g_ĩ = g_i / max(1, ||g_i||_2 / C)
        3. Aggregate: g = Σ_i g_ĩ
        4. Add noise: ĝ = g + N(0, σ² * C² * I)
        5. Update: θ = θ - η * ĝ
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from ..dp_mechanisms.gradient_clipper import (
    compute_per_sample_gradients,
    clip_gradients_l2
)
from ..dp_mechanisms.noise_addition import add_gaussian_noise
from ..dp_mechanisms.privacy_accountant import RDPAccountant


@dataclass
class DPSGDMetrics:
    """Metrics from a DP-SGD step."""
    clip_norm_mean: float  # Average L2 norm before clipping
    clip_norm_max: float   # Maximum L2 norm before clipping
    clip_fraction: float   # Fraction of gradients clipped
    noise_std: float       # Standard deviation of noise added
    epsilon_spent: float   # Privacy budget consumed this step


class DPSGDOptimizer:
    """Differentially Private SGD Optimizer.

    This implements DP-SGD from Abadi et al. 2016 with:
    - Per-sample gradient computation
    - L2 gradient clipping to bound C
    - Gaussian noise addition with multiplier σ
    - Automatic privacy accounting

    Usage:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = DPSGDOptimizer(
        ...     model=model,
        ...     noise_multiplier=1.5,
        ...     clipping_bound=1.0,
        ...     batch_size=32,
        ...     lr=0.01,
        ...     dataset_size=1000
        ... )
        >>>
        >>> for x_batch, y_batch in dataloader:
        ...     metrics = optimizer.step(x_batch, y_batch, loss_fn)
        ...     print(f"ε spent: {metrics.epsilon_spent:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float,
        clipping_bound: float,
        batch_size: int,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        delta: float = 1e-5,
        sampling_rate: Optional[float] = None,
        dataset_size: Optional[int] = None
    ):
        """Initialize DP-SGD optimizer.

        Args:
            model: PyTorch model to optimize
            noise_multiplier: σ (noise multiplier for Gaussian mechanism)
            clipping_bound: C (L2 norm bound for gradient clipping)
            batch_size: Batch size B
            lr: Learning rate η
            momentum: Momentum coefficient (default: 0, no momentum)
            weight_decay: L2 regularization coefficient
            delta: Privacy parameter δ (for accounting)
            sampling_rate: q = batch_size / dataset_size (optional)
            dataset_size: Total training set size (optional, used to compute q)

        Raises:
            ValueError: If parameters are invalid
        """
        if noise_multiplier <= 0:
            raise ValueError(f"Noise multiplier must be positive, got {noise_multiplier}")
        if clipping_bound <= 0:
            raise ValueError(f"Clipping bound must be positive, got {clipping_bound}")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        self.model = model
        self.params = list(model.parameters())
        self.noise_multiplier = noise_multiplier
        self.clipping_bound = clipping_bound
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Compute sampling rate
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        elif dataset_size is not None:
            self.sampling_rate = batch_size / dataset_size
        else:
            raise ValueError("Must provide either sampling_rate or dataset_size")

        # Privacy accounting
        self.delta = delta
        self.accountant = RDPAccountant(
            noise_multiplier=noise_multiplier,
            sampling_rate=self.sampling_rate,
            delta=delta
        )
        self.steps = 0

        # Momentum state
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module
    ) -> DPSGDMetrics:
        """Execute one DP-SGD step.

        Algorithm:
            1. Compute per-sample gradients
            2. Clip to L2 norm bound C
            3. Aggregate clipped gradients
            4. Add Gaussian noise N(0, σ² * C²)
            5. Apply momentum (if enabled)
            6. Update parameters

        Args:
            inputs: Batch of inputs (batch_size, ...)
            targets: Batch of targets (batch_size, ...)
            loss_fn: Loss function with reduction='none'

        Returns:
            metrics: DPSGDMetrics with statistics from this step
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Step 1: Compute per-sample gradients
        # Shape: (batch_size, num_params)
        per_sample_grads = compute_per_sample_gradients(
            self.model,
            inputs,
            targets,
            loss_fn
        )

        # Step 2: Clip gradients to L2 norm bound C
        clipped_grads, clip_norms = clip_gradients_l2(
            per_sample_grads,
            self.clipping_bound
        )

        # Compute clipping statistics
        clip_norm_mean = clip_norms.mean().item()
        clip_norm_max = clip_norms.max().item()
        clip_fraction = (clip_norms > self.clipping_bound).float().mean().item()

        # Step 3: Aggregate clipped gradients
        # Shape: (num_params,)
        aggregated_grads = clipped_grads.sum(dim=0)

        # Step 4: Add Gaussian noise
        # Noise scale: σ * C * sqrt(batch_size)
        noised_grads = add_gaussian_noise(
            aggregated_grads,
            noise_multiplier=self.noise_multiplier,
            clipping_bound=self.clipping_bound,
            num_samples=batch_size
        )

        # Compute noise statistics
        noise_std = self.noise_multiplier * self.clipping_bound * (batch_size ** 0.5)

        # Step 5: Apply weight decay (L2 regularization)
        if self.weight_decay > 0:
            for p in self.params:
                noised_grads = noised_grads  # Placeholder - applied per-parameter below

        # Reshape flattened gradients and apply to parameters
        idx = 0
        for i, p in enumerate(self.params):
            # Get gradient for this parameter
            param_size = p.numel()
            p_grad = noised_grads[idx:idx + param_size].view_as(p)

            # Apply weight decay
            if self.weight_decay > 0:
                p_grad = p_grad + self.weight_decay * p.data

            # Apply momentum
            if self.momentum > 0:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + p_grad
                )
                update = self.velocity[i]
            else:
                update = p_grad

            # Step 6: Update parameters
            p.data = p.data - self.lr * update

            idx += param_size

        # Update privacy accounting
        self.accountant.step(num_steps=1)
        self.steps += 1

        epsilon_spent = self.accountant.get_epsilon()

        return DPSGDMetrics(
            clip_norm_mean=clip_norm_mean,
            clip_norm_max=clip_norm_max,
            clip_fraction=clip_fraction,
            noise_std=noise_std,
            epsilon_spent=epsilon_spent
        )

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get total privacy budget consumed.

        Returns:
            (epsilon, delta): Total (ε, δ) spent so far
        """
        return self.accountant.get_privacy_spent()

    def get_epsilon(self) -> float:
        """Get total ε consumed."""
        return self.accountant.get_epsilon()


def create_dp_sgd_optimizer(
    model: nn.Module,
    noise_multiplier: float,
    clipping_bound: float,
    batch_size: int,
    lr: float,
    dataset_size: int,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    delta: float = 1e-5
) -> DPSGDOptimizer:
    """Convenience function to create DP-SGD optimizer from a model.

    Args:
        model: PyTorch model
        noise_multiplier: σ
        clipping_bound: C
        batch_size: B
        lr: Learning rate
        dataset_size: Total training set size n
        momentum: Momentum coefficient
        weight_decay: L2 regularization
        delta: Privacy parameter δ

    Returns:
        DPSGDOptimizer instance
    """
    return DPSGDOptimizer(
        model=model,
        noise_multiplier=noise_multiplier,
        clipping_bound=clipping_bound,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        delta=delta,
        dataset_size=dataset_size
    )


if __name__ == "__main__":
    print("Testing DP-SGD Optimizer...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    # Create DP-SGD optimizer
    optimizer = create_dp_sgd_optimizer(
        model=model,
        noise_multiplier=1.5,
        clipping_bound=1.0,
        batch_size=32,
        lr=0.01,
        dataset_size=1000,
        delta=1e-5
    )

    print(f"Created DP-SGD optimizer:")
    print(f"  Noise multiplier: {optimizer.noise_multiplier}")
    print(f"  Clipping bound: {optimizer.clipping_bound}")
    print(f"  Sampling rate: {optimizer.sampling_rate:.4f}")

    # Simulate training step
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    metrics = optimizer.step(inputs, targets, loss_fn)

    print(f"\nStep metrics:")
    print(f"  Clip norm mean: {metrics.clip_norm_mean:.4f}")
    print(f"  Clip norm max: {metrics.clip_norm_max:.4f}")
    print(f"  Clip fraction: {metrics.clip_fraction:.2%}")
    print(f"  Noise std: {metrics.noise_std:.4f}")
    print(f"  ε spent: {metrics.epsilon_spent:.6f}")

    # Multiple steps
    print("\nRunning 10 more steps...")
    for i in range(10):
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        metrics = optimizer.step(inputs, targets, loss_fn)

    eps, delta = optimizer.get_privacy_spent()
    print(f"After 11 steps: ε = {eps:.4f}, δ = {delta:.2e}")
