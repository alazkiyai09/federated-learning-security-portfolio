"""
Privacy Calibration Utilities
==============================

Tools for computing DP parameters from target privacy budget.

This module provides utilities to calibrate noise multiplier σ
from target (ε, δ) and training parameters (q, T).
"""

import numpy as np
from typing import Optional, tuple
import yaml


def compute_noise_multiplier_from_epsilon(
    target_epsilon: float,
    delta: float,
    steps: int,
    sampling_rate: float,
    orders: Optional[np.ndarray] = None,
    verbose: bool = False
) -> float:
    """Compute noise multiplier σ needed to achieve target (ε, δ).

    This is the main calibration function. Given a target privacy budget
    and training parameters, it computes the minimum noise multiplier
    needed to stay within the budget.

    Args:
        target_epsilon: Target ε (privacy budget)
        delta: Failure probability δ (typically 1/n or 1e-5)
        steps: Number of training steps (rounds * local_epochs * batches_per_epoch)
        sampling_rate: q = batch_size / training_set_size
        orders: RDP orders (optional, uses defaults if None)
        verbose: Print calibration progress

    Returns:
        noise_multiplier: Minimum σ to achieve (ε, δ)

    Raises:
        ValueError: If parameters are invalid or target is unreachable

    Example:
        >>> sigma = compute_noise_multiplier_from_epsilon(
        ...     target_epsilon=1.0,
        ...     delta=1e-5,
        ...     steps=5000,  # 100 rounds * 5 epochs * 10 batches
        ...     sampling_rate=0.01  # batch_size=100, n=10000
        ... )
        >>> print(f"Required σ = {sigma:.2f}")
    """
    from ..dp_mechanisms.noise_addition import compute_noise_multiplier

    if target_epsilon <= 0:
        raise ValueError(f"Target epsilon must be positive, got {target_epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"Delta must be in (0, 1), got {delta}")
    if steps <= 0:
        raise ValueError(f"Steps must be positive, got {steps}")
    if not (0 < sampling_rate <= 1):
        raise ValueError(f"Sampling rate must be in (0, 1], got {sampling_rate}")

    if verbose:
        print(f"Calibrating σ for:")
        print(f"  Target ε = {target_epsilon}")
        print(f"  δ = {delta:.2e}")
        print(f"  Steps = {steps}")
        print(f"  Sampling rate q = {sampling_rate}")

    sigma = compute_noise_multiplier(
        target_epsilon=target_epsilon,
        delta=delta,
        steps=steps,
        sampling_rate=sampling_rate,
        orders=orders
    )

    if verbose:
        print(f"\nCalibrated σ = {sigma:.4f}")

    return sigma


def compute_steps_from_epsilon(
    target_epsilon: float,
    delta: float,
    noise_multiplier: float,
    sampling_rate: float,
    verbose: bool = False
) -> int:
    """Compute maximum steps within ε budget given noise multiplier.

    Inverse of compute_noise_multiplier_from_epsilon. Given a fixed
    noise multiplier (e.g., using a pretrained DP model), compute how
    many training steps you can run.

    Args:
        target_epsilon: Target ε
        delta: Failure probability δ
        noise_multiplier: Fixed noise multiplier σ
        sampling_rate: q
        verbose: Print progress

    Returns:
        max_steps: Maximum steps within ε budget

    Example:
        >>> max_steps = compute_steps_from_epsilon(
        ...     target_epsilon=1.0,
        ...     delta=1e-5,
        ...     noise_multiplier=1.5,
        ...     sampling_rate=0.01
        ... )
        >>> print(f"Can run {max_steps} steps")
    """
    from ..dp_mechanisms.privacy_accountant import compute_steps_for_epsilon

    if verbose:
        print(f"Computing max steps for:")
        print(f"  Target ε = {target_epsilon}")
        print(f"  δ = {delta:.2e}")
        print(f"  σ = {noise_multiplier}")
        print(f"  Sampling rate q = {sampling_rate}")

    max_steps = compute_steps_for_epsilon(
        target_epsilon=target_epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate
    )

    if verbose:
        print(f"\nMax steps = {max_steps}")

    return max_steps


def calibrate_from_config(config_path: str) -> dict:
    """Load config and compute DP parameters.

    Convenience function to load YAML config and compute all
    necessary DP parameters.

    Args:
        config_path: Path to config YAML file

    Returns:
        params: Dictionary with computed DP parameters

    Example:
        >>> params = calibrate_from_config("config/privacy.yaml")
        >>> print(params)
        {'noise_multiplier': 1.23, 'steps': 5000, 'epsilon': 1.0, ...}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    privacy = config['privacy']
    noise = config['noise']

    # Compute noise multiplier for each target epsilon
    results = {}
    for eps in privacy['epsilon_grid']:
        sigma = compute_noise_multiplier_from_epsilon(
            target_epsilon=eps,
            delta=privacy['delta'],
            steps=noise['steps'],
            sampling_rate=noise['sampling_rate']
        )

        results[eps] = {
            'noise_multiplier': sigma,
            'steps': noise['steps'],
            'sampling_rate': noise['sampling_rate'],
            'delta': privacy['delta']
        }

    return results


def compute_sampling_rate(
    batch_size: int,
    dataset_size: int
) -> float:
    """Compute sampling rate q = batch_size / dataset_size.

    Args:
        batch_size: Batch size
        dataset_size: Total training set size

    Returns:
        sampling_rate: q

    Example:
        >>> q = compute_sampling_rate(batch_size=32, dataset_size=10000)
        >>> print(f"Sampling rate q = {q:.4f}")
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    if dataset_size <= 0:
        raise ValueError(f"Dataset size must be positive, got {dataset_size}")
    if batch_size > dataset_size:
        raise ValueError(
            f"Batch size ({batch_size}) cannot exceed dataset size ({dataset_size})"
        )

    return batch_size / dataset_size


def compute_total_steps(
    num_rounds: int,
    local_epochs: int,
    batches_per_epoch: int
) -> int:
    """Compute total training steps in federated learning.

    Args:
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per round
        batches_per_epoch: Batches per local epoch

    Returns:
        total_steps: Total number of gradient steps

    Example:
        >>> total_steps = compute_total_steps(
        ...     num_rounds=100,
        ...     local_epochs=5,
        ...     batches_per_epoch=10
        ... )
        >>> print(f"Total steps = {total_steps}")
        Total steps = 5000
    """
    return num_rounds * local_epochs * batches_per_epoch


def validate_dp_parameters(
    noise_multiplier: float,
    sampling_rate: float,
    steps: int,
    target_epsilon: float,
    delta: float
) -> bool:
    """Validate that DP parameters achieve target privacy budget.

    Args:
        noise_multiplier: σ
        sampling_rate: q
        steps: Number of steps
        target_epsilon: Target ε
        delta: Target δ

    Returns:
        valid: True if parameters achieve (or exceed) target privacy

    Example:
        >>> valid = validate_dp_parameters(
        ...     noise_multiplier=1.5,
        ...     sampling_rate=0.01,
        ...     steps=1000,
        ...     target_epsilon=1.0,
        ...     delta=1e-5
        ... )
        >>> print(f"Parameters valid: {valid}")
    """
    from ..dp_mechanisms.privacy_accountant import compute_rdp

    actual_epsilon, _ = compute_rdp(
        noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate,
        steps=steps,
        delta=delta
    )

    # We want actual_epsilon <= target_epsilon (more private is okay)
    return actual_epsilon <= target_epsilon


def print_privacy_summary(
    noise_multiplier: float,
    sampling_rate: float,
    steps: int,
    delta: float
) -> None:
    """Print summary of privacy parameters and resulting budget.

    Args:
        noise_multiplier: σ
        sampling_rate: q
        steps: Number of steps
        delta: δ
    """
    from ..dp_mechanisms.privacy_accountant import compute_rdp

    epsilon, _ = compute_rdp(
        noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate,
        steps=steps,
        delta=delta
    )

    print("=" * 60)
    print("Differential Privacy Summary")
    print("=" * 60)
    print(f"Noise multiplier (σ)     : {noise_multiplier:.4f}")
    print(f"Sampling rate (q)        : {sampling_rate:.4f}")
    print(f"Training steps           : {steps}")
    print(f"Delta (δ)                : {delta:.2e}")
    print("-" * 60)
    print(f"Privacy spent (ε)        : {epsilon:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    print("Testing privacy calibration...")

    # Test 1: Compute sigma from epsilon
    print("\n--- Test 1: Calibrate σ from ε ---")
    sigma = compute_noise_multiplier_from_epsilon(
        target_epsilon=1.0,
        delta=1e-5,
        steps=5000,
        sampling_rate=0.01,
        verbose=True
    )

    # Test 2: Compute max steps from sigma
    print("\n--- Test 2: Compute Max Steps from σ ---")
    max_steps = compute_steps_from_epsilon(
        target_epsilon=1.0,
        delta=1e-5,
        noise_multiplier=1.5,
        sampling_rate=0.01,
        verbose=True
    )

    # Test 3: Helper functions
    print("\n--- Test 3: Helper Functions ---")
    q = compute_sampling_rate(batch_size=32, dataset_size=10000)
    print(f"Sampling rate: {q:.4f}")

    steps = compute_total_steps(num_rounds=100, local_epochs=5, batches_per_epoch=10)
    print(f"Total steps: {steps}")

    # Test 4: Validation
    print("\n--- Test 4: Validate Parameters ---")
    valid = validate_dp_parameters(
        noise_multiplier=1.5,
        sampling_rate=0.01,
        steps=1000,
        target_epsilon=1.0,
        delta=1e-5
    )
    print(f"Parameters achieve ε=1.0: {valid}")

    # Test 5: Privacy summary
    print("\n--- Test 5: Privacy Summary ---")
    print_privacy_summary(
        noise_multiplier=1.5,
        sampling_rate=0.01,
        steps=1000,
        delta=1e-5
    )
