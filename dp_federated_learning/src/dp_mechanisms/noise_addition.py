"""
Gaussian Noise Addition for Differential Privacy
=================================================

Implements calibrated Gaussian noise injection for (ε, δ)-DP.

Mathematical Background:
    To achieve (ε, δ)-DP with DP-SGD:
    1. Clip gradients to bound sensitivity: Δ = 2C (for L2 clipping)
    2. Add Gaussian noise: N(0, σ²) where σ ≥ Δ * sqrt(2 ln(1.25/δ)) / ε

    In the federated setting with sampling rate q = batch_size / n:
    - The noise multiplier is calibrated differently
    - Use RDP accountant for tight bounds
"""

import torch
import numpy as np
from typing import Optional


def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_multiplier: float,
    clipping_bound: float,
    num_samples: int,
    num_clients: Optional[int] = None
) -> torch.Tensor:
    """Add calibrated Gaussian noise for (ε, δ)-DP.

    This adds noise: N(0, σ² * C²) where σ is the noise multiplier
    and C is the clipping bound.

    The sensitivity is:
    - Central DP: Δ = C (after aggregation)
    - Local DP: Δ = 2C (per-sample sensitivity)

    Args:
        tensor: Tensor to add noise to (typically aggregated gradients)
        noise_multiplier: σ (noise multiplier relative to clipping bound)
        clipping_bound: C (L2 clipping threshold)
        num_samples: Number of samples in the batch (for scaling)
        num_clients: Number of clients (for central DP scaling)

    Returns:
        noised_tensor: Tensor with calibrated Gaussian noise added

    Mathematical Details:
        Standard Gaussian mechanism (Dwork et al.):
            For sensitivity Δ and privacy parameters (ε, δ):
            σ ≥ Δ * sqrt(2 ln(1.25/δ)) / ε

        In DP-SGD with sampling rate q and noise multiplier z:
            Noise scale = z * C * sqrt(num_samples)

        For federated aggregation with K clients:
            Noise scale = z * C * sqrt(K)  (central DP)

    Example:
        >>> grads = torch.randn(100)
        >>> noised = add_gaussian_noise(grads, noise_multiplier=1.0,
        ...                             clipping_bound=1.0, num_samples=32)
        >>> torch.allclose(grads, noised)
        False  # Noise has been added
    """
    if noise_multiplier < 0:
        raise ValueError(f"Noise multiplier must be non-negative, got {noise_multiplier}")
    if clipping_bound <= 0:
        raise ValueError(f"Clipping bound must be positive, got {clipping_bound}")

    device = tensor.device
    dtype = tensor.dtype

    # Compute noise standard deviation
    # Base scale: σ * C
    # Scale by sqrt(num_samples) for aggregation
    if num_clients is not None:
        # Central DP: scale by sqrt(num_clients)
        scale = noise_multiplier * clipping_bound * np.sqrt(num_clients)
    else:
        # Local DP or standard DP-SGD: scale by sqrt(num_samples)
        scale = noise_multiplier * clipping_bound * np.sqrt(num_samples)

    # Generate and add Gaussian noise
    noise = torch.randn_like(tensor) * scale
    noised_tensor = tensor + noise

    return noised_tensor


def add_gaussian_noise_per_dimension(
    tensor: torch.Tensor,
    noise_multiplier: float,
    clipping_bound: float
) -> torch.Tensor:
    """Add independent Gaussian noise to each dimension.

    This adds noise N(0, σ² * C²) to each dimension independently.
    Useful for per-layer noise addition in neural networks.

    Args:
        tensor: Input tensor
        noise_multiplier: σ
        clipping_bound: C

    Returns:
        Noised tensor
    """
    scale = noise_multiplier * clipping_bound
    noise = torch.randn_like(tensor) * scale
    return tensor + noise


def compute_noise_multiplier(
    target_epsilon: float,
    delta: float,
    steps: int,
    sampling_rate: float,
    orders: Optional[np.ndarray] = None
) -> float:
    """Compute noise multiplier σ for target (ε, δ) using RDP analysis.

    This implements the binary search from Abadi et al. 2016 to find
    the minimum noise multiplier needed to achieve the target privacy
    budget after T steps with sampling rate q.

    The function uses Renyi Differential Privacy (RDP) analysis to
    compute the tight privacy bounds for DP-SGD.

    Args:
        target_epsilon: Target ε (privacy budget)
        delta: Failure probability δ (typically 1/n or 1e-5)
        steps: Number of training steps (rounds * local_epochs)
        sampling_rate: q = batch_size / training_set_size
        orders: Array of RDP orders (α values)

    Returns:
        noise_multiplier: Minimum σ to achieve (ε, δ)-DP

    Mathematical Background:
        For Gaussian mechanism with noise multiplier z and sampling rate q,
        the RDP at order α is:
            ε_α = q² * (α z²) / 2

        After T steps, total RDP is T * ε_α.

        Convert RDP to (ε, δ)-DP:
            ε = min_α [ ε_α + (log(1/δ) / (α-1)) ]

        We binary search for the minimum z such that ε ≤ target_epsilon.

    Algorithm:
        1. Binary search on z ∈ [0.1, 100.0]
        2. For each candidate z, compute RDP over T steps
        3. Convert to (ε, δ)-DP
        4. Adjust search bounds until convergence

    Example:
        >>> sigma = compute_noise_multiplier(
        ...     target_epsilon=1.0,
        ...     delta=1e-5,
        ...     steps=1000,
        ...     sampling_rate=0.01
        ... )
        >>> print(f"Required noise multiplier: {sigma:.2f}")
    """
    if orders is None:
        # Default RDP orders from Abadi et al.
        orders = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          12.0, 16.0, 20.0, 32.0, 64.0, 128.0])

    def compute_epsilon_for_sigma(sigma: float) -> float:
        """Compute ε for a given noise multiplier sigma."""
        # RDP for Gaussian mechanism with sampling rate q
        # Formula: ε_α = q² * α / (2 * σ²)  (from Abadi et al. 2016)
        # But we use sampling_rate = q, so:
        # ε_α = (sampling_rate² * α) / (2 * σ²)

        # Actually, the correct formula from Abadi et al. is:
        # For Gaussian mechanism with noise σ and sampling probability q:
        # RDP at order α is approximately q² * (α * z²) / 2
        # where z is the noise multiplier (σ / sensitivity)

        # Using the formula from the paper:
        # ε_α(q) ≈ q² * α / (2 * σ²)
        rdp = (sampling_rate ** 2) * orders / (2 * sigma ** 2)

        # Total RDP after T steps
        total_rdp = steps * rdp

        # Convert to (ε, δ)-DP
        # ε = min_α [ total_rdp_α + log(1/δ) / (α - 1) ]
        epsilons = total_rdp + np.log(1/delta) / (orders - 1)

        return epsilons.min()

    # Binary search for noise multiplier
    # Lower bound: very small σ (very private)
    # Upper bound: very large σ (not private)
    sigma_low, sigma_high = 0.1, 100.0

    # First, check if upper bound is sufficient
    epsilon_high = compute_epsilon_for_sigma(sigma_high)
    if epsilon_high > target_epsilon:
        raise ValueError(
            f"Cannot achieve ε={target_epsilon} with σ={sigma_high}. "
            f"Got ε={epsilon_high:.2f}. Try increasing sigma_high or reducing steps/delta."
        )

    # Binary search
    for _ in range(40):  # 40 iterations gives ~1e-12 precision
        sigma_mid = (sigma_low + sigma_high) / 2
        epsilon_mid = compute_epsilon_for_sigma(sigma_mid)

        if epsilon_mid < target_epsilon:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return sigma_high


def compute_noise_multiplier_standard(
    target_epsilon: float,
    delta: float
) -> float:
    """Compute noise multiplier using standard Gaussian mechanism formula.

    This is the basic formula without sampling or composition:
        σ = Δ * sqrt(2 ln(1.25/δ)) / ε

    where Δ is the sensitivity (2C for per-sample gradients).

    Use this for simple cases without sampling or repeated queries.

    Args:
        target_epsilon: Target ε
        delta: Failure probability δ

    Returns:
        noise_multiplier: Required σ (assuming Δ=1 for unit sensitivity)

    Example:
        >>> sigma = compute_noise_multiplier_standard(1.0, 1e-5)
        >>> print(f"σ = {sigma:.2f}")
        σ = 4.69
    """
    if target_epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {target_epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"Delta must be in (0, 1), got {delta}")

    sigma = np.sqrt(2 * np.log(1.25 / delta)) / target_epsilon
    return sigma


def compute_delta_for_epsilon(
    epsilon: float,
    sigma: float,
    steps: int,
    sampling_rate: float
) -> float:
    """Compute δ for given ε and noise multiplier.

    This is useful when you have fixed ε and σ and want to know
    what δ you can achieve.

    Args:
        epsilon: Target ε
        sigma: Noise multiplier
        steps: Number of steps
        sampling_rate: q

    Returns:
        delta: Required δ
    """
    # Compute RDP
    orders = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                      12.0, 16.0, 20.0, 32.0, 64.0, 128.0])

    rdp = (sampling_rate ** 2) * orders / (2 * sigma ** 2)
    total_rdp = steps * rdp

    # Solve for δ: δ = exp((α - 1) * (ε - total_rdp_α))
    # We want the minimum δ across all α
    deltas = np.exp((orders - 1) * (epsilon - total_rdp))
    delta = deltas.min()

    return min(delta, 1.0)


if __name__ == "__main__":
    print("Testing noise addition...")

    # Test 1: Basic noise addition
    tensor = torch.zeros(10)
    noised = add_gaussian_noise(tensor, noise_multiplier=1.0,
                                clipping_bound=1.0, num_samples=32)
    print(f"Original: {tensor}")
    print(f"Noised: {noised}")

    # Test 2: Noise calibration
    print("\n--- Noise Multiplier Calibration ---")
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        sigma = compute_noise_multiplier(
            target_epsilon=eps,
            delta=1e-5,
            steps=1000,
            sampling_rate=0.01
        )
        print(f"ε = {eps:4.1f} → σ = {sigma:.2f}")

    # Test 3: Standard formula
    print("\n--- Standard Gaussian Mechanism ---")
    sigma_standard = compute_noise_multiplier_standard(1.0, 1e-5)
    print(f"Standard formula (ε=1.0, δ=1e-5): σ = {sigma_standard:.2f}")
    print(f"Note: RDP gives tighter bounds for composed mechanisms")
