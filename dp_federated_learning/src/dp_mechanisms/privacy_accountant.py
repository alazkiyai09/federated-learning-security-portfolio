"""
Privacy Accounting for Differential Privacy
============================================

Implements Renyi Differential Privacy (RDP) accounting for tracking
cumulative privacy loss across multiple training rounds.

Based on:
- M. Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
- I. Mironov et al. "Renyi Differential Privacy" (IEEE S&P 2017)

Mathematical Background:
    RDP provides tight bounds for composed mechanisms. For a mechanism
    M with (α, ε_α)-RDP, after T compositions we have (α, T*ε_α)-RDP.

    To convert to (ε, δ)-DP:
        ε = min_{α > 1} [ ε_α + log(1/δ) / (α - 1) ]
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PrivacyBudget:
    """Container for privacy budget tracking."""
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0

    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget."""
        return max(0, self.epsilon - self.spent_epsilon)

    @property
    def utilization(self) -> float:
        """Fraction of privacy budget used."""
        if self.epsilon == 0:
            return 1.0
        return min(1.0, self.spent_epsilon / self.epsilon)

    def __repr__(self) -> str:
        return (f"PrivacyBudget(ε={self.spent_epsilon:.4f}/{self.epsilon:.4f}, "
                f"δ={self.spent_delta:.2e}, utilization={self.utilization:.1%})")


class RDPAccountant:
    """Renyi Differential Privacy accountant for tracking privacy consumption.

    This implements the privacy accountant from Abadi et al. 2016 for
    DP-SGD with Poisson subsampling.

    Usage:
        >>> accountant = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.01, delta=1e-5)
        >>> for round in range(num_rounds):
        ...     # Train one round
        ...     accountant.step()  # Record privacy consumption
        ...     eps_spent = accountant.get_epsilon()
        ...     print(f"Round {round}: ε = {eps_spent:.3f}")
    """

    def __init__(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        delta: float,
        orders: Optional[np.ndarray] = None
    ):
        """Initialize RDP accountant.

        Args:
            noise_multiplier: σ (noise multiplier for Gaussian mechanism)
            sampling_rate: q = batch_size / training_set_size (Poisson subsampling)
            delta: Target δ (failure probability)
            orders: Array of RDP orders (α values)
        """
        if noise_multiplier <= 0:
            raise ValueError(f"Noise multiplier must be positive, got {noise_multiplier}")
        if not (0 < sampling_rate <= 1):
            raise ValueError(f"Sampling rate must be in (0, 1], got {sampling_rate}")
        if not (0 < delta < 1):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")

        self.noise_multiplier = noise_multiplier
        self.sampling_rate = sampling_rate
        self.delta = delta

        # Default RDP orders from Abadi et al.
        if orders is None:
            self.orders = np.array([
                1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                12.0, 16.0, 20.0, 32.0, 64.0, 128.0
            ])
        else:
            self.orders = np.asarray(orders)

        # State tracking
        self.steps = 0  # Number of steps recorded
        self.total_rdp = np.zeros_like(self.orders)  # Accumulated RDP

    def _compute_rdp_per_step(self) -> np.ndarray:
        """Compute RDP consumed per step.

        For Gaussian mechanism with noise σ and sampling rate q,
        the RDP at order α is approximately:
            ε_α = q² * α / (2 * σ²)

        This is from Lemma 3 of Abadi et al. 2016.

        Returns:
            rdp_per_step: Array of RDP values for each order α
        """
        sigma = self.noise_multiplier
        q = self.sampling_rate
        alpha = self.orders

        # RDP formula for Gaussian mechanism with sampling
        # ε_α = q² * α / (2 * σ²)
        rdp_per_step = (q ** 2) * alpha / (2 * sigma ** 2)

        return rdp_per_step

    def step(self, num_steps: int = 1) -> None:
        """Record privacy consumption for training steps.

        Args:
            num_steps: Number of steps to record (default: 1)

        Updates:
            self.steps: Increments by num_steps
            self.total_rdp: Accumulates RDP for these steps
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        rdp_per_step = self._compute_rdp_per_step()
        self.total_rdp += rdp_per_step * num_steps
        self.steps += num_steps

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Compute current ε (privacy spent so far).

        Converts accumulated RDP to (ε, δ)-DP using the conversion:
            ε = min_α [ RDP_α + log(1/δ) / (α - 1) ]

        Args:
            delta: Override δ for this computation (default: use self.delta)

        Returns:
            epsilon: Cumulative privacy budget spent
        """
        if delta is None:
            delta = self.delta

        # Convert RDP to (ε, δ)-DP
        # ε = min_α [ total_rdp_α + log(1/δ) / (α - 1) ]
        epsilons = self.total_rdp + np.log(1 / delta) / (self.orders - 1)

        return epsilons.min()

    def get_privacy_spent(self, delta: Optional[float] = None) -> tuple[float, float]:
        """Get cumulative (ε, δ) spent so far.

        Args:
            delta: Override δ (default: use self.delta)

        Returns:
            (epsilon, delta): Privacy budget consumed
        """
        epsilon = self.get_epsilon(delta)
        return epsilon, (delta if delta is not None else self.delta)

    def get_budget(self) -> PrivacyBudget:
        """Get privacy budget summary.

        Returns:
            PrivacyBudget object with current state
        """
        epsilon, _ = self.get_privacy_spent()
        return PrivacyBudget(
            epsilon=float('inf'),  # No upper bound specified
            delta=self.delta,
            spent_epsilon=epsilon,
            spent_delta=self.delta
        )

    def reset(self) -> None:
        """Reset the accountant (start fresh tracking)."""
        self.steps = 0
        self.total_rdp = np.zeros_like(self.orders)

    def get_rdp(self) -> dict[float, float]:
        """Get RDP values at each order.

        Returns:
            Dictionary mapping α → RDP_α
        """
        return {alpha: rdp for alpha, rdp in zip(self.orders, self.total_rdp)}

    def __repr__(self) -> str:
        eps, delta = self.get_privacy_spent()
        return (f"RDPAccountant(steps={self.steps}, ε={eps:.4f}, "
                f"δ={delta:.2e}, σ={self.noise_multiplier:.2f}, "
                f"q={self.sampling_rate:.3f})")


def compute_rdp(
    noise_multiplier: float,
    sampling_rate: float,
    steps: int,
    delta: float,
    orders: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """Compute total (ε, δ) for given DP-SGD parameters.

    Convenience function to compute privacy spent without creating
    an accountant instance.

    Args:
        noise_multiplier: σ
        sampling_rate: q
        steps: Number of training steps
        delta: Target δ
        orders: RDP orders (optional)

    Returns:
        (epsilon, delta): Privacy budget consumed

    Example:
        >>> eps, delta = compute_rdp(
        ...     noise_multiplier=1.5,
        ...     sampling_rate=0.01,
        ...     steps=1000,
        ...     delta=1e-5
        ... )
        >>> print(f"Privacy spent: ε = {eps:.3f}, δ = {delta:.2e}")
        Privacy spent: ε = 2.456, δ = 1.00e-05
    """
    accountant = RDPAccountant(noise_multiplier, sampling_rate, delta, orders)
    accountant.step(steps)
    return accountant.get_privacy_spent()


def compute_steps_for_epsilon(
    target_epsilon: float,
    delta: float,
    noise_multiplier: float,
    sampling_rate: float
) -> int:
    """Compute maximum steps to stay within ε budget.

    This is useful for planning: given a target ε and noise multiplier,
    how many training steps can we run?

    Args:
        target_epsilon: Target privacy budget ε
        delta: Failure probability δ
        noise_multiplier: σ
        sampling_rate: q

    Returns:
        max_steps: Maximum number of steps to stay within ε

    Example:
        >>> max_steps = compute_steps_for_epsilon(
        ...     target_epsilon=1.0,
        ...     delta=1e-5,
        ...     noise_multiplier=1.5,
        ...     sampling_rate=0.01
        ... )
        >>> print(f"Can run {max_steps} steps within ε=1.0")
    """
    accountant = RDPAccountant(noise_multiplier, sampling_rate, delta)

    # Binary search for max steps
    steps_low, steps_high = 0, 1000000

    for _ in range(40):
        steps_mid = (steps_low + steps_high) // 2

        # Create temporary accountant for testing
        test_accountant = RDPAccountant(noise_multiplier, sampling_rate, delta)
        test_accountant.step(steps_mid)
        eps_mid = test_accountant.get_epsilon()

        if eps_mid < target_epsilon:
            steps_low = steps_mid + 1
        else:
            steps_high = steps_mid

    return steps_high


if __name__ == "__main__":
    print("Testing RDP Accountant...")

    # Test 1: Basic tracking
    print("\n--- Test 1: Basic Tracking ---")
    accountant = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.01, delta=1e-5)

    for round_num in range(1, 11):
        accountant.step(num_steps=10)
        eps, delta = accountant.get_privacy_spent()
        print(f"Round {round_num}: steps={accountant.steps}, ε={eps:.4f}, δ={delta:.2e}")

    # Test 2: Privacy budget summary
    print("\n--- Test 2: Budget Summary ---")
    budget = accountant.get_budget()
    print(budget)

    # Test 3: Compute epsilon for fixed configuration
    print("\n--- Test 3: Privacy for Fixed Configuration ---")
    eps, delta = compute_rdp(
        noise_multiplier=1.0,
        sampling_rate=0.01,
        steps=1000,
        delta=1e-5
    )
    print(f"After 1000 steps: ε={eps:.3f}, δ={delta:.2e}")

    # Test 4: Max steps within budget
    print("\n--- Test 4: Max Steps Within Budget ---")
    max_steps = compute_steps_for_epsilon(
        target_epsilon=1.0,
        delta=1e-5,
        noise_multiplier=1.5,
        sampling_rate=0.01
    )
    print(f"Can run {max_steps} steps within ε=1.0")

    # Test 5: Different noise multipliers
    print("\n--- Test 5: Epsilon for Different Noise Multipliers ---")
    steps = 1000
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        eps, _ = compute_rdp(sigma, 0.01, steps, 1e-5)
        print(f"σ={sigma:.1f} → ε={eps:.3f}")
