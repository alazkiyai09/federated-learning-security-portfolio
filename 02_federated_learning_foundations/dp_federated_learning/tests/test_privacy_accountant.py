"""
Unit Tests for Privacy Accountant
==================================

Tests to verify RDP accounting correctly tracks privacy consumption.
"""

import pytest
import numpy as np

from src.dp_mechanisms.privacy_accountant import (
    RDPAccountant,
    PrivacyBudget,
    compute_rdp,
    compute_steps_for_epsilon
)


class TestRDPAccountantInit:
    """Test RDPAccountant initialization."""

    def test_initialization(self):
        """Verify accountant initializes correctly."""
        accountant = RDPAccountant(
            noise_multiplier=1.5,
            sampling_rate=0.01,
            delta=1e-5
        )

        assert accountant.noise_multiplier == 1.5
        assert accountant.sampling_rate == 0.01
        assert accountant.delta == 1e-5
        assert accountant.steps == 0
        assert accountant.total_rdp.sum() == 0

    def test_invalid_noise_multiplier(self):
        """Verify invalid noise multipliers raise errors."""
        with pytest.raises(ValueError, match="Noise multiplier must be positive"):
            RDPAccountant(noise_multiplier=0, sampling_rate=0.01, delta=1e-5)

        with pytest.raises(ValueError, match="Noise multiplier must be positive"):
            RDPAccountant(noise_multiplier=-1.0, sampling_rate=0.01, delta=1e-5)

    def test_invalid_sampling_rate(self):
        """Verify invalid sampling rates raise errors."""
        with pytest.raises(ValueError, match="Sampling rate must be in \\(0, 1\\]"):
            RDPAccountant(noise_multiplier=1.5, sampling_rate=0, delta=1e-5)

        with pytest.raises(ValueError, match="Sampling rate must be in \\(0, 1\\]"):
            RDPAccountant(noise_multiplier=1.5, sampling_rate=1.5, delta=1e-5)

    def test_invalid_delta(self):
        """Verify invalid delta values raise errors."""
        with pytest.raises(ValueError, match="Delta must be in \\(0, 1\\)"):
            RDPAccountant(noise_multiplier=1.5, sampling_rate=0.01, delta=0)

        with pytest.raises(ValueError, match="Delta must be in \\(0, 1\\)"):
            RDPAccountant(noise_multiplier=1.5, sampling_rate=0.01, delta=1.0)


class TestPrivacyTracking:
    """Test privacy consumption tracking."""

    def test_step_increments_counter(self):
        """Verify step() increments the step counter."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        accountant.step()
        assert accountant.steps == 1

        accountant.step(num_steps=10)
        assert accountant.steps == 11

    def test_step_accumulates_rdp(self):
        """Verify step() accumulates RDP."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        initial_rdp = accountant.total_rdp.copy()
        accountant.step()

        # RDP should increase
        assert (accountant.total_rdp > initial_rdp).all(), "RDP did not increase"

    def test_epsilon_increases_with_steps(self):
        """Verify ε increases with more steps."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        eps1 = accountant.get_epsilon()
        accountant.step(num_steps=100)
        eps2 = accountant.get_epsilon()

        assert eps2 > eps1, f"ε did not increase: {eps1:.4f} → {eps2:.4f}"

    def test_epsilon_monotonic(self):
        """Verify ε is monotonically increasing."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        previous_eps = 0.0
        for _ in range(10):
            accountant.step(num_steps=50)
            current_eps = accountant.get_epsilon()
            assert current_eps > previous_eps, \
                f"ε not monotonic: {previous_eps:.4f} → {current_eps:.4f}"
            previous_eps = current_eps


class TestNoiseMultiplierImpact:
    """Test how noise multiplier affects privacy consumption."""

    def test_higher_noise_less_privacy_loss(self):
        """Verify higher noise multiplier gives less privacy loss."""
        acc_low_noise = RDPAccountant(noise_multiplier=0.5, sampling_rate=0.01, delta=1e-5)
        acc_high_noise = RDPAccountant(noise_multiplier=2.0, sampling_rate=0.01, delta=1e-5)

        # Run same number of steps
        acc_low_noise.step(num_steps=1000)
        acc_high_noise.step(num_steps=1000)

        eps_low_noise = acc_low_noise.get_epsilon()
        eps_high_noise = acc_high_noise.get_epsilon()

        assert eps_low_noise > eps_high_noise, \
            f"Lower noise should consume more privacy: ε(σ=0.5)={eps_low_noise:.2f}, ε(σ=2.0)={eps_high_noise:.2f}"

    def test_noise_multiplier_scaling(self):
        """Verify how ε scales with σ."""
        steps = 1000
        sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
        epsilons = []

        for sigma in sigmas:
            acc = RDPAccountant(sigma, 0.01, 1e-5)
            acc.step(steps)
            epsilons.append(acc.get_epsilon())

        # ε should decrease as σ increases
        for i in range(len(sigmas) - 1):
            assert epsilons[i] > epsilons[i+1], \
                f"ε not decreasing with σ: σ={sigmas[i]:.1f} → ε={epsilons[i]:.2f}, σ={sigmas[i+1]:.1f} → ε={epsilons[i+1]:.2f}"


class TestSamplingRateImpact:
    """Test how sampling rate affects privacy consumption."""

    def test_higher_sampling_rate_more_privacy_loss(self):
        """Verify higher sampling rate gives more privacy loss."""
        acc_low_q = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.001, delta=1e-5)
        acc_high_q = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.1, delta=1e-5)

        acc_low_q.step(num_steps=1000)
        acc_high_q.step(num_steps=1000)

        eps_low_q = acc_low_q.get_epsilon()
        eps_high_q = acc_high_q.get_epsilon()

        assert eps_low_q < eps_high_q, \
            f"Lower sampling rate should consume less privacy: ε(q=0.001)={eps_low_q:.2f}, ε(q=0.1)={eps_high_q:.2f}"

    def test_sampling_rate_quadratic_scaling(self):
        """Verify ε scales quadratically with sampling rate."""
        # ε ∝ q² (approximately)
        acc_q1 = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.01, delta=1e-5)
        acc_q2 = RDPAccountant(noise_multiplier=1.5, sampling_rate=0.02, delta=1e-5)

        acc_q1.step(num_steps=1000)
        acc_q2.step(num_steps=1000)

        eps_q1 = acc_q1.get_epsilon()
        eps_q2 = acc_q2.get_epsilon()

        # q2 = 2 * q1, so ε should be > 2x (ideally 4x, but δ affects optimal α)
        ratio = eps_q2 / eps_q1
        assert ratio > 2.0, \
            f"ε ratio too small: {ratio:.2f}, expected >2.0 (ideally ~4.0)"


class TestResetFunctionality:
    """Test reset functionality."""

    def test_reset_clears_history(self):
        """Verify reset() clears accumulated privacy."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        accountant.step(num_steps=100)
        assert accountant.steps == 100
        assert accountant.get_epsilon() > 0

        accountant.reset()

        assert accountant.steps == 0
        assert accountant.total_rdp.sum() == 0
        assert accountant.get_epsilon() < 0.1  # Very small (just from δ)


class TestPrivacyBudget:
    """Test PrivacyBudget dataclass."""

    def test_budget_remaining(self):
        """Verify remaining_epsilon calculation."""
        budget = PrivacyBudget(
            epsilon=10.0,
            delta=1e-5,
            spent_epsilon=3.0
        )

        assert budget.remaining_epsilon == 7.0

    def test_utilization_calculation(self):
        """Verify utilization percentage."""
        budget = PrivacyBudget(
            epsilon=10.0,
            delta=1e-5,
            spent_epsilon=3.0
        )

        assert abs(budget.utilization - 0.3) < 1e-6

    def test_utilization_capped_at_one(self):
        """Verify utilization is capped at 100%."""
        budget = PrivacyBudget(
            epsilon=5.0,
            delta=1e-5,
            spent_epsilon=10.0  # Overspent
        )

        assert budget.utilization == 1.0

    def test_get_budget(self):
        """Verify get_budget() returns correct summary."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)
        accountant.step(num_steps=100)

        budget = accountant.get_budget()

        assert budget.spent_epsilon > 0
        assert budget.spent_delta == 1e-5


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_rdp(self):
        """Verify compute_rdp() returns correct values."""
        epsilon, delta = compute_rdp(
            noise_multiplier=1.5,
            sampling_rate=0.01,
            steps=1000,
            delta=1e-5
        )

        assert epsilon > 0
        assert delta == 1e-5

    def test_compute_steps_for_epsilon(self):
        """Verify compute_steps_for_epsilon() returns reasonable values."""
        max_steps = compute_steps_for_epsilon(
            target_epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.5,
            sampling_rate=0.01
        )

        assert max_steps > 0
        assert max_steps < 1000000  # Should be less than our upper bound

    def test_steps_decrease_with_epsilon(self):
        """Verify stricter ε budget allows fewer steps."""
        steps_strict = compute_steps_for_epsilon(
            target_epsilon=0.5,
            delta=1e-5,
            noise_multiplier=1.5,
            sampling_rate=0.01
        )
        steps_relaxed = compute_steps_for_epsilon(
            target_epsilon=5.0,
            delta=1e-5,
            noise_multiplier=1.5,
            sampling_rate=0.01
        )

        assert steps_strict < steps_relaxed, \
            f"Stricter ε should allow fewer steps: {steps_strict} vs {steps_relaxed}"


class TestRDPOrders:
    """Test RDP order functionality."""

    def test_custom_orders(self):
        """Verify custom RDP orders work."""
        custom_orders = np.array([2.0, 4.0, 8.0, 16.0])
        accountant = RDPAccountant(
            noise_multiplier=1.5,
            sampling_rate=0.01,
            delta=1e-5,
            orders=custom_orders
        )

        accountant.step(num_steps=100)
        rdp_dict = accountant.get_rdp()

        assert len(rdp_dict) == len(custom_orders)
        assert all(alpha in rdp_dict for alpha in custom_orders)

    def test_default_orders(self):
        """Verify default orders are reasonable."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        # Should have many orders
        assert len(accountant.orders) > 10

        # Orders should be in range (1, inf)
        assert (accountant.orders > 1).all()
        assert (accountant.orders < 1000).all()


class TestNumericalStability:
    """Test numerical stability."""

    def test_very_small_noise(self):
        """Verify behavior with very small noise multiplier."""
        accountant = RDPAccountant(
            noise_multiplier=0.1,
            sampling_rate=0.01,
            delta=1e-5
        )

        accountant.step(num_steps=10)

        # Should still compute epsilon (very large)
        epsilon = accountant.get_epsilon()
        assert epsilon > 0
        assert not np.isinf(epsilon)

    def test_very_large_noise(self):
        """Verify behavior with very large noise multiplier."""
        accountant = RDPAccountant(
            noise_multiplier=100.0,
            sampling_rate=0.01,
            delta=1e-5
        )

        accountant.step(num_steps=1000)

        # Should compute very small epsilon
        epsilon = accountant.get_epsilon()
        assert epsilon >= 0
        assert epsilon < 0.1  # Very private

    def test_many_steps(self):
        """Verify behavior with many steps."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)

        accountant.step(num_steps=1000000)

        epsilon = accountant.get_epsilon()
        assert not np.isinf(epsilon)
        assert not np.isnan(epsilon)


class TestRepr:
    """Test string representations."""

    def test_accountant_repr(self):
        """Verify __repr__ is informative."""
        accountant = RDPAccountant(1.5, 0.01, 1e-5)
        accountant.step(num_steps=100)

        repr_str = repr(accountant)

        assert "RDPAccountant" in repr_str
        assert "steps=100" in repr_str
        assert "σ=" in repr_str

    def test_budget_repr(self):
        """Verify PrivacyBudget __repr__ is informative."""
        budget = PrivacyBudget(
            epsilon=10.0,
            delta=1e-5,
            spent_epsilon=3.0
        )

        repr_str = repr(budget)

        assert "PrivacyBudget" in repr_str
        assert "3.0000" in repr_str or "3.00" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
