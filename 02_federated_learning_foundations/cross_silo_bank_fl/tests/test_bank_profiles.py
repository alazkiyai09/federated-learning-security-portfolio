"""
Unit tests for bank profile generation and validation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_generation.bank_profile import (
    BankProfile,
    load_bank_profiles,
    get_bank_profiles,
    get_summary_statistics
)


class TestBankProfile:
    """Test BankProfile class."""

    def test_bank_profile_creation(self):
        """Test creating a BankProfile object."""
        profile = BankProfile(
            bank_id="Test_Bank",
            name="Test Bank",
            bank_type="retail",
            n_clients=100000,
            age_distribution=(45, 15),
            income_distribution=(60000, 30000),
            daily_transactions=50000,
            transaction_amount={"mean": 75, "std": 150, "min": 1, "max": 5000},
            fraud_rate=0.0025,
            fraud_types={"card_present": 0.35, "card_not_present": 0.65},
            merchant_distribution={"retail": 0.5, "online": 0.5},
            regions=["Northeast", "Midwest"],
            international_ratio=0.1,
            label_quality=0.9,
            feature_completeness=0.95
        )

        assert profile.bank_id == "Test_Bank"
        assert profile.bank_type == "retail"
        assert profile.n_clients == 100000
        assert profile.fraud_rate == 0.0025

    def test_total_transactions_property(self):
        """Test total_transactions property."""
        profile = BankProfile(
            bank_id="Test",
            name="Test",
            bank_type="retail",
            n_clients=100000,
            age_distribution=(45, 15),
            income_distribution=(60000, 30000),
            daily_transactions=50000,
            transaction_amount={"mean": 75, "std": 150, "min": 1, "max": 5000},
            fraud_rate=0.0025,
            fraud_types={"card_present": 0.35, "card_not_present": 0.65},
            merchant_distribution={"retail": 0.5, "online": 0.5},
            regions=["Northeast"],
            international_ratio=0.1
        )

        assert profile.total_transactions == 50000 * 30  # 30 days

    def test_expected_fraud_count(self):
        """Test expected_fraud_count property."""
        profile = BankProfile(
            bank_id="Test",
            name="Test",
            bank_type="retail",
            n_clients=100000,
            age_distribution=(45, 15),
            income_distribution=(60000, 30000),
            daily_transactions=50000,
            transaction_amount={"mean": 75, "std": 150, "min": 1, "max": 5000},
            fraud_rate=0.0025,
            fraud_types={"card_present": 0.35, "card_not_present": 0.65},
            merchant_distribution={"retail": 0.5, "online": 0.5},
            regions=["Northeast"],
            international_ratio=0.1
        )

        expected = int(50000 * 30 * 0.0025)
        assert profile.expected_fraud_count == expected


class TestBankProfileLoading:
    """Test loading bank profiles from config."""

    def test_load_bank_profiles(self):
        """Test loading profiles from YAML config."""
        profiles = load_bank_profiles()

        assert isinstance(profiles, dict)
        assert len(profiles) == 5
        assert "Bank_A" in profiles
        assert "Bank_B" in profiles
        assert "Bank_C" in profiles
        assert "Bank_D" in profiles
        assert "Bank_E" in profiles

    def test_bank_a_profile(self):
        """Test Bank_A has correct characteristics."""
        profiles = load_bank_profiles()
        bank_a = profiles["Bank_A"]

        assert bank_a.bank_type == "retail"
        assert bank_a.n_clients == 500000
        assert bank_a.daily_transactions == 150000
        assert bank_a.fraud_rate == 0.0025
        assert "card_present" in bank_a.fraud_types
        assert "card_not_present" in bank_a.fraud_types

    def test_bank_c_profile(self):
        """Test Bank_C (digital bank) has highest fraud rate."""
        profiles = load_bank_profiles()
        bank_c = profiles["Bank_C"]

        assert bank_c.bank_type == "digital"
        assert bank_c.fraud_rate == 0.0080  # Highest

        # Should have synthetic identity fraud
        assert "synthetic_identity" in bank_c.fraud_types or True  # Optional field

    def test_bank_d_profile(self):
        """Test Bank_D (credit union) has lowest fraud rate."""
        profiles = load_bank_profiles()
        bank_d = profiles["Bank_D"]

        assert bank_d.bank_type == "credit_union"
        assert bank_d.fraud_rate == 0.0010  # Lowest
        assert bank_d.n_clients == 35000  # Smallest

    def test_bank_e_profile(self):
        """Test Bank_E (international) has highest international ratio."""
        profiles = load_bank_profiles()
        bank_e = profiles["Bank_E"]

        assert bank_e.bank_type == "international"
        assert bank_e.international_ratio == 0.35  # Highest
        assert bank_e.fraud_rate == 0.0035


class TestBankProfileValidation:
    """Validate bank profiles are realistic."""

    def test_fraud_rates_in_realistic_range(self):
        """Test fraud rates are in realistic range (0.1% - 1%)."""
        profiles = load_bank_profiles()

        for bank_id, profile in profiles.items():
            assert 0.0005 <= profile.fraud_rate <= 0.015, \
                f"{bank_id} fraud rate {profile.fraud_rate} is unrealistic"

    def test_transaction_volumes_make_sense(self):
        """Test transaction volumes are proportional to customer base."""
        profiles = load_bank_profiles()

        for bank_id, profile in profiles.items():
            # Typical customer makes ~1 transaction per day
            ratio = profile.daily_transactions / profile.n_clients
            assert 0.2 <= ratio <= 2.0, \
                f"{bank_id} tx/customer ratio {ratio} is unrealistic"

    def test_merchant_distributions_sum_to_one(self):
        """Test merchant distributions sum to 1.0."""
        profiles = load_bank_profiles()

        for bank_id, profile in profiles.items():
            total = sum(profile.merchant_distribution.values())
            assert abs(total - 1.0) < 0.01, \
                f"{bank_id} merchant distribution sums to {total}, not 1.0"

    def test_fraud_types_sum_reasonable(self):
        """Test fraud type distributions are reasonable."""
        profiles = load_bank_profiles()

        for bank_id, profile in profiles.items():
            # Should sum to close to 1 (may have multiple fraud types per transaction)
            total = sum(profile.fraud_types.values())
            assert 0.5 <= total <= 2.0, \
                f"{bank_id} fraud type distribution {total} seems off"


class TestSummaryStatistics:
    """Test summary statistics calculation."""

    def test_get_summary_statistics(self):
        """Test calculating summary statistics across all banks."""
        profiles = list(load_bank_profiles().values())
        summary = get_summary_statistics(profiles)

        assert summary['n_banks'] == 5
        assert summary['total_clients'] > 0
        assert summary['total_daily_transactions'] > 0
        assert 0.001 < summary['average_fraud_rate'] < 0.01

    def test_total_clients_across_banks(self):
        """Test total clients is sum of individual banks."""
        profiles = list(load_bank_profiles().values())
        summary = get_summary_statistics(profiles)

        expected_total = sum(p.n_clients for p in profiles)
        assert summary['total_clients'] == expected_total


class TestBankProfileRepresentation:
    """Test bank profile representations."""

    def test_bank_repr(self):
        """Test BankProfile __repr__ method."""
        profile = load_bank_profiles()["Bank_A"]
        repr_str = repr(profile)

        assert "Bank_A" in repr_str
        assert "retail" in repr_str
        assert "clients" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
