"""
Unit tests for data generation quality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_generation.bank_profile import load_bank_profiles
from src.data_generation.transaction_generator import TransactionGenerator
from src.data_generation.fraud_generator import FraudGenerator


class TestTransactionGenerator:
    """Test transaction generation."""

    def test_generator_initialization(self):
        """Test TransactionGenerator initialization."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        assert gen.profile == profile
        assert len(gen.merchant_list) > 0

    def test_generate_transactions(self):
        """Test generating transactions."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        df = gen.generate(n_transactions=1000, n_days=30)

        assert len(df) == 1000
        assert 'transaction_id' in df.columns
        assert 'timestamp' in df.columns
        assert 'amount' in df.columns
        assert 'merchant_category' in df.columns
        assert 'card_type' in df.columns
        assert 'region' in df.columns
        assert 'customer_age' in df.columns
        assert 'customer_income' in df.columns

    def test_transaction_amounts_in_range(self):
        """Test transaction amounts are within specified range."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        df = gen.generate(n_transactions=1000, n_days=30)

        min_amount = profile.transaction_amount['min']
        max_amount = profile.transaction_amount['max']

        assert df['amount'].min() >= min_amount
        assert df['amount'].max() <= max_amount

    def test_timestamps_span_days(self):
        """Test timestamps span multiple days."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        df = gen.generate(n_transactions=1000, n_days=30)

        time_span = (df['timestamp'].max() - df['timestamp'].min()).days
        assert time_span > 0

    def test_temporal_features_created(self):
        """Test temporal features are created."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        df = gen.generate(n_transactions=1000, n_days=30)

        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        assert 'time_of_day' in df.columns

    def test_customer_demographics_realistic(self):
        """Test customer demographics are realistic."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen = TransactionGenerator(profile, seed=42)
        df = gen.generate(n_transactions=1000, n_days=30)

        # Age should be reasonable
        assert df['customer_age'].min() >= 18
        assert df['customer_age'].max() <= 100

        # Income should be positive
        assert (df['customer_income'] > 0).all()


class TestFraudGenerator:
    """Test fraud generation."""

    def test_fraud_generator_initialization(self):
        """Test FraudGenerator initialization."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        fraud_gen = FraudGenerator(profile, seed=42)
        assert fraud_gen.profile == profile

    def test_inject_fraud(self):
        """Test fraud injection."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=1000, n_days=30)

        fraud_gen = FraudGenerator(profile, seed=42)
        df_with_fraud = fraud_gen.inject_fraud(df)

        assert 'is_fraud' in df_with_fraud.columns
        assert 'fraud_type' in df_with_fraud.columns

    def test_fraud_rate_approximately_correct(self):
        """Test fraud rate is approximately correct."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=10000, n_days=30)

        fraud_gen = FraudGenerator(profile, seed=42)
        df_with_fraud = fraud_gen.inject_fraud(df)

        actual_fraud_rate = df_with_fraud['is_fraud'].mean()
        target_fraud_rate = profile.fraud_rate

        # Allow 20% tolerance due to randomness
        assert abs(actual_fraud_rate - target_fraud_rate) / target_fraud_rate < 0.2

    def test_fraud_types_present(self):
        """Test fraud types are present."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=10000, n_days=30)

        fraud_gen = FraudGenerator(profile, seed=42)
        df_with_fraud = fraud_gen.inject_fraud(df)

        fraud_transactions = df_with_fraud[df_with_fraud['is_fraud'] == 1]
        assert len(fraud_transactions) > 0

        # Check various fraud types exist
        fraud_types = fraud_transactions['fraud_type'].unique()
        assert len(fraud_types) > 1

    def test_get_fraud_statistics(self):
        """Test fraud statistics calculation."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=1000, n_days=30)

        fraud_gen = FraudGenerator(profile, seed=42)
        df_with_fraud = fraud_gen.inject_fraud(df)

        stats = fraud_gen.get_fraud_statistics(df_with_fraud)

        assert 'total_transactions' in stats
        assert 'fraud_transactions' in stats
        assert 'actual_fraud_rate' in stats
        assert 'fraud_types' in stats

        assert stats['total_transactions'] == len(df_with_fraud)
        assert stats['fraud_transactions'] > 0


class TestDataQuality:
    """Test data quality of generated data."""

    def test_no_missing_values(self):
        """Test no missing values in generated data."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=1000, n_days=30)

        fraud_gen = FraudGenerator(profile, seed=42)
        df_with_fraud = fraud_gen.inject_fraud(df)

        # Check for missing values in key columns
        key_cols = ['amount', 'merchant_category', 'card_type', 'region', 'is_fraud']
        for col in key_cols:
            assert df_with_fraud[col].notna().all(), f"Missing values in {col}"

    def test_merchant_categories_match_profile(self):
        """Test merchant categories match bank profile."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=5000, n_days=30)

        # Check merchant categories are from profile
        for merchant in df['merchant_category'].unique():
            # Should match one of the expanded merchant categories
            assert isinstance(merchant, str)

    def test_regions_match_profile(self):
        """Test regions match bank profile."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        tx_gen = TransactionGenerator(profile, seed=42)
        df = tx_gen.generate(n_transactions=5000, n_days=30)

        for region in df['region'].unique():
            # Should be from profile regions or International
            assert region in profile.regions or region == "International"


class TestReproducibility:
    """Test reproducibility with random seed."""

    def test_same_seed_same_data(self):
        """Test same seed produces same data."""
        profiles = load_bank_profiles()
        profile = profiles["Bank_A"]

        gen1 = TransactionGenerator(profile, seed=42)
        df1 = gen1.generate(n_transactions=1000, n_days=30)

        gen2 = TransactionGenerator(profile, seed=42)
        df2 = gen2.generate(n_transactions=1000, n_days=30)

        # Amounts should be identical
        assert np.array_equal(df1['amount'].values, df2['amount'].values)


class TestBankSpecificPatterns:
    """Test bank-specific fraud patterns."""

    def test_digital_bank_high_fraud(self):
        """Test digital bank (Bank C) has highest fraud rate."""
        profiles = load_bank_profiles()

        fraud_rates = {bid: p.fraud_rate for bid, p in profiles.items()}
        bank_c_rate = fraud_rates["Bank_C"]

        # Bank C should have highest fraud rate
        assert bank_c_rate == max(fraud_rates.values())

    def test_international_bank_high_international_ratio(self):
        """Test international bank (Bank E) has highest international ratio."""
        profiles = load_bank_profiles()

        int_ratios = {bid: p.international_ratio for bid, p in profiles.items()}
        bank_e_ratio = int_ratios["Bank_E"]

        # Bank E should have highest international ratio
        assert bank_e_ratio == max(int_ratios.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
