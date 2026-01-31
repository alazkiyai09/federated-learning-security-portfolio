"""
Generate realistic fraud patterns for different bank types.
Based on real-world fraud patterns in banking industry.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import timedelta
from .bank_profile import BankProfile


class FraudGenerator:
    """
    Inject realistic fraud patterns into transaction data.

    Fraud patterns vary by bank type:
    - Retail bank: Mixed fraud types
    - Regional bank: Local card-present fraud
    - Digital bank: Synthetic identity, account takeover
    - Credit union: Internal/member fraud
    - International bank: Cross-border fraud
    """

    def __init__(self, profile: BankProfile, seed: Optional[int] = None):
        """
        Initialize fraud generator for a specific bank.

        Args:
            profile: BankProfile with bank characteristics
            seed: Random seed for reproducibility
        """
        self.profile = profile
        self.rng = np.random.RandomState(seed)

    def inject_fraud(
        self,
        transactions: pd.DataFrame,
        fraud_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Inject fraud labels into transaction dataset.

        Args:
            transactions: DataFrame of legitimate transactions
            fraud_rate: Override default fraud rate

        Returns:
            DataFrame with is_fraud column and fraud features
        """
        df = transactions.copy()

        if fraud_rate is None:
            fraud_rate = self.profile.fraud_rate

        n_fraud = int(len(df) * fraud_rate)

        # Select random transactions to mark as fraud
        fraud_indices = self.rng.choice(
            len(df),
            size=n_fraud,
            replace=False
        )

        # Initialize fraud column
        df['is_fraud'] = 0
        df['fraud_type'] = 'none'

        # Apply fraud patterns to selected transactions
        for idx in fraud_indices:
            fraud_type = self._sample_fraud_type()
            df = self._apply_fraud_pattern(df, idx, fraud_type)

        # Apply label noise (simulating imperfect labeling)
        df = self._apply_label_noise(df)

        return df

    def _sample_fraud_type(self) -> str:
        """Sample fraud type based on bank's fraud distribution."""
        fraud_types = list(self.profile.fraud_types.keys())
        probs = list(self.profile.fraud_types.values())

        # Normalize probabilities
        probs = np.array(probs) / sum(probs)

        return self.rng.choice(fraud_types, p=probs)

    def _apply_fraud_pattern(
        self,
        df: pd.DataFrame,
        idx: int,
        fraud_type: str
    ) -> pd.DataFrame:
        """
        Apply specific fraud pattern to a transaction.

        Args:
            df: Transaction DataFrame
            idx: Index of transaction to modify
            fraud_type: Type of fraud to inject

        Returns:
            Modified DataFrame
        """
        if fraud_type == "card_present":
            df = self._apply_card_present_fraud(df, idx)
        elif fraud_type == "card_not_present":
            df = self._apply_card_not_present_fraud(df, idx)
        elif fraud_type == "account_takeover":
            df = self._apply_account_takeover_fraud(df, idx)
        elif fraud_type == "synthetic_identity":
            df = self._apply_synthetic_identity_fraud(df, idx)
        elif fraud_type == "cross_border_fraud":
            df = self._apply_cross_border_fraud(df, idx)
        elif fraud_type == "internal_fraud":
            df = self._apply_internal_fraud(df, idx)

        return df

    def _apply_card_present_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply card-present fraud patterns (skimming, cloned cards)."""
        # Mark as fraud
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'card_present'

        # Modify characteristics
        # High amounts at gas stations or ATMs
        if self.rng.random() < 0.4:
            df.at[idx, 'merchant_category'] = 'gas_station'
            df.at[idx, 'amount'] *= 1.8  # Larger amounts

        # Unusual timing (late night)
        if self.rng.random() < 0.3:
            df.at[idx, 'hour'] = self.rng.randint(0, 5)

        # Magnetic stripe entry (skimming)
        if self.rng.random() < 0.6:
            df.at[idx, 'entry_mode'] = 'magnetic_stripe'

        return df

    def _apply_card_not_present_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply card-not-present fraud (online fraud)."""
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'card_not_present'

        # Online transaction
        df.at[idx, 'entry_mode'] = 'online'
        df.at[idx, 'merchant_category'] = 'online_retail'

        # Multiple attempts behavior (would need session tracking, simplified here)
        df.at[idx, 'amount'] *= 1.5

        # Unusual location
        if self.rng.random() < 0.5:
            df.at[idx, 'region'] = 'International'
            df.at[idx, 'is_international'] = 1

        return df

    def _apply_account_takeover_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply account takeover patterns."""
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'account_takeover'

        # Changed behavior patterns
        df.at[idx, 'entry_mode'] = 'online'

        # Unusual merchant for customer profile
        unusual_merchants = ['luxury_goods', 'electronics', 'digital_goods']
        df.at[idx, 'merchant_category'] = self.rng.choice(unusual_merchants)

        # High value transactions
        df.at[idx, 'amount'] *= 2.5

        # Rapid succession indicator (would need time deltas, simplified)
        df.at[idx, 'hour'] = self.rng.randint(2, 6)  # Early morning

        return df

    def _apply_synthetic_identity_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply synthetic identity fraud (unique to digital banks)."""
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'synthetic_identity'

        # New account behavior
        df.at[idx, 'card_type'] = 'credit'

        # Suspicious patterns
        df.at[idx, 'amount'] *= 0.8  # Start small to build trust

        # Online transactions
        df.at[idx, 'entry_mode'] = 'online'
        df.at[idx, 'merchant_category'] = 'online_retail'

        # No established customer pattern
        df.at[idx, 'customer_age'] = self.rng.randint(25, 35)  # Prime synthetic ID age

        return df

    def _apply_cross_border_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply cross-border fraud patterns (unique to international banks)."""
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'cross_border_fraud'

        # International transaction
        df.at[idx, 'region'] = 'International'
        df.at[idx, 'is_international'] = 1

        # Travel-related merchants
        travel_merchants = ['airline', 'hotel', 'car_rental', 'luxury_goods']
        df.at[idx, 'merchant_category'] = self.rng.choice(travel_merchants)

        # High amounts
        df.at[idx, 'amount'] *= 3.0

        # Short time between geographically distant transactions (simplified)
        df.at[idx, 'hour'] = self.rng.randint(0, 24)

        return df

    def _apply_internal_fraud(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Apply internal/member fraud (unique to credit unions)."""
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'internal_fraud'

        # Insider knowledge patterns
        # Just below reporting thresholds
        df.at[idx, 'amount'] = 900 + self.rng.randint(0, 200)

        # Familiar merchants (regular shopping patterns)
        regular_merchants = ['retail', 'groceries', 'restaurants']
        base_category = self.rng.choice(regular_merchants)
        # Find merchant with this category
        matching_merchants = [m for m in df['merchant_category'].values if base_category in str(m)]
        if matching_merchants:
            df.at[idx, 'merchant_category'] = self.rng.choice(matching_merchants)

        # Normal entry modes
        df.at[idx, 'entry_mode'] = self.rng.choice(['chip', 'contactless'])

        return df

    def _apply_label_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label noise to simulate imperfect fraud labeling.

        Some fraud is mislabeled as legitimate (false negatives)
        Some legitimate is mislabeled as fraud (false positives)
        """
        label_quality = self.profile.label_quality

        # Flip some labels
        fraud_indices = df[df['is_fraud'] == 1].index
        n_fraud_to_flip = int(len(fraud_indices) * (1 - label_quality))

        if n_fraud_to_flip > 0:
            indices_to_flip = self.rng.choice(
                fraud_indices,
                size=n_fraud_to_flip,
                replace=False
            )
            df.loc[indices_to_flip, 'is_fraud'] = 0
            df.loc[indices_to_flip, 'fraud_type'] = 'none'

        # Also flip some legitimate to fraud (false positives - rarer)
        legit_indices = df[df['is_fraud'] == 0].index
        n_legit_to_flip = int(len(legit_indices) * (1 - label_quality) * 0.1)

        if n_legit_to_flip > 0:
            indices_to_flip = self.rng.choice(
                legit_indices,
                size=n_legit_to_flip,
                replace=False
            )
            df.loc[indices_to_flip, 'is_fraud'] = 1
            df.loc[indices_to_flip, 'fraud_type'] = 'false_positive'

        return df

    def get_fraud_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate fraud statistics for generated dataset.

        Args:
            df: DataFrame with is_fraud column

        Returns:
            Dictionary with fraud statistics
        """
        n_total = len(df)
        n_fraud = df['is_fraud'].sum()
        actual_fraud_rate = n_fraud / n_total

        fraud_type_counts = df[df['is_fraud'] == 1]['fraud_type'].value_counts().to_dict()

        return {
            "total_transactions": n_total,
            "fraud_transactions": n_fraud,
            "actual_fraud_rate": actual_fraud_rate,
            "target_fraud_rate": self.profile.fraud_rate,
            "fraud_types": fraud_type_counts
        }
