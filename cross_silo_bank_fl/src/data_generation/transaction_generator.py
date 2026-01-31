"""
Generate realistic transaction patterns for different bank types.
Based on real-world banking transaction characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .bank_profile import BankProfile


class TransactionGenerator:
    """
    Generate realistic transaction data for bank fraud detection simulation.

    Creates transaction patterns based on bank characteristics:
    - Transaction amounts (log-normal distribution)
    - Timestamps (temporal patterns)
    - Merchant categories (bank-specific distributions)
    - Geographic locations
    - Customer demographics
    """

    # Merchant category codes (MCC-like)
    MERCHANT_CATEGORIES = {
        "retail": ["department_store", "clothing", "electronics", "home_goods"],
        "groceries": ["supermarket", "grocery", "food_market"],
        "restaurants": ["restaurant", "fast_food", "cafe", "bar"],
        "gas_stations": ["gas_station", "fuel", "service_station"],
        "online": ["online_retail", "digital_goods", "subscription", "marketplace"],
        "travel": ["airline", "hotel", "car_rental", "travel_agency"],
        "luxury": ["jewelry", "luxury_goods", "fine_dining", "spa"]
    }

    # Card types
    CARD_TYPES = ["debit", "credit", "prepaid"]

    # Transaction entry modes
    ENTRY_MODES = ["chip", "contactless", "magnetic_stripe", "online"]

    def __init__(self, profile: BankProfile, seed: Optional[int] = None):
        """
        Initialize transaction generator for a specific bank.

        Args:
            profile: BankProfile with bank characteristics
            seed: Random seed for reproducibility
        """
        self.profile = profile
        self.rng = np.random.RandomState(seed)

        # Expand merchant distribution to specific merchants
        self.merchant_list = self._build_merchant_list()

    def _build_merchant_list(self) -> List[Tuple[str, float]]:
        """Build detailed merchant list with probabilities."""
        merchants = []
        for category, prob in self.profile.merchant_distribution.items():
            if category in self.MERCHANT_CATEGORIES:
                sub_merchants = self.MERCHANT_CATEGORIES[category]
                sub_prob = prob / len(sub_merchants)
                for merchant in sub_merchants:
                    merchants.append((merchant, sub_prob))
        return merchants

    def generate(
        self,
        n_transactions: int,
        n_days: int = 30,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate transaction dataset.

        Args:
            n_transactions: Number of transactions to generate
            n_days: Number of days to span
            start_date: Start date for transactions

        Returns:
            DataFrame with transaction features
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)

        transactions = []

        # Generate transactions
        for i in range(n_transactions):
            tx = self._generate_single_transaction(i, n_days, start_date)
            transactions.append(tx)

        df = pd.DataFrame(transactions)

        # Add derived features
        df = self._add_temporal_features(df)

        return df

    def _generate_single_transaction(
        self,
        idx: int,
        n_days: int,
        start_date: datetime
    ) -> Dict:
        """Generate a single transaction record."""
        # Generate timestamp (more transactions during business hours/days)
        day_offset = self.rng.randint(0, n_days)
        hour = self._sample_hour()

        timestamp = start_date + timedelta(days=day_offset, hours=hour)

        # Sample customer demographics
        customer_age = self.rng.normal(
            self.profile.age_distribution[0],
            self.profile.age_distribution[1]
        )
        customer_age = max(18, min(100, customer_age))  # Clamp to valid range

        customer_income = self.rng.normal(
            self.profile.income_distribution[0],
            self.profile.income_distribution[1]
        )
        customer_income = max(15000, customer_income)  # Minimum income

        # Generate transaction amount (log-normal distribution)
        amount = self.rng.lognormal(
            mean=np.log(self.profile.transaction_amount['mean']),
            sigma=self.profile.transaction_amount['std'] / self.profile.transaction_amount['mean']
        )
        amount = max(
            self.profile.transaction_amount['min'],
            min(self.profile.transaction_amount['max'], amount)
        )

        # Sample merchant
        merchant_probs = [m[1] for m in self.merchant_list]
        merchant = self.merchant_list[
            self.rng.choice(len(self.merchant_list), p=merchant_probs)
        ][0]

        # Sample card type
        card_type = self.rng.choice(self.CARD_TYPES)

        # Sample entry mode based on card type and merchant
        if merchant in self.merchant_list and "online" in str(merchant):
            entry_mode = "online"
        else:
            entry_mode = self.rng.choice(
                self.ENTRY_MODES,
                p=[0.45, 0.35, 0.15, 0.05]  # Probabilities for each mode
            )

        # Sample region
        region = self.rng.choice(self.profile.regions)

        # Determine if international
        is_international = self.rng.random() < self.profile.international_ratio
        if is_international:
            region = "International"

        return {
            "transaction_id": idx,
            "timestamp": timestamp,
            "amount": amount,
            "merchant_category": merchant,
            "card_type": card_type,
            "entry_mode": entry_mode,
            "region": region,
            "customer_age": customer_age,
            "customer_income": customer_income,
            "is_international": 1 if is_international else 0
        }

    def _sample_hour(self) -> int:
        """Sample hour with realistic daily pattern."""
        # More transactions during business hours and evening
        hour_probs = np.array([
            0.01, 0.00, 0.00, 0.00, 0.00, 0.01,  # 0-5
            0.02, 0.03, 0.04, 0.05, 0.06, 0.07,  # 6-11
            0.08, 0.09, 0.08, 0.07, 0.06, 0.06,  # 12-17
            0.08, 0.09, 0.08, 0.06, 0.04, 0.02   # 18-23
        ])
        hour_probs = hour_probs / hour_probs.sum()
        return self.rng.choice(24, p=hour_probs)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from timestamp."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        ).astype(str)

        return df

    def generate_customer_profiles(self, n_customers: int = None) -> pd.DataFrame:
        """
        Generate customer profile dataset.

        Args:
            n_customers: Number of customers to generate. If None, uses n_clients from profile.

        Returns:
            DataFrame with customer demographics
        """
        if n_customers is None:
            n_customers = min(self.profile.n_clients, 10000)  # Limit for memory

        customers = []
        for i in range(n_customers):
            age = self.rng.normal(
                self.profile.age_distribution[0],
                self.profile.age_distribution[1]
            )
            age = max(18, min(100, age))

            income = self.rng.normal(
                self.profile.income_distribution[0],
                self.profile.income_distribution[1]
            )
            income = max(15000, income)

            customers.append({
                "customer_id": i,
                "age": age,
                "income": income,
                "bank_id": self.profile.bank_id
            })

        return pd.DataFrame(customers)
