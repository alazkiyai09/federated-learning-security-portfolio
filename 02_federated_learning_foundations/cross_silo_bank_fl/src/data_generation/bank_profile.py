"""
Bank profile definitions for realistic federated learning simulation.
Each bank has unique characteristics based on real-world banking patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import yaml
from pathlib import Path


@dataclass
class BankProfile:
    """
    Realistic bank profile for federated fraud detection simulation.

    Attributes:
        name: Display name of the bank
        bank_type: Type of banking institution
        n_clients: Total customer base size
        age_distribution: Mean and std of customer age
        income_distribution: Mean and std of customer income
        daily_transactions: Average daily transaction volume
        transaction_amount: Transaction amount statistics
        fraud_rate: Base fraud rate (proportion)
        fraud_types: Distribution of different fraud types
        merchant_distribution: Distribution across merchant categories
        regions: Geographic regions served
        international_ratio: Proportion of international transactions
        label_quality: Accuracy of fraud labels (0-1)
        feature_completeness: Proportion of complete features (0-1)
    """
    # Identity
    bank_id: str
    name: str
    bank_type: Literal["retail", "regional", "digital", "credit_union", "international"]

    # Customer demographics
    n_clients: int
    age_distribution: Tuple[float, float]  # (mean, std)
    income_distribution: Tuple[float, float]  # (mean, std)

    # Transaction characteristics
    daily_transactions: int
    transaction_amount: Dict[str, float]  # {mean, std, min, max}

    # Fraud characteristics
    fraud_rate: float
    fraud_types: Dict[str, float]  # Fraud type distribution

    # Transaction patterns
    merchant_distribution: Dict[str, float]  # Merchant category distribution
    regions: List[str]
    international_ratio: float

    # Data quality
    label_quality: float = 0.90
    feature_completeness: float = 0.95

    @property
    def total_transactions(self) -> int:
        """Calculate total transactions for simulation period."""
        return self.daily_transactions * 30  # 30 days

    @property
    def expected_fraud_count(self) -> int:
        """Expected number of fraudulent transactions."""
        return int(self.total_transactions * self.fraud_rate)

    def __repr__(self) -> str:
        return f"BankProfile({self.bank_id}: {self.name}, {self.bank_type}, " \
               f"{self.n_clients:,} clients, {self.daily_transactions:,} tx/day)"


def load_bank_profiles(config_path: str = None) -> Dict[str, BankProfile]:
    """
    Load bank profiles from YAML configuration file.

    Args:
        config_path: Path to bank_profiles.yaml. If None, uses default path.

    Returns:
        Dictionary mapping bank_id to BankProfile objects
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "bank_profiles.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract label quality from config
    label_quality = config.get('label_quality', {})
    feature_completeness = config.get('data_quality', {}).get('feature_completeness', {})

    profiles = {}
    for bank_id, bank_config in config['banks'].items():
        profiles[bank_id] = BankProfile(
            bank_id=bank_id,
            name=bank_config['name'],
            bank_type=bank_config['bank_type'],
            n_clients=bank_config['n_clients'],
            age_distribution=(
                bank_config['age_distribution']['mean'],
                bank_config['age_distribution']['std']
            ),
            income_distribution=(
                bank_config['income_distribution']['mean'],
                bank_config['income_distribution']['std']
            ),
            daily_transactions=bank_config['daily_transactions'],
            transaction_amount=bank_config['transaction_amount'],
            fraud_rate=bank_config['fraud_rate'],
            fraud_types=bank_config['fraud_types'],
            merchant_distribution=bank_config['merchant_distribution'],
            regions=bank_config['regions'],
            international_ratio=bank_config['international_ratio'],
            label_quality=label_quality.get(bank_id, 0.90),
            feature_completeness=feature_completeness.get(bank_id, 0.95)
        )

    return profiles


def get_bank_profiles(config_path: str = None) -> List[BankProfile]:
    """
    Get list of bank profiles for simulation.

    Args:
        config_path: Optional path to config file

    Returns:
        List of BankProfile objects
    """
    profiles_dict = load_bank_profiles(config_path)
    return list(profiles_dict.values())


def get_summary_statistics(profiles: List[BankProfile]) -> Dict:
    """
    Calculate summary statistics across all banks.

    Args:
        profiles: List of BankProfile objects

    Returns:
        Dictionary with aggregated statistics
    """
    total_clients = sum(p.n_clients for p in profiles)
    total_daily_tx = sum(p.daily_transactions for p in profiles)
    avg_fraud_rate = sum(p.fraud_rate for p in profiles) / len(profiles)

    return {
        "n_banks": len(profiles),
        "total_clients": total_clients,
        "total_daily_transactions": total_daily_tx,
        "average_fraud_rate": avg_fraud_rate,
        "bank_types": {p.bank_type: p.name for p in profiles}
    }


# Preset profiles for quick access (can be overridden by config)
DEFAULT_PROFILES = {
    "Bank_A": {
        "name": "Large Retail Bank",
        "bank_type": "retail",
        "n_clients": 500000,
        "age_distribution": (45, 15),
        "income_distribution": (65000, 35000),
        "daily_transactions": 150000,
        "transaction_amount": {"mean": 75, "std": 150, "min": 1, "max": 5000},
        "fraud_rate": 0.0025,
        "fraud_types": {"card_present": 0.35, "card_not_present": 0.65},
        "merchant_distribution": {"retail": 0.35, "groceries": 0.20, "restaurants": 0.15, "gas_stations": 0.12, "online": 0.18},
        "regions": ["Northeast", "Midwest", "South", "West"],
        "international_ratio": 0.08,
    },
    "Bank_B": {
        "name": "Regional Bank",
        "bank_type": "regional",
        "n_clients": 80000,
        "age_distribution": (52, 12),
        "income_distribution": (55000, 25000),
        "daily_transactions": 25000,
        "transaction_amount": {"mean": 85, "std": 120, "min": 5, "max": 3000},
        "fraud_rate": 0.0018,
        "fraud_types": {"card_present": 0.50, "card_not_present": 0.50},
        "merchant_distribution": {"retail": 0.30, "groceries": 0.25, "restaurants": 0.18, "gas_stations": 0.15, "online": 0.12},
        "regions": ["Midwest"],
        "international_ratio": 0.03,
    },
    "Bank_C": {
        "name": "Digital-Only Bank",
        "bank_type": "digital",
        "n_clients": 120000,
        "age_distribution": (32, 10),
        "income_distribution": (48000, 20000),
        "daily_transactions": 45000,
        "transaction_amount": {"mean": 45, "std": 80, "min": 1, "max": 1500},
        "fraud_rate": 0.0080,
        "fraud_types": {"card_present": 0.05, "card_not_present": 0.95, "synthetic_identity": 0.40},
        "merchant_distribution": {"retail": 0.10, "groceries": 0.08, "restaurants": 0.12, "online": 0.70},
        "regions": ["West", "Northeast"],
        "international_ratio": 0.05,
    },
    "Bank_D": {
        "name": "Credit Union",
        "bank_type": "credit_union",
        "n_clients": 35000,
        "age_distribution": (48, 14),
        "income_distribution": (52000, 22000),
        "daily_transactions": 8000,
        "transaction_amount": {"mean": 95, "std": 100, "min": 10, "max": 2000},
        "fraud_rate": 0.0010,
        "fraud_types": {"card_present": 0.60, "card_not_present": 0.40},
        "merchant_distribution": {"retail": 0.25, "groceries": 0.30, "restaurants": 0.20, "gas_stations": 0.15, "online": 0.10},
        "regions": ["South", "Midwest"],
        "international_ratio": 0.02,
    },
    "Bank_E": {
        "name": "International Bank",
        "bank_type": "international",
        "n_clients": 300000,
        "age_distribution": (42, 16),
        "income_distribution": (85000, 45000),
        "daily_transactions": 90000,
        "transaction_amount": {"mean": 150, "std": 250, "min": 5, "max": 10000},
        "fraud_rate": 0.0035,
        "fraud_types": {"card_present": 0.25, "card_not_present": 0.75, "cross_border_fraud": 0.30},
        "merchant_distribution": {"retail": 0.20, "groceries": 0.15, "restaurants": 0.15, "travel": 0.20, "luxury": 0.10, "online": 0.20},
        "regions": ["Northeast", "West", "International"],
        "international_ratio": 0.35,
    }
}
