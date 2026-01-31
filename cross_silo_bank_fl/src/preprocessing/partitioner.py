"""
Data partitioning for federated learning simulation.
Splits data non-IID across banks and creates train/val/test splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle

from ..data_generation.bank_profile import BankProfile, load_bank_profiles
from ..data_generation.transaction_generator import TransactionGenerator
from ..data_generation.fraud_generator import FraudGenerator


def generate_all_bank_data(
    profiles: List[BankProfile],
    n_days: int = 30,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate transaction data for all banks.

    Args:
        profiles: List of BankProfile objects
        n_days: Number of days of data to generate
        output_dir: Optional directory to save raw data
        seed: Random seed

    Returns:
        Dictionary mapping bank_id to DataFrame
    """
    bank_data = {}

    for profile in profiles:
        print(f"Generating data for {profile.name}...")

        # Set seed for this bank
        bank_seed = seed + hash(profile.bank_id) % 1000

        # Generate transactions
        tx_gen = TransactionGenerator(profile, seed=bank_seed)
        n_tx = profile.total_transactions

        df = tx_gen.generate(
            n_transactions=n_tx,
            n_days=n_days,
            start_date=None
        )

        # Inject fraud
        fraud_gen = FraudGenerator(profile, seed=bank_seed)
        df = fraud_gen.inject_fraud(df)

        # Add bank_id column
        df['bank_id'] = profile.bank_id

        bank_data[profile.bank_id] = df

        # Print statistics
        fraud_stats = fraud_gen.get_fraud_statistics(df)
        print(f"  Generated {len(df):,} transactions")
        print(f"  Fraud rate: {fraud_stats['actual_fraud_rate']:.4f}")
        print(f"  Fraud types: {fraud_stats['fraud_types']}")

        # Save if output directory provided
        if output_dir:
            output_path = Path(output_dir) / f"{profile.bank_id}_raw.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(df, f)
            print(f"  Saved to {output_path}")

    return bank_data


def partition_data_by_bank(
    bank_data: Dict[str, pd.DataFrame],
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Partition data into train/val/test for each bank.

    Args:
        bank_data: Dictionary mapping bank_id to full DataFrame
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        seed: Random seed

    Returns:
        Nested dictionary: bank_id -> split -> DataFrame
    """
    split_data = {}

    for bank_id, df in bank_data.items():
        print(f"Splitting data for {bank_id}...")

        # Shuffle
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Calculate split sizes
        n_total = len(df_shuffled)
        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * val_size)
        n_train = n_total - n_test - n_val

        # Split
        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
        test_df = df_shuffled.iloc[n_train + n_val:].copy()

        split_data[bank_id] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

        # Print fraud rates
        for split_name, split_df in split_data[bank_id].items():
            fraud_rate = split_df['is_fraud'].mean()
            print(f"    {split_name} fraud rate: {fraud_rate:.4f}")

    return split_data


def create_federated_splits(
    split_data: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Optional[str] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Create federated data splits for Flower simulation.

    Args:
        split_data: Nested dictionary from partition_data_by_bank
        output_dir: Optional directory to save splits

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    train_data = {}
    val_data = {}
    test_data = {}

    for bank_id, splits in split_data.items():
        train_data[bank_id] = splits['train']
        val_data[bank_id] = splits['val']
        test_data[bank_id] = splits['test']

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'train_splits.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(output_path / 'val_splits.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        with open(output_path / 'test_splits.pkl', 'wb') as f:
            pickle.dump(test_data, f)

        print(f"Saved federated splits to {output_path}")

    return train_data, val_data, test_data


def create_centralized_dataset(
    bank_data: Dict[str, pd.DataFrame],
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create centralized dataset by pooling all bank data.

    This is the privacy-invasive baseline (upper bound on performance).

    Args:
        bank_data: Dictionary mapping bank_id to DataFrame
        test_size: Proportion for testing
        val_size: Proportion for validation
        seed: Random seed

    Returns:
        Dictionary with train/val/test DataFrames
    """
    # Combine all data
    all_dfs = []
    for bank_id, df in bank_data.items():
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"Combined dataset: {len(combined_df):,} transactions")

    # Shuffle and split
    combined_shuffled = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(combined_shuffled)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)
    n_train = n_total - n_test - n_val

    train_df = combined_shuffled.iloc[:n_train].copy()
    val_df = combined_shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = combined_shuffled.iloc[n_train + n_val:].copy()

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Print fraud rates
    for split_name, split_df in splits.items():
        fraud_rate = split_df['is_fraud'].mean()
        print(f"  {split_name} fraud rate: {fraud_rate:.4f}")

    return splits


def analyze_non_iidness(
    split_data: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Analyze non-IID characteristics across banks.

    Args:
        split_data: Nested dictionary from partition_data_by_bank

    Returns:
        DataFrame with per-bank statistics
    """
    stats = []

    for bank_id, splits in split_data.items():
        train_df = splits['train']

        # Calculate statistics
        n_total = len(train_df)
        n_fraud = train_df['is_fraud'].sum()
        fraud_rate = n_fraud / n_total

        avg_amount = train_df['amount'].mean()
        std_amount = train_df['amount'].std()

        # Fraud type distribution
        fraud_types = train_df[train_df['is_fraud'] == 1]['fraud_type'].value_counts()
        main_fraud_type = fraud_types.index[0] if len(fraud_types) > 0 else 'none'

        # Merchant distribution
        merchant_dist = train_df['merchant_category'].value_counts(normalize=True)
        top_merchant = merchant_dist.index[0] if len(merchant_dist) > 0 else 'unknown'

        # International ratio
        international_ratio = train_df['is_international'].mean()

        stats.append({
            'bank_id': bank_id,
            'n_samples': n_total,
            'n_fraud': n_fraud,
            'fraud_rate': fraud_rate,
            'avg_amount': avg_amount,
            'std_amount': std_amount,
            'main_fraud_type': main_fraud_type,
            'top_merchant': top_merchant,
            'international_ratio': international_ratio
        })

    return pd.DataFrame(stats)


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple train/val/test split for a single DataFrame.

    Args:
        df: Input DataFrame
        test_size: Proportion for testing
        val_size: Proportion of training data for validation
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)
    n_train = n_total - n_test - n_val

    train_df = df_shuffled.iloc[:n_train].copy()
    val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = df_shuffled.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df
