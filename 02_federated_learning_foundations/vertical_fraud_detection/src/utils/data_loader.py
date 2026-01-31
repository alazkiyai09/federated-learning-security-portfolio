"""
Data generation and loading utilities for Vertical Federated Learning.

Generates synthetic fraud detection data split between two parties:
- Party A: Transaction features (amount, frequency, time patterns)
- Party B: Credit features (credit score, account age, income bracket)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split


def generate_synthetic_data(
    num_samples: int = 100000,
    fraud_ratio: float = 0.05,
    random_seed: int = 42,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic fraud detection data split between two parties.

    Party A Features (Transaction):
        - transaction_amount: Log-normal distribution
        - transaction_count_7d: Count in last 7 days
        - transaction_count_30d: Count in last 30 days
        - avg_amount_7d: Average amount in 7 days
        - time_since_last: Hours since last transaction
        - hour_of_day: Transaction hour (0-23)
        - day_of_week: Day of week (0-6)

    Party B Features (Credit):
        - credit_score: Credit score (300-850)
        - account_age_days: Account age in days
        - income_bracket: Income category (1-5)

    Args:
        num_samples: Total number of samples to generate
        fraud_ratio: Proportion of fraudulent transactions
        random_seed: Random seed for reproducibility
        save_path: If provided, save data to this directory

    Returns:
        Tuple of (party_a_df, party_b_df, labels_df)
    """
    np.random.seed(random_seed)

    # Generate user IDs
    user_ids = [f"user_{i:06d}" for i in range(num_samples)]

    # Generate fraud labels (imbalanced)
    is_fraud = np.random.random(num_samples) < fraud_ratio

    # ===== Party A: Transaction Features =====
    transaction_amount = np.zeros(num_samples)
    transaction_count_7d = np.zeros(num_samples)
    transaction_count_30d = np.zeros(num_samples)
    avg_amount_7d = np.zeros(num_samples)
    time_since_last = np.zeros(num_samples)
    hour_of_day = np.zeros(num_samples, dtype=int)
    day_of_week = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        fraud = is_fraud[i]

        if fraud:
            # Fraudulent transactions: higher amounts, unusual patterns
            transaction_amount[i] = np.random.lognormal(5.5, 1.2)  # Higher avg
            transaction_count_7d[i] = np.random.poisson(3)
            transaction_count_30d[i] = np.random.poisson(15)
            avg_amount_7d[i] = np.random.lognormal(5.0, 1.0)
            time_since_last[i] = np.random.exponential(48)  # More sporadic
            hour_of_day[i] = np.random.randint(0, 24)
            day_of_week[i] = np.random.randint(0, 7)
        else:
            # Legitimate transactions: normal patterns
            transaction_amount[i] = np.random.lognormal(3.5, 0.8)
            transaction_count_7d[i] = np.random.poisson(8)
            transaction_count_30d[i] = np.random.poisson(40)
            avg_amount_7d[i] = np.random.lognormal(3.5, 0.7)
            time_since_last[i] = np.random.exponential(24)
            hour_of_day[i] = np.random.randint(6, 22)  # Business hours biased
            day_of_week[i] = np.random.randint(0, 7)

    party_a_df = pd.DataFrame({
        'user_id': user_ids,
        'transaction_amount': transaction_amount,
        'transaction_count_7d': transaction_count_7d,
        'transaction_count_30d': transaction_count_30d,
        'avg_amount_7d': avg_amount_7d,
        'time_since_last': time_since_last,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
    })

    # ===== Party B: Credit Features =====
    credit_score = np.zeros(num_samples)
    account_age_days = np.zeros(num_samples)
    income_bracket = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        fraud = is_fraud[i]

        if fraud:
            # Fraud more likely with lower credit scores, newer accounts
            credit_score[i] = np.random.normal(580, 80)
            credit_score[i] = np.clip(credit_score[i], 300, 850)
            account_age_days[i] = np.random.exponential(180)  # Newer accounts
            income_bracket[i] = np.random.randint(1, 4)  # Lower income
        else:
            credit_score[i] = np.random.normal(680, 70)
            credit_score[i] = np.clip(credit_score[i], 300, 850)
            account_age_days[i] = np.random.exponential(730)  # Older accounts
            income_bracket[i] = np.random.randint(2, 6)  # All brackets

    party_b_df = pd.DataFrame({
        'user_id': user_ids,
        'credit_score': credit_score,
        'account_age_days': account_age_days,
        'income_bracket': income_bracket,
    })

    # Labels
    labels_df = pd.DataFrame({
        'user_id': user_ids,
        'is_fraud': is_fraud.astype(int),
    })

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        party_a_df.to_csv(os.path.join(save_path, 'party_a_transactions.csv'), index=False)
        party_b_df.to_csv(os.path.join(save_path, 'party_b_credit.csv'), index=False)
        labels_df.to_csv(os.path.join(save_path, 'labels.csv'), index=False)

    return party_a_df, party_b_df, labels_df


def load_aligned_data(
    data_dir: str,
    split: str = 'train',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Load aligned data from processed directory.

    Args:
        data_dir: Path to processed data directory
        split: One of 'train', 'val', 'test'
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed

    Returns:
        Dictionary with 'X_a', 'X_b', 'y' arrays
    """
    # Load aligned data
    X_a = np.load(os.path.join(data_dir, 'aligned_party_a.npy'))
    X_b = np.load(os.path.join(data_dir, 'aligned_party_b.npy'))
    y = np.load(os.path.join(data_dir, 'aligned_labels.npy'))

    # Create splits
    total = len(y)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    if split == 'train':
        return {'X_a': X_a[:train_end], 'X_b': X_b[:train_end], 'y': y[:train_end]}
    elif split == 'val':
        return {'X_a': X_a[train_end:val_end], 'X_b': X_b[train_end:val_end], 'y': y[train_end:val_end]}
    elif split == 'test':
        return {'X_a': X_a[val_end:], 'X_b': X_b[val_end:], 'y': y[val_end:]}
    else:
        raise ValueError(f"Invalid split: {split}")


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> None:
    """
    Create train/val/test splits and save to processed directory.

    Args:
        data_dir: Path to raw data directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
    """
    # Load raw data
    party_a = pd.read_csv(os.path.join(data_dir, 'raw', 'party_a_transactions.csv'))
    party_b = pd.read_csv(os.path.join(data_dir, 'raw', 'party_b_credit.csv'))
    labels = pd.read_csv(os.path.join(data_dir, 'raw', 'labels.csv'))

    # Merge on user_id (already aligned)
    merged = party_a.merge(party_b, on='user_id').merge(labels, on='user_id')

    # Extract features and labels
    feature_cols_a = ['transaction_amount', 'transaction_count_7d', 'transaction_count_30d',
                      'avg_amount_7d', 'time_since_last', 'hour_of_day', 'day_of_week']
    feature_cols_b = ['credit_score', 'account_age_days', 'income_bracket']

    X_a = merged[feature_cols_a].values
    X_b = merged[feature_cols_b].values
    y = merged['is_fraud'].values

    # Normalize features
    from sklearn.preprocessing import StandardScaler

    scaler_a = StandardScaler()
    scaler_b = StandardScaler()

    X_a = scaler_a.fit_transform(X_a)
    X_b = scaler_b.fit_transform(X_b)

    # Create splits (stratified by label)
    X_a_train, X_a_temp, y_train, y_temp = train_test_split(
        X_a, y, train_size=train_ratio, random_state=random_seed, stratify=y
    )
    X_b_train, X_b_temp, _, _ = train_test_split(
        X_b, y, train_size=train_ratio, random_state=random_seed, stratify=y
    )

    # Split remaining into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_a_val, X_a_test, y_val, y_test = train_test_split(
        X_a_temp, y_temp, train_size=val_ratio_adjusted, random_state=random_seed, stratify=y_temp
    )
    X_b_val, X_b_test, _, _ = train_test_split(
        X_b_temp, y_temp, train_size=val_ratio_adjusted, random_state=random_seed, stratify=y_temp
    )

    # Combine and save
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Train
    np.save(os.path.join(processed_dir, 'aligned_party_a.npy'), np.concatenate([X_a_train, X_a_val, X_a_test]))
    np.save(os.path.join(processed_dir, 'aligned_party_b.npy'), np.concatenate([X_b_train, X_b_val, X_b_test]))
    np.save(os.path.join(processed_dir, 'aligned_labels.npy'), np.concatenate([y_train, y_val, y_test]))

    # Save split indices
    split_info = {
        'train_end': len(y_train),
        'val_end': len(y_train) + len(y_val),
        'total': len(y_train) + len(y_val) + len(y_test)
    }

    with open(os.path.join(processed_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic fraud detection data...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    raw_dir = os.path.join(data_dir, 'raw')

    party_a, party_b, labels = generate_synthetic_data(
        num_samples=100000,
        fraud_ratio=0.05,
        save_path=raw_dir
    )

    print(f"Party A (Transaction): {party_a.shape}")
    print(f"Party B (Credit): {party_b.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Fraud rate: {labels['is_fraud'].mean():.2%}")

    # Create splits
    print("\nCreating train/val/test splits...")
    create_data_splits(data_dir)
    print("Data saved to:", os.path.join(data_dir, 'processed'))
