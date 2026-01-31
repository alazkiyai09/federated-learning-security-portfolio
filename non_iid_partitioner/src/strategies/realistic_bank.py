"""
Realistic bank simulation partition strategy.

This simulates a federated learning scenario across multiple banks
with geographical and demographic heterogeneity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils import set_random_state


def realistic_bank_partition(df: pd.DataFrame,
                            n_clients: int,
                            region_col: str = 'region',
                            label_col: str = 'label',
                            feature_cols: Optional[List[str]] = None,
                            balance_within_regions: bool = True,
                            random_state: int = None) -> Dict[int, pd.DataFrame]:
    """
    Partition data across banks using realistic geographic and demographic factors.

    This strategy simulates real-world FL scenarios where:

    1. Banks are located in different geographic regions
    2. Each region has different demographic characteristics
    3. Fraud patterns vary by region (label distribution skew)
    4. Transaction amounts and frequencies vary (feature skew)

    The function can work with:
    - Pre-existing region column in the data
    - Automatic region inference from features (if region_col not provided)

    Args:
        df: Input DataFrame with features and labels
        n_clients: Number of banks (clients)
        region_col: Name of column containing region labels
        label_col: Name of column containing target labels
        feature_cols: List of feature column names (default: all except region/label)
        balance_within_regions: Whether to balance classes within each region
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to DataFrame with that client's data

    Raises:
        ValueError: If required columns are missing
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    rng = set_random_state(random_state)

    # Determine feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns
                       if col not in [region_col, label_col]]

    # Handle region assignment
    if region_col in df.columns:
        # Use existing regions
        unique_regions = df[region_col].unique()
        n_regions = len(unique_regions)
    else:
        # Infer regions from feature space using clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Prepare features for clustering
        X_region = df[feature_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_region)

        # Determine number of regions (max n_clients, but adaptive to data size)
        n_regions = min(n_clients, max(3, df.shape[0] // 100))

        # Cluster to create regions
        kmeans = KMeans(n_clusters=n_regions, random_state=random_state, n_init=10)
        df = df.copy()
        df['_inferred_region'] = kmeans.fit_predict(X_scaled)
        region_col = '_inferred_region'
        unique_regions = np.arange(n_regions)

    # Assign regions to clients
    # If n_regions == n_clients: 1 region per client
    # If n_regions < n_clients: split regions across multiple clients
    # If n_regions > n_clients: merge regions
    partitions = {}

    if n_regions == n_clients:
        # Direct assignment
        region_to_client = {region: i for i, region in enumerate(unique_regions)}
        for region, client_id in region_to_client.items():
            if region_col in df.columns:
                client_data = df[df[region_col] == region]
            else:
                client_data = df[df[region_col] == region].copy()
                client_data = client_data.drop(columns=['_inferred_region'], errors='ignore')
            partitions[client_id] = client_data

    elif n_regions < n_clients:
        # Split regions across multiple clients
        regions_per_client = n_regions // n_clients
        remainder = n_regions % n_clients

        region_list = list(unique_regions)
        region_idx = 0

        for client_id in range(n_clients):
            # Determine how many regions this client gets
            n_client_regions = regions_per_client + (1 if client_id < remainder else 0)

            # Collect data from assigned regions
            client_data_list = []
            for _ in range(n_client_regions):
                if region_idx < len(region_list):
                    region = region_list[region_idx]
                    if region_col in df.columns:
                        region_data = df[df[region_col] == region]
                    else:
                        region_data = df[df[region_col] == region].copy()
                        region_data = region_data.drop(columns=['_inferred_region'], errors='ignore')
                    client_data_list.append(region_data)
                    region_idx += 1

            if client_data_list:
                client_data = pd.concat(client_data_list, ignore_index=True)
                partitions[client_id] = client_data

    else:  # n_regions > n_clients
        # Merge regions into clients
        regions_per_client = n_regions // n_clients
        remainder = n_regions % n_clients

        region_assignments = {}
        region_idx = 0
        for client_id in range(n_clients):
            n_client_regions = regions_per_client + (1 if client_id < remainder else 0)
            client_regions = region_list[region_idx:region_idx + n_client_regions]
            region_assignments[client_id] = client_regions
            region_idx += n_client_regions

        # Create client DataFrames
        for client_id, regions in region_assignments.items():
            client_data_list = []
            for region in regions:
                if region_col in df.columns:
                    region_data = df[df[region_col] == region]
                else:
                    region_data = df[df[region_col] == region].copy()
                    region_data = region_data.drop(columns=['_inferred_region'], errors='ignore')
                client_data_list.append(region_data)

            if client_data_list:
                client_data = pd.concat(client_data_list, ignore_index=True)
                partitions[client_id] = client_data

    # Optional: Balance classes within each client
    if balance_within_regions and label_col in df.columns:
        unique_labels = df[label_col].unique()
        for client_id in partitions:
            client_data = partitions[client_id]

            # Check if class balancing is needed
            label_counts = client_data[label_col].value_counts()
            min_count = label_counts.min()

            if min_count > 0:
                # Sample equal number from each class
                balanced_data_list = []
                for label in unique_labels:
                    label_data = client_data[client_data[label_col] == label]
                    if len(label_data) > min_count:
                        label_data = label_data.sample(n=min_count, random_state=random_state)
                    balanced_data_list.append(label_data)

                partitions[client_id] = pd.concat(balanced_data_list, ignore_index=True)

    # Ensure all clients have data
    for client_id in range(n_clients):
        if client_id not in partitions or len(partitions[client_id]) == 0:
            # Assign random samples to empty clients
            remaining_samples = df[~df.index.isin(
                pd.concat([partitions[c] for c in partitions if c in partitions], ignore_index=True).index
            )]
            if len(remaining_samples) > 0:
                partitions[client_id] = remaining_samples.sample(min(1, len(remaining_samples)))
            else:
                # Take from client 0 if no samples left
                if 0 in partitions and len(partitions[0]) > 1:
                    partitions[client_id] = partitions[0].iloc[[0]]
                    partitions[0] = partitions[0].iloc[1:]

    return partitions


def realistic_bank_partition_from_arrays(X: np.ndarray,
                                        y: np.ndarray,
                                        n_clients: int,
                                        region_labels: Optional[np.ndarray] = None,
                                        balance_within_regions: bool = True,
                                        random_state: int = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Partition array data using realistic bank simulation.

    This is a convenience function for working with numpy arrays instead
    of DataFrames.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        n_clients: Number of banks (clients)
        region_labels: Optional array of region labels for each sample
        balance_within_regions: Whether to balance classes within each region
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping client_id to (X_client, y_client) tuples
    """
    # Convert to DataFrame
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = y

    if region_labels is not None:
        df['region'] = region_labels
        region_col = 'region'
    else:
        region_col = None

    # Partition
    partitions = realistic_bank_partition(
        df=df,
        n_clients=n_clients,
        region_col=region_col,
        label_col='label',
        feature_cols=feature_cols,
        balance_within_regions=balance_within_regions,
        random_state=random_state
    )

    # Convert back to arrays
    result = {}
    for client_id, client_df in partitions.items():
        X_client = client_df[feature_cols].values
        y_client = client_df['label'].values
        result[client_id] = (X_client, y_client)

    return result
