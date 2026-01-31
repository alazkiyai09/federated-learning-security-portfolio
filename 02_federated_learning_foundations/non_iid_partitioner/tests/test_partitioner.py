"""
Integration tests for the main NonIIDPartitioner class.
"""

import numpy as np
import pandas as pd
import pytest
from src.partitioner import NonIIDPartitioner
from src.utils import validate_partitions


class TestNonIIDPartitioner:
    """Integration tests for the main partitioner class."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        # Create synthetic dataset
        self.X = np.random.randn(1000, 20)
        self.y = np.random.randint(0, 5, 1000)
        self.n_clients = 10

    def test_initialization(self):
        """Test partitioner initialization."""
        partitioner = NonIIDPartitioner(n_clients=10, random_state=42)
        assert partitioner.n_clients == 10
        assert partitioner.random_state == 42

    def test_invalid_n_clients(self):
        """Test that invalid n_clients raises error."""
        with pytest.raises(ValueError):
            NonIIDPartitioner(n_clients=0)

        with pytest.raises(ValueError):
            NonIIDPartitioner(n_clients=-5)

    def test_partition_iid(self):
        """Test IID partition through main class."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitions = partitioner.partition_iid(self.X, self.y)

        assert len(partitions) == self.n_clients
        assert all(isinstance(data, tuple) for data in partitions.values())
        assert all(len(data) == 2 for data in partitions.values())

        # Verify coverage
        total_samples = sum(len(data[1]) for data in partitions.values())
        assert total_samples == len(self.y)

    def test_partition_label_skew(self):
        """Test label skew partition through main class."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitions = partitioner.partition_label_skew(self.X, self.y, alpha=0.5)

        assert len(partitions) == self.n_clients
        total_samples = sum(len(data[1]) for data in partitions.values())
        assert total_samples == len(self.y)

    def test_partition_quantity_skew(self):
        """Test quantity skew partition through main class."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitions = partitioner.partition_quantity_skew(self.X, self.y, exponent=1.5)

        assert len(partitions) == self.n_clients

        # Check for skew
        sizes = partitioner.get_client_sizes()
        assert max(sizes.values()) > 1.5 * min(sizes.values())

    def test_partition_feature_skew(self):
        """Test feature skew partition through main class."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitions = partitioner.partition_feature_skew(self.X, self.y, n_clusters=5)

        assert len(partitions) == self.n_clients
        total_samples = sum(len(data[1]) for data in partitions.values())
        assert total_samples == len(self.y)

    def test_partition_realistic_bank_arrays(self):
        """Test realistic bank partition with arrays."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)

        # Create region labels
        regions = np.random.randint(0, 5, 1000)

        partitions = partitioner.partition_realistic_bank_arrays(
            self.X, self.y, region_labels=regions
        )

        assert len(partitions) == self.n_clients

    def test_partition_realistic_bank_dataframe(self):
        """Test realistic bank partition with DataFrame."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)

        # Create DataFrame
        df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(20)])
        df['label'] = self.y
        df['region'] = np.random.randint(0, 5, 1000)

        partitions = partitioner.partition_realistic_bank(
            df, region_col='region', label_col='label'
        )

        assert len(partitions) == self.n_clients
        assert all(isinstance(df, pd.DataFrame) for df in partitions.values())

    def test_get_partition_statistics(self):
        """Test getting partition statistics."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitioner.partition_iid(self.X, self.y)

        stats = partitioner.get_partition_statistics()

        assert len(stats) == self.n_clients
        assert all('n_samples' in s for s in stats.values())
        assert all('label_distribution' in s for s in stats.values())

    def test_get_statistics_no_partition(self):
        """Test that getting stats before partitioning raises error."""
        partitioner = NonIIDPartitioner(n_clients=10, random_state=42)

        with pytest.raises(ValueError):
            partitioner.get_partition_statistics()

    def test_get_client_sizes(self):
        """Test getting client sizes."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)
        partitioner.partition_iid(self.X, self.y)

        sizes = partitioner.get_client_sizes()

        assert len(sizes) == self.n_clients
        assert all(isinstance(size, int) for size in sizes.values())
        assert sum(sizes.values()) == len(self.y)

    def test_reproducibility_across_methods(self):
        """Test that same random_state produces same results."""
        partitioner1 = NonIIDPartitioner(n_clients=5, random_state=42)
        partitioner2 = NonIIDPartitioner(n_clients=5, random_state=42)

        partitions1 = partitioner1.partition_label_skew(self.X, self.y, alpha=1.0)
        partitions2 = partitioner2.partition_label_skew(self.X, self.y, alpha=1.0)

        for cid in range(5):
            np.testing.assert_array_equal(partitions1[cid][0], partitions2[cid][0])
            np.testing.assert_array_equal(partitions1[cid][1], partitions2[cid][1])

    def test_invalid_dataframe_column(self):
        """Test that invalid column name raises error."""
        partitioner = NonIIDPartitioner(n_clients=5, random_state=42)

        df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(20)])
        df['label'] = self.y

        with pytest.raises(ValueError):
            partitioner.partition_realistic_bank(
                df, region_col='nonexistent_region', label_col='label'
            )

    def test_invalid_data_dimensions(self):
        """Test that invalid data raises error."""
        partitioner = NonIIDPartitioner(n_clients=5, random_state=42)

        # Mismatched dimensions
        X_bad = np.random.randn(100, 10)
        y_bad = np.random.randint(0, 3, 50)  # Wrong length

        with pytest.raises(ValueError):
            partitioner.partition_iid(X_bad, y_bad)


class TestMultipleStrategiesComparison:
    """Tests comparing different strategies."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X = np.random.randn(1000, 20)
        self.y = np.random.randint(0, 5, 1000)
        self.n_clients = 10

    def test_strategies_produce_different_partitions(self):
        """Test that different strategies produce different distributions."""
        partitioner = NonIIDPartitioner(n_clients=self.n_clients, random_state=42)

        # Get partitions from different strategies
        iid_parts = partitioner.partition_iid(self.X, self.y)
        label_parts = partitioner.partition_label_skew(self.X, self.y, alpha=0.3)
        quantity_parts = partitioner.partition_quantity_skew(self.X, self.y, exponent=1.3)

        # Extract size distributions
        iid_sizes = sorted(len(p[1]) for p in iid_parts.values())
        label_sizes = sorted(len(p[1]) for p in label_parts.values())
        quantity_sizes = sorted(len(p[1]) for p in quantity_parts.values())

        # IID should be most uniform
        iid_variance = np.var(iid_sizes)

        # Label skew might have variance in sizes
        label_variance = np.var(label_sizes)

        # Quantity skew should have highest variance
        quantity_variance = np.var(quantity_sizes)

        assert quantity_variance > iid_variance

    def test_all_strategies_valid(self):
        """Test that all strategies produce valid partitions."""
        from src.utils import validate_data

        partitioner = NonIIDPartitioner(n_clients=10, random_state=42)

        # Validate data first
        validate_data(self.X, self.y)

        # Test all strategies
        strategies = [
            ('iid', lambda: partitioner.partition_iid(self.X, self.y)),
            ('label_skew', lambda: partitioner.partition_label_skew(self.X, self.y, alpha=0.5)),
            ('quantity_skew', lambda: partitioner.partition_quantity_skew(self.X, self.y, exponent=2.0)),
            ('feature_skew', lambda: partitioner.partition_feature_skew(self.X, self.y, n_clusters=5)),
        ]

        for name, strategy_fn in strategies:
            partitions = strategy_fn()
            assert len(partitions) == 10, f"{name} produced wrong number of clients"

            # Check total samples preserved
            total = sum(len(p[1]) for p in partitions.values())
            assert total == len(self.y), f"{name} lost or gained samples"
