"""
Unit tests for individual partition strategies.
"""

import numpy as np
import pytest
from src.strategies.iid import iid_partition
from src.strategies.label_skew import dirichlet_partition, dirichlet_partition_indices
from src.strategies.quantity_skew import (
    power_law_allocation,
    quantity_skew_partition,
    quantity_skew_partition_indices
)
from src.strategies.feature_skew import feature_based_partition
from src.utils import validate_partitions


class TestIIDPartition:
    """Tests for IID partition strategy."""

    def test_iid_basic(self):
        """Test basic IID partition."""
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 5, 1000)
        n_clients = 10

        partitions = iid_partition(X, y, n_clients, random_state=42)

        assert len(partitions) == n_clients
        validate_partitions(
            {cid: indices for cid, (indices, _) in
             {cid: (X[indices], y[indices]) for cid, indices in partitions.items()}.items()},
            1000, n_clients
        )

    def test_iid_reproducibility(self):
        """Test that same random_state produces identical results."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        partitions1 = iid_partition(X, y, 5, random_state=42)
        partitions2 = iid_partition(X, y, 5, random_state=42)

        for cid in range(5):
            np.testing.assert_array_equal(partitions1[cid][0], partitions2[cid][0])
            np.testing.assert_array_equal(partitions1[cid][1], partitions2[cid][1])


class TestLabelSkew:
    """Tests for Dirichlet label skew strategy."""

    def test_dirichlet_basic(self):
        """Test basic Dirichlet partition."""
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 5, 1000)
        n_clients = 10
        alpha = 1.0

        partitions = dirichlet_partition(X, y, n_clients, alpha, random_state=42)

        assert len(partitions) == n_clients

        # Check all samples assigned
        total_samples = sum(len(p[1]) for p in partitions.values())
        assert total_samples == 1000

    def test_dirichlet_alpha_validation(self):
        """Test that invalid alpha raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        with pytest.raises(ValueError):
            dirichlet_partition(X, y, 5, alpha=0)

        with pytest.raises(ValueError):
            dirichlet_partition(X, y, 5, alpha=-1)

    def test_dirichlet_reproducibility(self):
        """Test reproducibility with random_state."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        partitions1 = dirichlet_partition(X, y, 5, alpha=0.5, random_state=42)
        partitions2 = dirichlet_partition(X, y, 5, alpha=0.5, random_state=42)

        for cid in range(5):
            np.testing.assert_array_equal(partitions1[cid][0], partitions2[cid][0])

    def test_dirichlet_extreme_skew(self):
        """Test that low alpha creates extreme skew."""
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 5, 1000)

        # Very low alpha should create high skew
        partitions = dirichlet_partition(X, y, 10, alpha=0.1, random_state=42)

        # At least some clients should have highly skewed distributions
        skewed_clients = 0
        for cid, (X_c, y_c) in partitions.items():
            if len(y_c) > 0:
                unique_labels = np.unique(y_c)
                max_count = max(np.bincount(y_c))
                if max_count / len(y_c) > 0.8:  # 80%+ from one class
                    skewed_clients += 1

        assert skewed_clients > 0

    def test_dirichlet_indices_version(self):
        """Test the indices-only version."""
        y = np.random.randint(0, 5, 1000)

        partitions = dirichlet_partition_indices(y, n_clients=10, alpha=1.0, random_state=42)

        assert len(partitions) == 10

        # Check all indices covered
        all_indices = np.concatenate(list(partitions.values()))
        assert len(np.unique(all_indices)) == 1000


class TestQuantitySkew:
    """Tests for power law quantity skew strategy."""

    def test_power_law_allocation(self):
        """Test power law allocation generation."""
        n_samples = 1000
        n_clients = 10

        counts = power_law_allocation(n_samples, n_clients, exponent=1.5, random_state=42)

        assert len(counts) == n_clients
        assert counts.sum() == n_samples
        assert np.all(counts >= 1)

    def test_power_law_exponent_validation(self):
        """Test exponent validation."""
        with pytest.raises(ValueError):
            power_law_allocation(100, 5, exponent=1.0)

        with pytest.raises(ValueError):
            power_law_allocation(100, 5, exponent=0.5)

    def test_power_law_minimum_validation(self):
        """Test minimum samples constraint."""
        with pytest.raises(ValueError):
            power_law_allocation(10, 20, min_samples_per_client=1)

    def test_quantity_skew_partition(self):
        """Test full quantity skew partition."""
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 3, 1000)

        partitions = quantity_skew_partition(X, y, 10, exponent=1.5, random_state=42)

        assert len(partitions) == 10

        # Check sample counts follow power law (some clients have much more)
        sample_counts = [len(p[1]) for p in partitions.values()]
        assert max(sample_counts) > 2 * min(sample_counts)  # At least 2x difference

    def test_quantity_skew_reproducibility(self):
        """Test reproducibility."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        partitions1 = quantity_skew_partition(X, y, 5, exponent=2.0, random_state=42)
        partitions2 = quantity_skew_partition(X, y, 5, exponent=2.0, random_state=42)

        for cid in range(5):
            np.testing.assert_array_equal(partitions1[cid][0], partitions2[cid][0])


class TestFeatureSkew:
    """Tests for feature-based skew strategy."""

    def test_feature_skew_basic(self):
        """Test basic feature-based partition."""
        # Create clustered data
        X = np.vstack([
            np.random.randn(200, 5) + 5,   # Cluster 1
            np.random.randn(200, 5) - 5,   # Cluster 2
            np.random.randn(200, 5),       # Cluster 3
        ])
        y = np.repeat([0, 1, 2], 200)

        partitions = feature_based_partition(X, y, n_clients=3, n_clusters=3, random_state=42)

        assert len(partitions) == 3
        total_samples = sum(len(p[1]) for p in partitions.values())
        assert total_samples == 600

    def test_feature_skew_clusters_validation(self):
        """Test that n_clusters cannot exceed n_samples."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)

        with pytest.raises(ValueError):
            feature_based_partition(X, y, n_clients=10, n_clusters=100)

    def test_feature_skew_reproducibility(self):
        """Test reproducibility."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        partitions1 = feature_based_partition(X, y, n_clients=5, random_state=42)
        partitions2 = feature_based_partition(X, y, n_clients=5, random_state=42)

        for cid in range(5):
            np.testing.assert_array_equal(partitions1[cid][0], partitions2[cid][0])


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_partitions_correct(self):
        """Test validation with correct partitions."""
        indices = {i: np.arange(i*10, (i+1)*10) for i in range(10)}
        validate_partitions(indices, n_samples=100, n_clients=10)

    def test_validate_partitions_missing_samples(self):
        """Test validation detects missing samples."""
        indices = {i: np.arange(i*9, (i+1)*9) for i in range(10)}
        with pytest.raises(ValueError):
            validate_partitions(indices, n_samples=100, n_clients=10)

    def test_validate_partitions_wrong_clients(self):
        """Test validation detects wrong number of clients."""
        indices = {i: np.arange(i*10, (i+1)*10) for i in range(5)}
        with pytest.raises(ValueError):
            validate_partitions(indices, n_samples=100, n_clients=10)
