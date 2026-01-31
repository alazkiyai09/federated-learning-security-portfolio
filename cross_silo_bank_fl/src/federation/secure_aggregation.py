"""
Secure aggregation simulation using additive masking.
"""

import numpy as np
from typing import List, Tuple


def apply_additive_masking(
    update: np.ndarray,
    n_clients: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply additive masking to model update for secure aggregation.

    Simulates secure aggregation where each client adds noise to their update.
    When all updates are summed, the noise cancels out.

    Args:
        update: Model update (flattened parameters)
        n_clients: Total number of clients
        seed: Random seed for reproducibility

    Returns:
        Tuple of (masked_update, mask)
    """
    rng = np.random.RandomState(seed)

    # Generate random mask
    mask = rng.randn(*update.shape) * 0.01  # Small noise

    # Apply mask
    masked_update = update + mask

    return masked_update, mask


def simulate_secure_aggregation(
    updates: List[np.ndarray],
    client_ids: List[str],
    n_bits: int = 32
) -> np.ndarray:
    """
    Simulate secure aggregation of client updates.

    In real secure aggregation:
    1. Each client adds pairwise masks with other clients
    2. Clients also add their own noise (encrypted)
    3. Server sums all masked updates
    4. Clients unmask their contributions
    5. Final aggregated update is revealed

    Here we simulate this by:
    1. Adding small noise to each update
    2. Summing all updates
    3. The noise would cancel out in real implementation

    Args:
        updates: List of model updates from clients
        client_ids: List of client identifiers
        n_bits: Number of bits for masking precision

    Returns:
        Aggregated update (sum of all updates)
    """
    n_clients = len(updates)

    # Apply masking to each update
    masked_updates = []
    for i, (update, client_id) in enumerate(zip(updates, client_ids)):
        # Use client_id as seed for reproducibility
        seed = hash(client_id) % (2 ** 32)
        masked_update, _ = apply_additive_masking(update, n_clients, seed)
        masked_updates.append(masked_update)

    # Sum all masked updates
    # In real secure aggregation, masks would cancel out
    # Here we just sum the original updates (masks ignored for simplicity)
    aggregated = sum(updates)

    return aggregated


def pairwise_masking(
    value: float,
    client_id: str,
    partner_id: str
) -> float:
    """
    Generate pairwise mask for secure aggregation.

    In real implementation, two clients agree on a shared secret.
    One adds +mask, the other adds -mask.
    When summed, they cancel out.

    Args:
        value: Value to mask
        client_id: First client ID
        partner_id: Second client ID

    Returns:
        Masked value
    """
    # Generate pseudo-random mask from client IDs
    seed = hash(f"{client_id}_{partner_id}") % (2 ** 32)
    rng = np.random.RandomState(seed)

    # Scale mask by value magnitude
    mask = rng.randn() * abs(value) * 0.01

    # Determine sign based on client_id comparison
    if client_id < partner_id:
        return value + mask
    else:
        return value - mask


def verify_cancellation(
    client_ids: List[str],
    base_value: float = 1.0
) -> bool:
    """
    Verify that pairwise masks cancel out correctly.

    Args:
        client_ids: List of client IDs
        base_value: Base value to test with

    Returns:
        True if masks cancel out (sum ≈ 0)
    """
    n_clients = len(client_ids)

    # Generate all pairwise masks
    total = 0.0

    for i, client_id in enumerate(client_ids):
        for j, partner_id in enumerate(client_ids):
            if i != j:
                # Each pair should cancel out
                seed = hash(f"{client_id}_{partner_id}") % (2 ** 32)
                rng = np.random.RandomState(seed)
                mask = rng.randn() * abs(base_value) * 0.01

                if client_id < partner_id:
                    total += mask
                else:
                    total -= mask

    # Check if total is close to zero
    return abs(total) < 1e-10


class SecureAggregator:
    """
    Simulates secure aggregation for federated learning.
    """

    def __init__(self, n_bits: int = 32):
        """
        Initialize secure aggregator.

        Args:
            n_bits: Precision for masking
        """
        self.n_bits = n_bits
        self.client_pairs = []

    def setup_pairs(self, client_ids: List[str]) -> None:
        """
        Setup client pairs for pairwise masking.

        Args:
            client_ids: List of client IDs
        """
        n_clients = len(client_ids)
        self.client_pairs = []

        # Create pairs (in real implementation, would use more sophisticated pairing)
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                self.client_pairs.append((client_ids[i], client_ids[j]))

    def mask_update(
        self,
        update: np.ndarray,
        client_id: str
    ) -> Tuple[np.ndarray, List]:
        """
        Mask a client's update.

        Args:
            update: Model update
            client_id: Client identifier

        Returns:
            Tuple of (masked_update, masks_for_later_unmasking)
        """
        masked_update = update.copy()
        masks = []

        for partner_id, _ in self.client_pairs:
            if partner_id != client_id:
                # Add pairwise mask
                seed = hash(f"{client_id}_{partner_id}") % (2 ** 32)
                rng = np.random.RandomState(seed)
                mask = rng.randn(*update.shape) * 0.01
                masked_update += mask
                masks.append(mask)

        return masked_update, masks

    def aggregate(
        self,
        masked_updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate masked updates.

        In real implementation, masks would cancel out.

        Args:
            masked_updates: List of masked updates

        Returns:
            Aggregated update
        """
        # Simply sum (masks would cancel in real implementation)
        return sum(masked_updates)
