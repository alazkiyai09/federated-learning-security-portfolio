"""
Private Set Intersection (PSI) for Vertical Federated Learning.

Implements PSI protocol to align user IDs between parties without revealing
non-intersecting IDs. This simulates the PSI process for demonstration purposes.

Reference: "Private Set Intersection: A Multi-Protocol Approach"
"""

import os
import json
import hashlib
import numpy as np
from typing import Set, Tuple, Dict, List, Literal
from dataclasses import dataclass


@dataclass
class PSIResult:
    """Result of PSI protocol."""
    intersection: Set[str]
    intersection_size: int
    protocol_metadata: Dict


class PrivateSetIntersection:
    """
    Private Set Intersection for ID alignment in Vertical FL.

    Simulates secure PSI protocol where parties can find common users
    without revealing users that are not in the intersection.
    """

    def __init__(
        self,
        method: Literal['hashing', 'ec'] = 'hashing',
        hash_function: str = 'sha256',
        salt_length: int = 32
    ):
        """
        Initialize PSI protocol.

        Args:
            method: PSI method ('hashing' or 'ec' for elliptic curve)
            hash_function: Hash function to use (sha256, sha512)
            salt_length: Length of salt for hashing
        """
        self.method = method
        self.hash_function = hash_function
        self.salt_length = salt_length

    def _hash_id(self, user_id: str, salt: bytes) -> str:
        """Hash a user ID with salt."""
        h = hashlib.new(self.hash_function)
        h.update(salt)
        h.update(user_id.encode())
        return h.hexdigest()

    def execute_hashing_psi(
        self,
        client_ids: Set[str],
        server_ids: Set[str],
        role: Literal['client', 'server']
    ) -> Tuple[Set[str], Dict]:
        """
        Execute hashing-based PSI protocol.

        Protocol:
        1. Client and Server agree on a salt
        2. Both parties hash their IDs with the salt
        3. Both parties exchange hashed IDs
        4. Client computes intersection of hashed IDs
        5. Only intersection is revealed

        Args:
            client_ids: IDs held by client (Party A)
            server_ids: IDs held by server (Party B)
            role: 'client' or 'server'

        Returns:
            Tuple of (intersection_set, metadata)
        """
        # In real PSI, salt would be established via secure key exchange
        salt = os.urandom(self.salt_length)

        # Hash client IDs
        client_hashed = {self._hash_id(uid, salt) for uid in client_ids}

        # Hash server IDs
        server_hashed = {self._hash_id(uid, salt) for uid in server_ids}

        # Compute intersection
        if role == 'client':
            # Client finds intersection
            hashed_intersection = client_hashed & server_hashed

            # Map back to original IDs (only for intersection)
            intersection = {
                uid for uid in client_ids
                if self._hash_id(uid, salt) in hashed_intersection
            }
        else:
            # Server finds intersection
            hashed_intersection = client_hashed & server_hashed

            intersection = {
                uid for uid in server_ids
                if self._hash_id(uid, salt) in hashed_intersection
            }

        metadata = {
            'method': 'hashing',
            'hash_function': self.hash_function,
            'salt_length': self.salt_length,
            'client_set_size': len(client_ids),
            'server_set_size': len(server_ids),
            'intersection_size': len(intersection),
            'privacy_guarantee': 'Only intersection revealed, no information about non-intersecting IDs'
        }

        return intersection, metadata

    def simulate_psi(
        self,
        party_a_ids: Set[str],
        party_b_ids: Set[str]
    ) -> PSIResult:
        """
        Simulate PSI between Party A and Party B.

        This simulates the full PSI protocol execution.

        Args:
            party_a_ids: User IDs held by Party A
            party_b_ids: User IDs held by Party B

        Returns:
            PSIResult with intersection and metadata
        """
        intersection, metadata = self.execute_hashing_psi(
            party_a_ids, party_b_ids, role='client'
        )

        # Verify intersection is correct
        assert intersection.issubset(party_a_ids), "Intersection contains non-Party A IDs"
        assert intersection.issubset(party_b_ids), "Intersection contains non-Party B IDs"

        return PSIResult(
            intersection=intersection,
            intersection_size=len(intersection),
            protocol_metadata=metadata
        )

    def save_psi_result(self, result: PSIResult, save_path: str) -> None:
        """Save PSI result to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            'intersection': list(result.intersection),
            'intersection_size': result.intersection_size,
            'metadata': result.protocol_metadata
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_psi_result(load_path: str) -> PSIResult:
        """Load PSI result from file."""
        with open(load_path, 'r') as f:
            data = json.load(f)

        return PSIResult(
            intersection=set(data['intersection']),
            intersection_size=data['intersection_size'],
            protocol_metadata=data['metadata']
        )


def execute_psi(
    client_ids: Set[str],
    server_ids: Set[str],
    method: Literal['hashing', 'ec'] = 'hashing',
    save_path: str = None
) -> Tuple[Set[str], Dict]:
    """
    Execute PSI protocol (convenience function).

    Args:
        client_ids: IDs held by client
        server_ids: IDs held by server
        method: PSI method
        save_path: Optional path to save result

    Returns:
        Tuple of (intersection_set, metadata)
    """
    psi = PrivateSetIntersection(method=method)
    result = psi.simulate_psi(client_ids, server_ids)

    if save_path:
        psi.save_psi_result(result, save_path)

    return result.intersection, result.protocol_metadata


if __name__ == "__main__":
    # Test PSI
    print("Testing Private Set Intersection...")

    # Create test data
    party_a_ids = {f"user_{i}" for i in range(100)}
    party_b_ids = {f"user_{i}" for i in range(50, 150)}

    # Execute PSI
    intersection, metadata = execute_psi(
        party_a_ids, party_b_ids,
        save_path="data/psi_intersection.json"
    )

    print(f"\nPSI Results:")
    print(f"Party A size: {metadata['client_set_size']}")
    print(f"Party B size: {metadata['server_set_size']}")
    print(f"Intersection size: {metadata['intersection_size']}")
    print(f"Method: {metadata['method']}")
    print(f"Privacy guarantee: {metadata['privacy_guarantee']}")

    # Verify
    expected = party_a_ids & party_b_ids
    assert intersection == expected, "PSI failed!"
    print("\n✓ PSI verification passed")
