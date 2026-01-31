"""
Unit tests for Private Set Intersection (PSI) protocol.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import tempfile
from psi.private_set_intersection import PrivateSetIntersection, execute_psi


def test_psi_intersection_correctness():
    """Test that PSI correctly computes intersection."""
    print("\n=== Testing PSI Intersection Correctness ===")

    psi = PrivateSetIntersection()

    # Create test sets
    party_a_ids = {f"user_{i}" for i in range(100)}
    party_b_ids = {f"user_{i}" for i in range(50, 150)}

    # Expected intersection
    expected = party_a_ids & party_b_ids

    # Execute PSI
    result = psi.simulate_psi(party_a_ids, party_b_ids)

    # Check correctness
    assert result.intersection == expected, "PSI intersection incorrect"
    assert result.intersection_size == len(expected), "PSI intersection size incorrect"

    print(f"✓ PSI intersection correct")
    print(f"  Party A size: {len(party_a_ids)}")
    print(f"  Party B size: {len(party_b_ids)}")
    print(f"  Intersection size: {result.intersection_size}")
    print(f"  Expected: {len(expected)}")

    return True


def test_psi_no_intersection():
    """Test PSI with disjoint sets."""
    print("\n=== Testing PSI with Disjoint Sets ===")

    psi = PrivateSetIntersection()

    party_a_ids = {f"user_a_{i}" for i in range(50)}
    party_b_ids = {f"user_b_{i}" for i in range(50)}

    result = psi.simulate_psi(party_a_ids, party_b_ids)

    assert result.intersection_size == 0, "Expected empty intersection"
    assert len(result.intersection) == 0, "Expected empty intersection set"

    print("✓ PSI correctly handles disjoint sets")
    return True


def test_psi_complete_overlap():
    """Test PSI with identical sets."""
    print("\n=== Testing PSI with Complete Overlap ===")

    psi = PrivateSetIntersection()

    party_ids = {f"user_{i}" for i in range(100)}

    result = psi.simulate_psi(party_ids, party_ids)

    assert result.intersection == party_ids, "Expected complete intersection"
    assert result.intersection_size == len(party_ids), "Expected full size"

    print("✓ PSI correctly handles complete overlap")
    return True


def test_psi_save_load():
    """Test saving and loading PSI results."""
    print("\n=== Testing PSI Save/Load ===")

    psi = PrivateSetIntersection()

    party_a_ids = {f"user_{i}" for i in range(100)}
    party_b_ids = {f"user_{i}" for i in range(50, 150)}

    result = psi.simulate_psi(party_a_ids, party_b_ids)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        psi.save_psi_result(result, temp_path)

        # Load back
        loaded_result = psi.load_psi_result(temp_path)

        # Verify
        assert loaded_result.intersection == result.intersection, "Loaded intersection incorrect"
        assert loaded_result.intersection_size == result.intersection_size, "Loaded size incorrect"

        print("✓ PSI save/load working correctly")
        return True

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_psi_metadata():
    """Test that PSI metadata is correct."""
    print("\n=== Testing PSI Metadata ===")

    psi = PrivateSetIntersection(
        method='hashing',
        hash_function='sha256',
        salt_length=32
    )

    party_a_ids = {f"user_{i}" for i in range(100)}
    party_b_ids = {f"user_{i}" for i in range(50, 150)}

    result = psi.simulate_psi(party_a_ids, party_b_ids)

    metadata = result.protocol_metadata

    assert metadata['method'] == 'hashing', "Method incorrect"
    assert metadata['hash_function'] == 'sha256', "Hash function incorrect"
    assert metadata['salt_length'] == 32, "Salt length incorrect"
    assert metadata['client_set_size'] == len(party_a_ids), "Client size incorrect"
    assert metadata['server_set_size'] == len(party_b_ids), "Server size incorrect"

    print("✓ PSI metadata correct")
    print(f"  Method: {metadata['method']}")
    print(f"  Hash function: {metadata['hash_function']}")
    print(f"  Salt length: {metadata['salt_length']}")

    return True


def test_psi_convenience_function():
    """Test the convenience function execute_psi."""
    print("\n=== Testing PSI Convenience Function ===")

    client_ids = {f"user_{i}" for i in range(100)}
    server_ids = {f"user_{i}" for i in range(50, 150)}

    intersection, metadata = execute_psi(client_ids, server_ids)

    expected = client_ids & server_ids

    assert intersection == expected, "Convenience function intersection incorrect"
    assert 'method' in metadata, "Metadata missing method"

    print("✓ PSI convenience function working correctly")

    return True


def test_psi_large_scale():
    """Test PSI with larger sets."""
    print("\n=== Testing PSI Large Scale ===")

    psi = PrivateSetIntersection()

    # Larger sets (simulating realistic scenario)
    party_a_ids = {f"user_{i}" for i in range(100000)}
    party_b_ids = {f"user_{i}" for i in range(50000, 150000)}

    result = psi.simulate_psi(party_a_ids, party_b_ids)

    expected = party_a_ids & party_b_ids

    assert result.intersection == expected, "Large scale PSI failed"
    assert result.intersection_size == 50000, "Large scale intersection size incorrect"

    print("✓ PSI handles large sets efficiently")
    print(f"  Party A size: {len(party_a_ids):,}")
    print(f"  Party B size: {len(party_b_ids):,}")
    print(f"  Intersection size: {result.intersection_size:,}")

    return True


def run_all_tests():
    """Run all PSI tests."""
    print("\n" + "="*80)
    print("PSI UNIT TESTS")
    print("="*80)

    tests = [
        test_psi_intersection_correctness,
        test_psi_no_intersection,
        test_psi_complete_overlap,
        test_psi_save_load,
        test_psi_metadata,
        test_psi_convenience_function,
        test_psi_large_scale,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except AssertionError as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
        except Exception as e:
            print(f"\n✗ {test.__name__} ERROR: {e}")
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
