#!/usr/bin/env python3
"""
Simple test runner that doesn't require pytest
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_test_file(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)

    try:
        # Execute the test file
        with open(test_file) as f:
            code = f.read()

        # Replace pytest.main with direct execution
        exec_globals = {"__file__": test_file, "__name__": "__main__"}
        exec(code, exec_globals)

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all test files."""
    tests_dir = Path(__file__).parent / "tests"
    test_files = [
        tests_dir / "test_utils.py",
        tests_dir / "test_client.py",
        tests_dir / "test_strategy.py",
    ]

    results = []
    for test_file in test_files:
        if test_file.exists():
            results.append(run_test_file(test_file))
        else:
            print(f"Test file not found: {test_file}")
            results.append(False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for test_file, result in zip(test_files, results):
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_file}")

    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
