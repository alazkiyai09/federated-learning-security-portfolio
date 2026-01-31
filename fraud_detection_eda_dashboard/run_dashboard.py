#!/usr/bin/env python3
"""
Quick start script for the Fraud Detection EDA Dashboard.

This script provides an easy way to launch the dashboard with
optional configuration parameters.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fraud_detection_dashboard.app import main, validate_data_path, DEFAULT_DATA_PATH


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch the Credit Card Fraud Detection EDA Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_dashboard.py

  # Run with custom data path
  python run_dashboard.py --data path/to/data.csv

  # Run on different port
  python run_dashboard.py --port 8080

  # Run in production mode
  python run_dashboard.py --no-debug
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to the creditcard.csv dataset file (default: data/creditcard.csv)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the server on (default: 8050)'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable debug mode (for production)'
    )

    return parser.parse_args()


def main_entry():
    """Main entry point for the script."""
    args = parse_arguments()

    # Determine data path
    data_path = args.data if args.data else DEFAULT_DATA_PATH

    # Validate data path
    print("="*60)
    print("💳 Credit Card Fraud Detection EDA Dashboard")
    print("="*60)
    print(f"\n📂 Dataset: {data_path}")

    if not validate_data_path(data_path):
        print("\n" + "="*60)
        print("⚠️  Setup Required!")
        print("="*60)
        print("\n1. Download the dataset from:")
        print("   https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print(f"\n2. Place it at: {data_path}")
        print("\nThen run this script again.")
        print("="*60 + "\n")
        sys.exit(1)

    print("✅ Dataset found!\n")

    # Run the dashboard
    try:
        main(
            data_path=data_path,
            host=args.host,
            port=args.port,
            debug=not args.no_debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
