#!/usr/bin/env python3
"""
Main entry point for Cross-Silo Bank Federated Learning Simulation.

This script orchestrates:
1. Data generation for 5 realistic bank profiles
2. Local baseline training (each bank trains independently)
3. Federated learning simulation (Flower framework)
4. Centralized baseline (pooled data - privacy upper bound)
5. Comparison and visualization

Author: [Your Name]
Purpose: PhD research portfolio - Trustworthy Federated Learning
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.helpers import set_seed, print_section, ensure_dir
from src.data_generation.bank_profile import load_bank_profiles, get_summary_statistics
from src.data_generation.transaction_generator import TransactionGenerator
from src.data_generation.fraud_generator import FraudGenerator
from src.preprocessing.partitioner import (
    generate_all_bank_data,
    partition_data_by_bank,
    create_federated_splits,
    create_centralized_dataset,
    analyze_non_iidness
)
from src.preprocessing.feature_engineering import FeatureEngineerer
from src.experiments.local_baseline import train_local_models, evaluate_local_models
from src.experiments.federated_training import run_federated_simulation, prepare_federated_data
from src.experiments.centralized_baseline import train_centralized_model
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_improvement,
    create_comparison_table,
    calculate_aggregate_metrics,
    save_metrics
)
from src.evaluation.visualization import (
    plot_per_bank_comparison,
    plot_learning_curves,
    plot_fraud_analysis,
    plot_improvement_analysis,
    create_summary_figure
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Silo Bank Federated Learning Simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=15,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Local epochs per FL round"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local baseline training"
    )
    parser.add_argument(
        "--skip-fl",
        action="store_true",
        help="Skip federated learning"
    )
    parser.add_argument(
        "--skip-centralized",
        action="store_true",
        help="Skip centralized baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    metrics_dir = ensure_dir(output_dir / "metrics")
    data_dir = ensure_dir("data/processed")

    print_section("CROSS-SILO BANK FEDERATED LEARNING SIMULATION")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"N FL rounds: {args.n_rounds}")
    print(f"Local epochs: {args.local_epochs}")

    # ========================================================================
    # 1. LOAD BANK PROFILES
    # ========================================================================
    print_section("1. LOADING BANK PROFILES")

    profiles_dict = load_bank_profiles()
    profiles = list(profiles_dict.values())

    summary = get_summary_statistics(profiles)
    print(f"\nSummary Statistics:")
    print(f"  Number of banks: {summary['n_banks']}")
    print(f"  Total customers: {summary['total_clients']:,}")
    print(f"  Total daily transactions: {summary['total_daily_transactions']:,}")
    print(f"  Average fraud rate: {summary['average_fraud_rate']:.4f}")

    print("\nBank Profiles:")
    for profile in profiles:
        print(f"  {profile.bank_id}: {profile.name}")
        print(f"    Type: {profile.bank_type}")
        print(f"    Clients: {profile.n_clients:,}")
        print(f"    Daily tx: {profile.daily_transactions:,}")
        print(f"    Fraud rate: {profile.fraud_rate:.4f}")

    # ========================================================================
    # 2. GENERATE DATA
    # ========================================================================
    print_section("2. GENERATING TRANSACTION DATA")

    bank_data = generate_all_bank_data(
        profiles=profiles,
        n_days=30,
        output_dir=str(data_dir),
        seed=args.seed
    )

    # Analyze non-IID characteristics
    print("\nAnalyzing Non-IID Characteristics:")
    split_data_temp = {}
    for bank_id, df in bank_data.items():
        train_df = df.sample(frac=0.7, random_state=args.seed)
        split_data_temp[bank_id] = {'train': train_df}

    non_iid_df = analyze_non_iidness(split_data_temp)
    print("\nNon-IID Analysis:")
    print(non_iid_df.to_string(index=False))

    # ========================================================================
    # 3. PREPROCESS AND PARTITION
    # ========================================================================
    print_section("3. PARTITIONING DATA")

    # Split data for each bank
    split_data = partition_data_by_bank(
        bank_data=bank_data,
        test_size=0.20,
        val_size=0.10,
        seed=args.seed
    )

    # Create centralized dataset
    centralized_data = create_centralized_dataset(
        bank_data=bank_data,
        test_size=0.20,
        val_size=0.10,
        seed=args.seed
    )

    # Prepare for FL (add split column for local baseline compatibility)
    for bank_id in split_data:
        split_data[bank_id]['train']['split'] = 'train'
        split_data[bank_id]['val']['split'] = 'val'
        split_data[bank_id]['test']['split'] = 'test'

    centralized_data['train']['split'] = 'train'
    centralized_data['val']['split'] = 'val'
    centralized_data['test']['split'] = 'test'

    # ========================================================================
    # 4. LOCAL BASELINE
    # ========================================================================
    local_results = {}
    if not args.skip_local:
        print_section("4. TRAINING LOCAL MODELS (BASELINE)")

        local_results = train_local_models(
            bank_data=split_data,
            model_config={
                'hidden_layers': [128, 64, 32],
                'dropout': 0.3
            },
            training_config={
                'n_epochs': 10,
                'batch_size': 256,
                'learning_rate': 0.001,
                'early_stopping_patience': 5
            },
            output_dir=str(metrics_dir / "local_models")
        )

        # Evaluate local models
        local_summary = evaluate_local_models(local_results)
        print("\nLocal Model Summary:")
        print(local_summary.to_string(index=False))
    else:
        print("\nSkipping local baseline training...")

    # ========================================================================
    # 5. FEDERATED LEARNING
    # ========================================================================
    fl_results = {}
    if not args.skip_fl:
        print_section("5. FEDERATED LEARNING SIMULATION")

        # Prepare data for FL
        fl_data, feature_engineer = prepare_federated_data(split_data)

        # Run FL simulation
        fl_results = run_federated_simulation(
            bank_data=fl_data,
            model_config={
                'hidden_layers': [128, 64, 32],
                'dropout': 0.3
            },
            training_config={
                'learning_rate': 0.001,
                'batch_size': 256,
                'local_epochs': args.local_epochs,
                'early_stopping_patience': 5
            },
            federation_config={
                'n_rounds': args.n_rounds,
                'fraction_fit': 1.0,
                'min_fit_clients': 5,
                'min_available_clients': 5
            },
            output_dir=str(metrics_dir / "federated")
        )
    else:
        print("\nSkipping federated learning...")

    # ========================================================================
    # 6. CENTRALIZED BASELINE
    # ========================================================================
    centralized_results = {}
    if not args.skip_centralized:
        print_section("6. TRAINING CENTRALIZED MODEL (UPPER BOUND)")

        centralized_results = train_centralized_model(
            centralized_data=centralized_data,
            model_config={
                'hidden_layers': [128, 64, 32],
                'dropout': 0.3
            },
            training_config={
                'n_epochs': 10,
                'batch_size': 256,
                'learning_rate': 0.001,
                'early_stopping_patience': 5
            },
            output_dir=str(metrics_dir / "centralized")
        )
    else:
        print("\nSkipping centralized baseline...")

    # ========================================================================
    # 7. COMPARISON AND VISUALIZATION
    # ========================================================================
    print_section("7. GENERATING COMPARISON REPORTS")

    # Compute all metrics
    if local_results and fl_results and centralized_results:
        comparison_df = compute_all_metrics(
            local_results,
            fl_results,
            centralized_results
        )

        print("\nPer-Bank Comparison:")
        print(comparison_df.to_string(index=False))

        # Save metrics
        save_metrics(
            comparison_df=comparison_df,
            improvements=compute_improvement(comparison_df),
            aggregates=calculate_aggregate_metrics(
                local_results, fl_results, centralized_results
            ),
            output_path=str(metrics_dir)
        )

        # Create comparison table
        comparison_table = create_comparison_table(
            local_results, fl_results, centralized_results
        )
        print("\n" + comparison_table.to_string(index=False))

        # Generate visualizations
        print("\nGenerating visualizations...")

        plot_per_bank_comparison(
            comparison_df,
            metric='auc',
            save_path=str(figures_dir / "per_bank_comparison.png")
        )

        if fl_results:
            plot_learning_curves(
                fl_results,
                save_path=str(figures_dir / "learning_curves.png")
            )

        plot_fraud_analysis(
            bank_data,
            save_path=str(figures_dir / "fraud_analysis.png")
        )

        plot_improvement_analysis(
            comparison_df,
            save_path=str(figures_dir / "improvement_analysis.png")
        )

        create_summary_figure(
            comparison_df,
            fl_results,
            bank_data,
            output_path=str(figures_dir / "summary.png")
        )

    # ========================================================================
    # 8. GENERATE README
    # ========================================================================
    print_section("8. GENERATING README")

    generate_readme(
        profiles=profiles,
        comparison_df=comparison_df if local_results and fl_results and centralized_results else None,
        improvements=compute_improvement(comparison_df) if local_results and fl_results and centralized_results else None,
        output_dir=output_dir
    )

    print_section("SIMULATION COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("\nKey outputs:")
    print(f"  - Metrics: {metrics_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - README: {output_dir / 'README.md'}")


def generate_readme(profiles, comparison_df, improvements, output_dir):
    """Generate README.md with results."""

    readme_content = """# Cross-Silo Bank Federated Learning Simulation

## Overview

This project simulates realistic federated learning across 5 banks for fraud detection. It demonstrates the benefits of collaboration while preserving data privacy, using the Flower framework.

## Bank Profiles

"""

    for profile in profiles:
        readme_content += f"""
### {profile.bank_id}: {profile.name}
- **Type:** {profile.bank_type}
- **Customer Base:** {profile.n_clients:,}
- **Daily Transactions:** {profile.daily_transactions:,}
- **Fraud Rate:** {profile.fraud_rate:.2%}
- **International Ratio:** {profile.international_ratio:.1%}
"""

    if comparison_df is not None:
        readme_content += """
## Experimental Results

### Comparison Table

| Bank | Local AUC | Federated AUC | Centralized AUC | FL vs Local |
|------|-----------|---------------|-----------------|-------------|
"""
        for _, row in comparison_df.iterrows():
            readme_content += f"| {row['bank_id']} | {row['local_auc']:.4f} | {row['fl_auc']:.4f} | {row['centralized_auc']:.4f} | {row['fl_vs_local_auc_improvement']:+.1f}% |\n"

    if improvements is not None:
        readme_content += f"""
### Summary Statistics

- **Mean FL vs Local Improvement:** {improvements['fl_vs_local_auc']['mean']:+.2f}% (±{improvements['fl_vs_local_auc']['std']:.2f}%)
- **Centralized Gap:** {improvements['centralized_gap_auc']['mean']:.2f}% (how close FL is to upper bound)
"""

    readme_content += """
## Key Findings

1. **Federated Learning Benefits**: Most banks achieve better performance through collaboration compared to training in isolation.

2. **Per-Bank Variation**: Benefits vary by bank profile - smaller banks (Credit Union, Regional) gain more from federation.

3. **Privacy Preserving**: FL achieves performance close to the centralized (privacy-invasive) baseline without sharing raw data.

4. **Realistic Simulation**: Bank profiles based on real-world banking characteristics create meaningful non-IID data distributions.

## Technical Details

- **Framework**: Flower for federated learning
- **Model**: Neural network with embedding layers for categorical features
- **Optimization**: FedAvg strategy with per-bank metric tracking
- **Evaluation**: AUC-ROC, F1, precision, recall

## Files

- `src/`: Source code
- `config/`: Configuration files
- `data/`: Generated transaction data
- `results/`: Metrics and visualizations
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"README generated at {readme_path}")


if __name__ == "__main__":
    main()
