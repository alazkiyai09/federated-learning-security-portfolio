#!/usr/bin/env python3
"""
Demo script for Fraud Model Explainability.

This script demonstrates how to:
1. Create a sample fraud detection model
2. Train an XGBoost classifier
3. Generate SHAP and LIME explanations
4. Create an HTML report
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using RandomForest instead.")
    from sklearn.ensemble import RandomForestClassifier

from api import create_explainer
from utils import format_risk_factors
from reports import ReportGenerator


def create_sample_fraud_data(n_samples=1000, n_features=15):
    """
    Create a synthetic fraud detection dataset.

    Features represent typical fraud detection variables:
    - Transaction amount
    - Time since last transaction
    - Geographic indicators
    - Device-related features
    - Historical patterns
    """
    print("Creating synthetic fraud detection dataset...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # 10% fraud rate (imbalanced)
        random_state=42
    )

    # Create meaningful feature names
    feature_names = [
        'transaction_amount',
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        'days_since_last_txn',
        'avg_txn_amount_7d',
        'txn_count_24h',
        'txn_count_7d',
        'is_international',
        'distance_from_home',
        'device_age_days',
        'has_ip_change',
        'email_domain_match',
        'shipping_billing_match',
        'risk_score_previous'
    ]

    return X, y, feature_names


def train_fraud_model(X_train, y_train, model_type='xgboost'):
    """Train a fraud detection model."""
    print(f"\nTraining {model_type} model...")

    if model_type == 'xgboost' and HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    model.fit(X_train, y_train)
    print("Model training complete!")

    return model, model_type if HAS_XGBOOST else 'random_forest'


def demonstrate_local_explanation(explainer, X_test, feature_names, model):
    """Demonstrate local explanation for a single transaction."""
    print("\n" + "="*60)
    print("LOCAL EXPLANATION DEMONSTRATION")
    print("="*60)

    # Pick a fraudulent transaction
    fraud_proba = model.predict_proba(X_test)[:, 1]
    fraud_idx = np.argmax(fraud_proba)

    X_sample = X_test[fraud_idx]
    fraud_prob = fraud_proba[fraud_idx]

    print(f"\nTransaction ID: TXN-{fraud_idx}")
    print(f"Fraud Probability: {fraud_prob:.2%}")
    print(f"Classification: {'FRAUD' if fraud_prob >= 0.5 else 'LEGITIMATE'}")

    # Generate local explanation
    print("\nGenerating local explanation...")
    local_exp = explainer.explain_local(X_sample, feature_names)

    # Format risk factors
    risk_factors = format_risk_factors(local_exp, top_n=5)

    print("\nTop 5 Risk Factors:")
    print("-" * 60)
    for i, factor in enumerate(risk_factors, 1):
        print(f"{i}. {factor['description']}")
        print(f"   Direction: {factor['direction']} fraud risk")
        print(f"   Impact: {factor['impact_level']} (score: {factor['importance']:.4f})")
        print()

    return X_sample, fraud_prob, risk_factors


def demonstrate_global_explanation(explainer, X_test, feature_names):
    """Demonstrate global feature importance."""
    print("\n" + "="*60)
    print("GLOBAL EXPLANATION DEMONSTRATION")
    print("="*60)

    print("\nGenerating global feature importance...")
    global_exp = explainer.explain_global(X_test[:200], feature_names)

    print("\nTop 10 Most Important Features (Global):")
    print("-" * 60)
    for i, (feature, importance) in enumerate(list(global_exp.items())[:10], 1):
        print(f"{i:2d}. {feature:30s} | Importance: {importance:.4f}")


def generate_html_report(transaction_id, prediction, risk_factors, model, X_test, feature_names, explainer):
    """Generate an HTML report."""
    print("\n" + "="*60)
    print("GENERATING HTML REPORT")
    print("="*60)

    # Get global importance for report
    global_exp = explainer.explain_global(X_test[:100], feature_names)
    global_factors = format_risk_factors(global_exp, top_n=10)

    # Model metadata
    model_metadata = {
        'name': f'{HAS_XGBOOST and "XGBoost" or "Random Forest"} Fraud Detection Model',
        'version': '1.0.0',
        'type': 'xgboost' if HAS_XGBOOST else 'random_forest',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'last_validated': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    # Generate report
    generator = ReportGenerator()
    html_report = generator.generate_html_report(
        transaction_id=transaction_id,
        prediction=prediction,
        predicted_class='Fraud' if prediction >= 0.5 else 'Legitimate',
        risk_factors=risk_factors,
        global_importance=global_factors,
        model_metadata=model_metadata
    )

    # Save report
    output_dir = Path(__file__).parent / 'reports'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'fraud_report_{transaction_id}.html'

    generator.save_report(html_report, str(output_path))

    print(f"\nReport saved to: {output_path}")
    print("Open the HTML file in your browser to view the full report!")


def main():
    """Main demo function."""
    print("="*60)
    print("FRAUD MODEL EXPLAINABILITY DEMO")
    print("="*60)

    # 1. Create data
    X, y, feature_names = create_sample_fraud_data(n_samples=1000, n_features=15)

    # 2. Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 3. Train model
    model, model_type = train_fraud_model(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy: {test_acc:.2%}")

    # 4. Create explainer
    print("\n" + "="*60)
    print("CREATING EXPLAINER")
    print("="*60)

    explainer_type = 'shap' if HAS_XGBOOST else 'lime'
    print(f"\nCreating {explainer_type.upper()} explainer...")

    explainer = create_explainer(
        model=model,
        model_type=model_type,
        explainer_type=explainer_type,
        training_data=X_train[:500],  # Background data for SHAP
        feature_names=feature_names,
        random_state=42
    )

    print(f"{explainer_type.upper()} explainer created successfully!")

    # 5. Demonstrate local explanation
    X_sample, fraud_prob, risk_factors = demonstrate_local_explanation(
        explainer, X_test, feature_names, model
    )

    # 6. Demonstrate global explanation
    demonstrate_global_explanation(explainer, X_test, feature_names)

    # 7. Generate HTML report
    generate_html_report(
        transaction_id='DEMO-001',
        prediction=fraud_prob,
        risk_factors=risk_factors,
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        explainer=explainer
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. View the generated HTML report")
    print("2. Try Streamlit UI: streamlit run app/streamlit_app.py")
    print("3. Run tests: pytest tests/")
    print("4. Check out README.md for more documentation")


if __name__ == '__main__':
    main()
