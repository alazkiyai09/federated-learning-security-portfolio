"""
Vertical Federated Learning experiment.

Main experiment runner for Vertical FL fraud detection.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Optional

from ..training.vertical_fl_trainer import VerticalFLTrainer, TrainingConfig
from ..utils.metrics import compute_metrics, print_metrics_table
from ..utils.visualization import plot_training_history, plot_gradient_leakage


class VerticalFLExperiment:
    """
    Main Vertical FL experiment.

    Compares Vertical FL against single-party and horizontal FL baselines.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_config: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize experiment.

        Args:
            config: Training configuration
            model_config: Model architecture configuration
            device: Device to train on
        """
        self.config = config
        self.model_config = model_config
        self.device = device
        self.trainer = None
        self.results = {}

    def run_vertical_fl(
        self,
        X_a_train: np.ndarray,
        X_b_train: np.ndarray,
        y_train: np.ndarray,
        X_a_val: np.ndarray,
        X_b_val: np.ndarray,
        y_val: np.ndarray,
        X_a_test: np.ndarray,
        X_b_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Run Vertical FL experiment.

        Args:
            X_a_train: Party A training features
            X_b_train: Party B training features
            y_train: Training labels
            X_a_val: Party A validation features
            X_b_val: Party B validation features
            y_val: Validation labels
            X_a_test: Party A test features
            X_b_test: Party B test features
            y_test: Test labels

        Returns:
            Dictionary with experiment results
        """
        print("\n" + "="*80)
        print("VERTICAL FEDERATED LEARNING EXPERIMENT")
        print("="*80)
        print("\nPrivacy Protocol:")
        print("  ✓ Party A raw features: STAY LOCAL")
        print("  ✓ Party B raw features: STAY LOCAL")
        print("  ✓ Server receives: EMBEDDINGS ONLY")
        print("  ✓ Backward pass: EMBEDDING GRADIENTS ONLY")
        print("="*80)

        # Initialize trainer
        self.trainer = VerticalFLTrainer(
            config=self.config,
            model_config=self.model_config,
            device=self.device
        )

        # Train
        history = self.trainer.train(
            X_a_train, X_b_train, y_train,
            X_a_val, X_b_val, y_val
        )

        # Evaluate on test set
        test_metrics = self._evaluate_test(X_a_test, X_b_test, y_test)

        # Store results
        self.results['vertical_fl'] = {
            'history': {
                'train_losses': history.train_losses,
                'train_accuracies': history.train_accuracies,
                'val_losses': history.val_losses,
                'val_accuracies': history.val_accuracies,
                'val_aucs': history.val_aucs,
                'leakage_metrics': history.leakage_metrics,
            },
            'test_metrics': test_metrics
        }

        # Print summary
        print("\n" + "="*80)
        print("VERTICAL FL RESULTS")
        print("="*80)
        print(f"\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        return self.results['vertical_fl']

    def _evaluate_test(
        self,
        X_a_test: np.ndarray,
        X_b_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        self.trainer.split_nn.eval_mode()

        num_samples = len(y_test)
        batch_size = self.config.batch_size
        num_batches = num_samples // batch_size + 1

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                if start_idx >= num_samples:
                    break

                batch_x_a = torch.FloatTensor(X_a_test[start_idx:end_idx]).to(self.device)
                batch_x_b = torch.FloatTensor(X_b_test[start_idx:end_idx]).to(self.device)
                batch_y = torch.LongTensor(y_test[start_idx:end_idx])

                predictions, _, _ = self.trainer.split_nn.forward_pass(batch_x_a, batch_x_b)

                all_predictions.append(predictions.cpu())
                all_labels.append(batch_y)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        preds = all_predictions.argmax(dim=1).numpy()
        probs = all_predictions[:, 1].numpy()

        metrics = compute_metrics(all_labels.numpy(), preds, probs)
        metrics['auc_roc'] = metrics.get('auc_roc', 0.0)
        metrics['auc_pr'] = metrics.get('auc_pr', 0.0)

        return metrics

    def save_results(self, save_dir: str) -> None:
        """Save experiment results."""
        os.makedirs(save_dir, exist_ok=True)

        # Save numerical results
        results_path = os.path.join(save_dir, 'vertical_fl_results.json')
        with open(results_path, 'w') as f:
            # Convert leakage metrics to serializable format
            results = self.results.copy()
            if 'vertical_fl' in results:
                results['vertical_fl']['history']['leakage_metrics'] = [
                    {k: v for k, v in lm.items() if k != 'party_a' and k != 'party_b'}
                    for lm in results['vertical_fl']['history']['leakage_metrics']
                ]
            json.dump(results, f, indent=2)

        # Save training history plot
        if 'vertical_fl' in self.results:
            history = self.results['vertical_fl']['history']
            plot_training_history(
                history,
                save_path=os.path.join(save_dir, 'training_history.png'),
                title='Vertical FL Training History'
            )

            # Save gradient leakage plot
            if history['leakage_metrics']:
                plot_gradient_leakage(
                    history['leakage_metrics'],
                    save_path=os.path.join(save_dir, 'gradient_leakage.png')
                )

        # Save models
        if self.trainer:
            self.trainer.save_models(save_dir)

        print(f"\nResults saved to: {save_dir}")

    def print_summary(self) -> None:
        """Print experiment summary."""
        if 'vertical_fl' not in self.results:
            print("No results to display.")
            return

        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)

        result = self.results['vertical_fl']
        history = result['history']
        test_metrics = result['test_metrics']

        print(f"\nTraining Summary:")
        print(f"  Epochs completed: {len(history['train_losses'])}")
        print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
        print(f"  Final val loss: {history['val_losses'][-1]:.4f}")
        print(f"  Best val AUC: {max(history['val_aucs']):.4f}")

        if history['leakage_metrics']:
            final_leakage = history['leakage_metrics'][-1]
            print(f"\nGradient Leakage Risk (Final Epoch):")
            print(f"  Party A: {final_leakage['party_a']['leakage_risk_percent']:.1f}%")
            print(f"  Party B: {final_leakage['party_b']['leakage_risk_percent']:.1f}%")

        print(f"\nTest Set Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\n" + "="*80)


def run_full_comparison(
    X_a_train: np.ndarray,
    X_a_test: np.ndarray,
    X_b_train: np.ndarray,
    X_b_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Run full comparison: Vertical FL vs baselines.

    Args:
        X_a_train: Party A training features
        X_a_test: Party A test features
        X_b_train: Party B training features
        X_b_test: Party B test features
        y_train: Training labels
        y_test: Test labels
        device: Device to train on

    Returns:
        Dictionary of all results
    """
    all_results = {}

    # Create validation split from training data
    from sklearn.model_selection import train_test_split

    X_a_train, X_a_val, X_b_train, X_b_val, y_train, y_val = train_test_split(
        X_a_train, X_b_train, y_train,
        test_size=0.15, random_state=42, stratify=y_train
    )

    # 1. Single-party baselines
    from .single_party_baseline import run_single_party_baselines

    baseline_results = run_single_party_baselines(
        X_a_train, X_a_test, X_b_train, X_b_test, y_train, y_test, device
    )
    all_results.update(baseline_results)

    # 2. Horizontal FL baseline
    from .horizontal_fl_baseline import run_horizontal_fl_baseline

    hfl_results = run_horizontal_fl_baseline(
        X_a_train, X_a_test, X_b_train, X_b_test, y_train, y_test,
        num_clients=3, num_rounds=20, device=device
    )
    all_results.update(hfl_results)

    # 3. Vertical FL
    config = TrainingConfig(
        num_epochs=50,
        batch_size=256,
        learning_rate=0.001,
        early_stopping_patience=10,
        analyze_gradient_leakage=True
    )

    vfl_exp = VerticalFLExperiment(config=config, device=device)
    vfl_exp.run_vertical_fl(
        X_a_train, X_b_train, y_train,
        X_a_val, X_b_val, y_val,
        X_a_test, X_b_test, y_test
    )

    all_results['Vertical FL'] = vfl_exp.results['vertical_fl']['test_metrics']

    # Print final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    comparison_results = {
        name: {k: v for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']}
        for name, metrics in all_results.items()
    }

    print_metrics_table(comparison_results, "All Methods Comparison")

    return all_results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Vertical FL Experiment...")

    # Generate synthetic data
    np.random.seed(42)
    n_train, n_val, n_test = 1000, 200, 500

    X_a_train = np.random.randn(n_train, 7)
    X_b_train = np.random.randn(n_train, 3)
    y_train = np.random.randint(0, 2, n_train)

    X_a_val = np.random.randn(n_val, 7)
    X_b_val = np.random.randn(n_val, 3)
    y_val = np.random.randint(0, 2, n_val)

    X_a_test = np.random.randn(n_test, 7)
    X_b_test = np.random.randn(n_test, 3)
    y_test = np.random.randint(0, 2, n_test)

    # Quick test
    config = TrainingConfig(
        num_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10
    )

    exp = VerticalFLExperiment(config=config, device='cpu')
    exp.run_vertical_fl(
        X_a_train, X_b_train, y_train,
        X_a_val, X_b_val, y_val,
        X_a_test, X_b_test, y_test
    )

    exp.print_summary()

    print("\n✓ Vertical FL experiment working correctly")
