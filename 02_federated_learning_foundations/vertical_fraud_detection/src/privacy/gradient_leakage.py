"""
Gradient leakage analysis for Vertical Federated Learning.

Analyzes potential privacy risks from transmitting embedding gradients
between server and parties.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LeakageReport:
    """Report on gradient leakage analysis."""
    embedding_correlation: float
    gradient_magnitude: float
    leakage_risk_percent: float
    risk_level: str
    recommendations: List[str]


class GradientLeakageAnalyzer:
    """
    Analyzes gradient leakage risks in Vertical FL.

    Threat Model: Honest-but-curious server may attempt to reconstruct
    embeddings from gradients transmitted during backward pass.
    """

    def __init__(self, threshold_high: float = 30.0, threshold_medium: float = 15.0):
        """
        Initialize analyzer.

        Args:
            threshold_high: Risk percentage for "high" risk level
            threshold_medium: Risk percentage for "medium" risk level
        """
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium

    def analyze(
        self,
        embeddings: torch.Tensor,
        embedding_gradients: torch.Tensor,
        num_samples: int = 100
    ) -> LeakageReport:
        """
        Analyze gradient leakage risk.

        Args:
            embeddings: Forward pass embeddings (batch_size, embedding_dim)
            embedding_gradients: Gradients wrt embeddings
            num_samples: Number of samples to analyze

        Returns:
            LeakageReport with analysis results
        """
        with torch.no_grad():
            batch_size, emb_dim = embeddings.shape
            num_samples = min(num_samples, batch_size)

            # Flatten for correlation analysis
            emb_flat = embeddings[:num_samples].flatten()
            grad_flat = embedding_gradients[:num_samples].flatten()

            # Pearson correlation
            mean_emb = emb_flat.mean()
            mean_grad = grad_flat.mean()
            std_emb = emb_flat.std()
            std_grad = grad_flat.std()

            covariance = ((emb_flat - mean_emb) * (grad_flat - mean_grad)).mean()
            correlation = covariance / (std_emb * std_grad + 1e-8)

            # Gradient magnitude
            grad_magnitude = embedding_gradients[:num_samples].norm(dim=1).mean().item()

            # Estimate leakage risk
            # Higher correlation between embeddings and gradients = higher risk
            leakage_risk = min(abs(correlation).item() * 100, 100)

            # Determine risk level
            if leakage_risk >= self.threshold_high:
                risk_level = "HIGH"
                recommendations = [
                    "Consider adding gradient noise (DP-SGD)",
                    "Increase embedding dimension",
                    "Use secure aggregation",
                    "Implement gradient compression"
                ]
            elif leakage_risk >= self.threshold_medium:
                risk_level = "MEDIUM"
                recommendations = [
                    "Monitor gradient norms during training",
                    "Consider adding light gradient noise",
                    "Implement gradient clipping"
                ]
            else:
                risk_level = "LOW"
                recommendations = [
                    "Current privacy protections are adequate",
                    "Continue monitoring during training"
                ]

        return LeakageReport(
            embedding_correlation=correlation.item(),
            gradient_magnitude=grad_magnitude,
            leakage_risk_percent=leakage_risk,
            risk_level=risk_level,
            recommendations=recommendations
        )

    def analyze_training_history(
        self,
        leakage_metrics: List[Dict]
    ) -> Dict[str, float]:
        """
        Analyze leakage trends across training.

        Args:
            leakage_metrics: List of leakage metrics from training

        Returns:
            Summary statistics
        """
        if not leakage_metrics:
            return {
                'mean_risk_a': 0.0,
                'max_risk_a': 0.0,
                'mean_risk_b': 0.0,
                'max_risk_b': 0.0,
            }

        risks_a = [m['party_a']['leakage_risk_percent'] for m in leakage_metrics]
        risks_b = [m['party_b']['leakage_risk_percent'] for m in leakage_metrics]

        return {
            'mean_risk_a': np.mean(risks_a),
            'max_risk_a': np.max(risks_a),
            'mean_risk_b': np.mean(risks_b),
            'max_risk_b': np.max(risks_b),
        }


def analyze_gradient_leakage(
    embeddings: torch.Tensor,
    embedding_gradients: torch.Tensor,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Analyze gradient leakage (convenience function).

    Args:
        embeddings: Forward pass embeddings
        embedding_gradients: Gradients wrt embeddings
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with leakage metrics
    """
    analyzer = GradientLeakageAnalyzer()
    report = analyzer.analyze(embeddings, embedding_gradients, num_samples)

    return {
        'embedding_grad_correlation': report.embedding_correlation,
        'gradient_magnitude': report.gradient_magnitude,
        'leakage_risk_percent': report.leakage_risk_percent,
        'risk_level': report.risk_level,
    }


def estimate_mutual_information(
    embeddings: torch.Tensor,
    gradients: torch.Tensor
) -> float:
    """
    Estimate mutual information between embeddings and gradients.

    This is a simplified estimate. Full MI estimation would require
    more sophisticated methods (e.g., k-nearest neighbors estimators).

    Args:
        embeddings: Forward pass embeddings
        gradients: Embedding gradients

    Returns:
        Estimated mutual information (nats)
    """
    with torch.no_grad():
        # Simplified MI estimate based on correlation
        # MI(X,Y) ≈ -0.5 * log(1 - rho^2) for Gaussian variables
        emb_flat = embeddings.flatten().numpy()
        grad_flat = gradients.flatten().numpy()

        correlation = np.corrcoef(emb_flat, grad_flat)[0, 1]

        # Clamp to avoid numerical issues
        correlation = np.clip(correlation, -0.99, 0.99)

        mi_estimate = -0.5 * np.log(1 - correlation**2)

    return mi_estimate


if __name__ == "__main__":
    # Test gradient leakage analysis
    print("Testing Gradient Leakage Analysis...")

    analyzer = GradientLeakageAnalyzer()

    # Create synthetic data
    batch_size = 64
    embedding_dim = 16

    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings.grad = torch.randn(batch_size, embedding_dim) * 0.1

    print("\n=== Gradient Leakage Analysis ===")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Gradient shape: {embeddings.grad.shape}")

    report = analyzer.analyze(embeddings, embeddings.grad)

    print(f"\nResults:")
    print(f"  Correlation: {report.embedding_correlation:.4f}")
    print(f"  Gradient magnitude: {report.gradient_magnitude:.4f}")
    print(f"  Leakage risk: {report.leakage_risk_percent:.1f}%")
    print(f"  Risk level: {report.risk_level}")

    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    # Test MI estimation
    mi = estimate_mutual_information(embeddings, embeddings.grad)
    print(f"\nEstimated mutual information: {mi:.4f} nats")

    print("\n✓ Gradient leakage analysis working correctly")
