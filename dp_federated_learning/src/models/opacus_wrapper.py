"""
Opacus Wrapper for Validation
==============================

Wrapper for Opacus library to validate our custom DP-SGD implementation.

Opacus: https://opacus.ai/
Reference: Opacus research paper
"""

import torch
import torch.nn as nn
from typing import Optional, Any

try:
    from opacus import PrivacyEngine
    from opacus.grad_sample import GradSampleModule
    from opacus.optimizers import DPOptimizer
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None
    GradSampleModule = None
    DPOptimizer = None


class OpacusWrapper:
    """Wrapper for Opacus DP-SGD implementation.

    This provides a consistent interface with our custom DPSGDOptimizer
    to enable fair comparison and validation.

    Usage:
        >>> if OPACUS_AVAILABLE:
        ...     wrapper = OpacusWrapper(
        ...         model=model,
        ...         noise_multiplier=1.5,
        ...         clipping_bound=1.0,
        ...         batch_size=32,
        ...         lr=0.01,
        ...         dataset_size=1000
        ...     )
        ...     for x_batch, y_batch in dataloader:
        ...         loss = wrapper.step(x_batch, y_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float,
        clipping_bound: float,
        batch_size: int,
        lr: float,
        dataset_size: int,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        delta: float = 1e-5
    ):
        """Initialize Opacus wrapper.

        Args:
            model: PyTorch model
            noise_multiplier: σ
            clipping_bound: C (max grad norm in Opacus)
            batch_size: B
            lr: Learning rate
            dataset_size: Total training set size n
            momentum: Momentum coefficient
            weight_decay: L2 regularization
            delta: Privacy parameter δ

        Raises:
            ImportError: If Opacus is not installed
        """
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is not installed. Install it with:\n"
                "  pip install opacus\n"
                "or:\n  pip install opacus --upgrade"
            )

        self.model = model
        self.noise_multiplier = noise_multiplier
        self.clipping_bound = clipping_bound
        self.batch_size = batch_size
        self.lr = lr
        self.dataset_size = dataset_size
        self.delta = delta

        # Create standard optimizer
        self.base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Attach privacy engine
        self.privacy_engine = PrivacyEngine(secure_mode=False)

        # Convert model to GradSampleModule
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=self.base_optimizer,
            data_loader=None,  # Will be set during training
            noise_multiplier=noise_multiplier,
            max_grad_norm=clipping_bound,
            batch_size=batch_size
        )

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Execute one training step.

        Args:
            inputs: Batch of inputs
            targets: Batch of targets

        Returns:
            loss: Loss value
        """
        # Forward pass
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step (includes clipping and noise)
        self.optimizer.step()

        return loss

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get total privacy budget consumed.

        Returns:
            (epsilon, delta): Total (ε, δ) spent
        """
        # Opacus computes privacy automatically
        # Note: The API might vary between Opacus versions
        try:
            # Opacus 1.x
            epsilon = self.optimizer.accountant.get_epsilon(self.delta)
            return epsilon, self.delta
        except AttributeError:
            # Fallback for different versions
            try:
                # Try alternative method
                privacy_dict = self.privacy_engine.get_privacy_spent(
                    self.delta,
                    len(self.optimizer.accountant.steps)
                )
                return privacy_dict[0], self.delta
            except Exception:
                # If all else fails, return placeholder
                return 0.0, self.delta

    def attach_data_loader(self, data_loader):
        """Attach data loader for sample rate computation."""
        self.data_loader = data_loader


def compare_implementations(
    model_custom: nn.Module,
    model_opacus: nn.Module,
    noise_multiplier: float,
    clipping_bound: float,
    batch_size: int,
    lr: float,
    dataset_size: int,
    num_steps: int = 10
) -> dict[str, Any]:
    """Compare custom DP-SGD with Opacus implementation.

    Args:
        model_custom: Model for custom implementation
        model_opacus: Model for Opacus implementation
        noise_multiplier: σ
        clipping_bound: C
        batch_size: B
        lr: Learning rate
        dataset_size: n
        num_steps: Number of steps to compare

    Returns:
        comparison: Dictionary with comparison results

    Raises:
        ImportError: If Opacus is not available
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is not installed. Cannot compare.")

    from .dp_sgd_custom import create_dp_sgd_optimizer

    # Create optimizers
    custom_opt = create_dp_sgd_optimizer(
        model=model_custom,
        noise_multiplier=noise_multiplier,
        clipping_bound=clipping_bound,
        batch_size=batch_size,
        lr=lr,
        dataset_size=dataset_size
    )

    opacus_wrapper = OpacusWrapper(
        model=model_opacus,
        noise_multiplier=noise_multiplier,
        clipping_bound=clipping_bound,
        batch_size=batch_size,
        lr=lr,
        dataset_size=dataset_size
    )

    results = {
        'custom_epsilon': [],
        'opacus_epsilon': [],
        'custom_loss': [],
        'opacus_loss': []
    }

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    for step in range(num_steps):
        # Generate same random data for both
        inputs = torch.randn(batch_size, 10)
        targets = torch.randint(0, 2, (batch_size,))

        # Custom implementation
        custom_metrics = custom_opt.step(inputs, targets, loss_fn)
        results['custom_epsilon'].append(custom_metrics.epsilon_spent)

        # Opacus implementation
        with torch.no_grad():
            outputs = model_opacus(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
        results['opacus_loss'].append(loss.item())

        opacus_loss = opacus_wrapper.step(inputs, targets)
        results['opacus_loss'].append(opacus_loss.item())

        # Track privacy
        custom_eps, _ = custom_opt.get_privacy_spent()
        opacus_eps, _ = opacus_wrapper.get_privacy_spent()

        results['custom_epsilon'].append(custom_eps)
        try:
            results['opacus_epsilon'].append(opacus_eps)
        except Exception:
            # Opacus privacy tracking might fail
            results['opacus_epsilon'].append(0.0)

    return results


def install_opacus_instructions() -> str:
    """Return instructions for installing Opacus."""
    return """
To install Opacus for validation:

    pip install opacus

Or for the latest version:

    pip install opacus --upgrade

Opacus GitHub: https://github.com/pytorch/opacus
Documentation: https://opacus.ai/
"""


if __name__ == "__main__":
    if OPACUS_AVAILABLE:
        print("Opacus is available!")

        # Test comparison
        model_custom = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        model_opacus = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        # Copy weights
        model_opacus.load_state_dict(model_custom.state_dict())

        try:
            results = compare_implementations(
                model_custom=model_custom,
                model_opacus=model_opacus,
                noise_multiplier=1.5,
                clipping_bound=1.0,
                batch_size=32,
                lr=0.01,
                dataset_size=1000,
                num_steps=5
            )

            print("\nComparison results:")
            print(f"Custom ε after 5 steps: {results['custom_epsilon'][-1]:.4f}")
            try:
                print(f"Opacus ε after 5 steps: {results['opacus_epsilon'][-1]:.4f}")
            except Exception:
                print("Opacus ε tracking not available")
        except Exception as e:
            print(f"Comparison failed: {e}")
    else:
        print("Opacus is not available.")
        print(install_opacus_instructions())
