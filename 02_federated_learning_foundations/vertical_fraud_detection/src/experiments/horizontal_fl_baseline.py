"""
Horizontal Federated Learning baseline.

Implements FedAvg for horizontal FL (same features, different users).
This is for comparison with Vertical FL.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score

from ..utils.metrics import compute_metrics, print_metrics_table


class ClientModel(nn.Module):
    """Client model for horizontal FL."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HorizontalFLBaseline:
    """
    Horizontal Federated Learning with FedAvg.

    Simulates multiple clients with horizontal partitioned data
    (different users, same features).
    """

    def __init__(
        self,
        input_dim: int,
        num_clients: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize Horizontal FL.

        Args:
            input_dim: Number of input features
            num_clients: Number of simulated clients
            device: Device to train on
        """
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.device = device

        # Global model
        self.global_model = ClientModel(input_dim).to(device)

    def _create_client_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[tuple]:
        """
        Split data horizontally among clients.

        Args:
            X: Features (num_samples, input_dim)
            y: Labels (num_samples,)

        Returns:
            List of (X_client, y_client) tuples
        """
        num_samples = len(X)
        samples_per_client = num_samples // self.num_clients

        client_data = []

        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else num_samples

            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))

        return client_data

    def _train_client(
        self,
        model: nn.Module,
        X_client: np.ndarray,
        y_client: np.ndarray,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.01
    ) -> nn.Module:
        """
        Train a single client model.

        Args:
            model: Client model (copy of global model)
            X_client: Client data
            y_client: Client labels
            epochs: Local training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Updated client model
        """
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        num_samples = len(X_client)
        num_batches = num_samples // batch_size + 1

        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                if start_idx >= num_samples:
                    break

                batch_x = torch.FloatTensor(X_client[start_idx:end_idx]).to(self.device)
                batch_y = torch.LongTensor(y_client[start_idx:end_idx]).to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def _fedavg_aggregate(
        self,
        client_models: List[nn.Module],
        client_sizes: List[int]
    ) -> None:
        """
        Aggregate client models using FedAvg.

        Args:
            client_models: List of client models
            client_sizes: Number of samples per client
        """
        # Calculate weights based on data size
        total_samples = sum(client_sizes)
        weights = [size / total_samples for size in client_sizes]

        # Get global model state dict
        global_state = self.global_model.state_dict()

        # Initialize aggregated state
        aggregated_state = copy.deepcopy(global_state)
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.zeros_like(aggregated_state[key])

        # Weighted average of client parameters
        for client_model, weight in zip(client_models, weights):
            client_state = client_model.state_dict()
            for key in aggregated_state.keys():
                aggregated_state[key] += weight * client_state[key]

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_rounds: int = 20,
        local_epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.01
    ) -> Dict:
        """
        Train Horizontal FL model.

        Args:
            X_train: Training features (combined)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_rounds: Number of FedAvg rounds
            local_epochs: Local training epochs per round
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Dictionary with training history
        """
        print(f"\n=== Horizontal FL Training ===")
        print(f"Number of clients: {self.num_clients}")
        print(f"Rounds: {num_rounds}")
        print(f"Local epochs: {local_epochs}\n")

        # Create client data (horizontal split)
        client_data = self._create_client_data(X_train, y_train)
        client_sizes = [len(y) for _, y in client_data]

        history = {'val_loss': [], 'val_auc': []}

        best_val_auc = 0.0

        for round_idx in range(num_rounds):
            print(f"Round [{round_idx+1}/{num_rounds}]")

            # Train each client
            client_models = []
            for client_idx, (X_client, y_client) in enumerate(client_data):
                client_model = copy.deepcopy(self.global_model)
                client_model = self._train_client(
                    client_model, X_client, y_client,
                    epochs=local_epochs, batch_size=batch_size, lr=lr
                )
                client_models.append(client_model)

            # Aggregate models
            self._fedavg_aggregate(client_models, client_sizes)

            # Validate global model
            self.global_model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(X_val).to(self.device)
                val_outputs = self.global_model(val_x)
                val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

            val_loss = nn.CrossEntropyLoss()(
                val_outputs,
                torch.LongTensor(y_val).to(self.device)
            ).item()

            val_auc = roc_auc_score(y_val, val_probs)
            best_val_auc = max(best_val_auc, val_auc)

            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        return {
            'history': history,
            'best_val_auc': best_val_auc
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate global model on test set."""
        self.global_model.eval()

        with torch.no_grad():
            test_x = torch.FloatTensor(X_test).to(self.device)
            outputs = self.global_model(test_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.cpu().numpy(), axis=1)

        metrics = compute_metrics(y_test, preds, probs)
        metrics['auc_roc'] = roc_auc_score(y_test, probs)

        return metrics


def run_horizontal_fl_baseline(
    X_a_train: np.ndarray,
    X_a_test: np.ndarray,
    X_b_train: np.ndarray,
    X_b_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_clients: int = 3,
    num_rounds: int = 20,
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Run Horizontal FL baseline.

    Note: Horizontal FL requires combined features (same feature space).
    We use combined features for this baseline.

    Args:
        X_a_train: Party A training features
        X_a_test: Party A test features
        X_b_train: Party B training features
        X_b_test: Party B test features
        y_train: Training labels
        y_test: Test labels
        num_clients: Number of clients
        num_rounds: Number of FL rounds
        device: Device to train on

    Returns:
        Dictionary of metrics
    """
    print("\n=== Horizontal FL Baseline ===")
    print("Note: Horizontal FL uses combined features (same feature space)")
    print("This is different from Vertical FL where parties have different features\n")

    # Combine features for horizontal FL
    X_combined_train = np.concatenate([X_a_train, X_b_train], axis=1)
    X_combined_test = np.concatenate([X_a_test, X_b_test], axis=1)

    # Run Horizontal FL
    hfl = HorizontalFLBaseline(
        input_dim=X_combined_train.shape[1],
        num_clients=num_clients,
        device=device
    )

    hfl.train(
        X_combined_train, y_train,
        X_combined_test, y_test,
        num_rounds=num_rounds
    )

    metrics = hfl.evaluate(X_combined_test, y_test)

    results = {'Horizontal FL': metrics}
    print_metrics_table(results, "Horizontal FL Results")

    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Horizontal FL Baseline...")

    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 1000, 500

    X_a_train = np.random.randn(n_train, 7)
    X_b_train = np.random.randn(n_train, 3)
    y_train = np.random.randint(0, 2, n_train)

    X_a_test = np.random.randn(n_test, 7)
    X_b_test = np.random.randn(n_test, 3)
    y_test = np.random.randint(0, 2, n_test)

    results = run_horizontal_fl_baseline(
        X_a_train, X_a_test, X_b_train, X_b_test, y_train, y_test,
        num_clients=3, num_rounds=10
    )

    print("\n✓ Horizontal FL baseline working correctly")
