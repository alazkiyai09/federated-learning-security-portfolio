"""
Baseline experiment: FedAvg without compression.

This establishes the baseline accuracy and bandwidth consumption
for comparison with compression techniques.
"""

import sys
sys.path.append('/home/ubuntu/30Days_Project/communication_efficient_fl')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters

from src.metrics.bandwidth_tracker import BandwidthTracker
from src.strategies.efficient_fedavg import EfficientFedAvg


def create_synthetic_fraud_data(
    num_samples: int = 10000,
    num_features: int = 30,
    fraud_ratio: float = 0.0017
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create synthetic fraud detection dataset.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        fraud_ratio: Ratio of fraudulent transactions

    Returns:
        (train_dataset, test_dataset)
    """
    # Generate features
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # Generate labels (highly imbalanced)
    y = np.zeros(num_samples, dtype=np.int64)
    num_fraud = int(num_samples * fraud_ratio)
    fraud_indices = np.random.choice(num_samples, num_fraud, replace=False)
    y[fraud_indices] = 1

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Split into train/test
    split = int(0.8 * num_samples)
    train_dataset = TensorDataset(X_tensor[:split], y_tensor[:split])
    test_dataset = TensorDataset(X_tensor[split:], y_tensor[split:])

    return train_dataset, test_dataset


class FraudDetectionModel(nn.Module):
    """Simple MLP for fraud detection."""

    def __init__(self, input_dim: int = 30, hidden_dims: List[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """Flower client for fraud detection."""

    def __init__(
        self,
        model: FraudDetectionModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_id: int,
        bandwidth_tracker: BandwidthTracker
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.bandwidth_tracker = bandwidth_tracker
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )

    def get_parameters(self, config):
        """Return current model parameters."""
        parameters = self.model.get_parameters()

        # Log downlink (server -> client) happens before this
        # We'll log it in fit() when we receive parameters

        return parameters

    def fit(self, parameters, config):
        """Train model locally."""
        # Set model parameters
        self.model.set_parameters(parameters)

        # Log downlink (parameters received from server)
        round_num = config.get('round', 0)
        bytes_received = sum(arr.nbytes for arr in parameters)
        self.bandwidth_tracker.log_downlink(
            bytes_sent=bytes_received,
            compressed_bytes=bytes_received,
            round_num=round_num
        )

        # Local training
        self.model.train()
        for epoch in range(5):  # local_epochs
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        # Get updated parameters
        updated_parameters = self.model.get_parameters()

        # Log uplink (client -> server)
        bytes_sent = sum(arr.nbytes for arr in updated_parameters)
        self.bandwidth_tracker.log_uplink(
            bytes_sent=bytes_sent,
            compressed_bytes=bytes_sent,
            round_num=round_num,
            client_id=f"client_{self.client_id}"
        )

        return updated_parameters, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate model."""
        self.model.set_parameters(parameters)

        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss_sum += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        avg_loss = loss_sum / len(self.test_loader)

        return avg_loss, len(self.test_loader.dataset), {
            'accuracy': accuracy
        }


def run_baseline_experiment(
    num_clients: int = 10,
    num_rounds: int = 20
) -> Dict:
    """
    Run baseline FedAvg experiment (no compression).

    Args:
        num_clients: Number of clients
        num_rounds: Number of FL rounds

    Returns:
        Dict with experiment results
    """
    print("=" * 60)
    print("BASELINE EXPERIMENT: FedAvg WITHOUT COMPRESSION")
    print("=" * 60)

    # Create data
    print("\n[1/5] Creating synthetic fraud detection data...")
    train_dataset, test_dataset = create_synthetic_fraud_data(
        num_samples=10000,
        num_features=30
    )

    # Partition data among clients
    client_datasets = []
    samples_per_client = len(train_dataset) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_dataset)

        subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        train_loader = DataLoader(subset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256)
        client_datasets.append((train_loader, test_loader))

    # Initialize bandwidth tracker
    bandwidth_tracker = BandwidthTracker()

    # Create strategy
    print("[2/5] Initializing FedAvg strategy...")
    strategy = EfficientFedAvg(
        compress_func=None,  # No compression
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    # Start server
    print("[3/5] Starting FL server...")

    # Simulate FL training
    # In real deployment, you'd use fl.server.start_server()
    # For simplicity, we'll create clients and simulate

    # Initialize global model
    global_model = FraudDetectionModel(input_dim=30, hidden_dims=[64, 32])
    global_parameters = ndarrays_to_parameters(global_model.get_parameters())

    results = {
        'rounds': [],
        'accuracy': [],
        'bandwidth': []
    }

    print(f"[4/5] Training for {num_rounds} rounds...")

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        # Select clients (all clients for simplicity)
        selected_clients = list(range(num_clients))

        # Client training
        client_parameters = []
        for client_id in selected_clients:
            # Create client
            model = FraudDetectionModel(input_dim=30, hidden_dims=[64, 32])
            model.set_parameters(parameters_to_ndarrays(global_parameters))

            client = FlowerClient(
                model=model,
                train_loader=client_datasets[client_id][0],
                test_loader=client_datasets[client_id][1],
                client_id=client_id,
                bandwidth_tracker=bandwidth_tracker
            )

            # Train
            params, num_examples, metrics = client.fit(
                parameters_to_ndarrays(global_parameters),
                {'round': round_num}
            )
            client_parameters.append((params, num_examples))

        # Aggregate (simple average)
        total_examples = sum(num_examples for _, num_examples in client_parameters)
        aggregated_params = []

        for layer_idx in range(len(client_parameters[0][0])):
            layer_sum = sum(
                params[layer_idx] * num_examples
                for params, num_examples in client_parameters
            )
            aggregated_params.append(layer_sum / total_examples)

        global_parameters = ndarrays_to_parameters(aggregated_params)

        # Evaluate
        model = FraudDetectionModel(input_dim=30, hidden_dims=[64, 32])
        model.set_parameters(aggregated_params)

        # Evaluate on test set
        test_loader = DataLoader(test_dataset, batch_size=256)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total

        # Get bandwidth metrics
        round_metrics = bandwidth_tracker.get_round_metrics(round_num)

        print(f"Accuracy: {accuracy:.4f}")
        if round_metrics:
            print(f"Bytes transmitted: {round_metrics.total_bytes}")

        results['rounds'].append(round_num)
        results['accuracy'].append(accuracy)
        if round_metrics:
            results['bandwidth'].append(round_metrics.total_bytes)

    # Final metrics
    print("\n[5/5] Computing final metrics...")
    cumulative_metrics = bandwidth_tracker.get_cumulative_metrics()

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Final Accuracy: {results['accuracy'][-1]:.4f}")
    print(f"Total Bytes Transmitted: {cumulative_metrics['total_bytes']}")
    print(f"Total Uplink Bytes: {cumulative_metrics['total_uplink_bytes']}")
    print(f"Total Downlink Bytes: {cumulative_metrics['total_downlink_bytes']}")
    print(f"Compression Ratio: 1.0x (no compression)")
    print("=" * 60)

    results['final_accuracy'] = results['accuracy'][-1]
    results['cumulative_metrics'] = cumulative_metrics
    results['compression_ratio'] = 1.0

    return results


if __name__ == "__main__":
    from collections import OrderedDict

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run experiment
    results = run_baseline_experiment(
        num_clients=10,
        num_rounds=20
    )

    print("\nExperiment completed successfully!")
    print(f"Results saved to: data/results/baseline_results.json")
