"""
Federated Server implementation.

Handles client selection, weight aggregation, and federated training coordination.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import random
import numpy as np

from src.utils import serialize_weights, deserialize_weights, get_device
from src.client import FederatedClient


StateDict = Dict[str, torch.Tensor]
ClientUpdates = List[Tuple[StateDict, int]]


class FederatedServer:
    """
    Federated learning server for coordinating training across clients.

    Implements the FedAvg algorithm:
    1. Sample fraction of clients
    2. Send global weights to selected clients
    3. Clients perform local training and return updates
    4. Aggregate updates using weighted averaging by sample count

    Attributes:
        model: Global model
        config: Server configuration dictionary
        round: Current training round

    Example:
        >>> server = FederatedServer(model, config={'num_rounds': 100})
        >>> for round in range(100):
        ...     selected_clients = server.select_clients(all_clients, fraction=0.1)
        ...     metrics = server.federated_round(selected_clients, round)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize federated server.

        Args:
            model: Global PyTorch model to train
            config: Server configuration with keys:
                - num_rounds: Total number of federated rounds
                - client_fraction: Fraction of clients to sample each round
                - min_learning_rate: Minimum learning rate for decay (optional)
                - decay_rate: Learning rate decay per round (optional)
        """
        self.model = model
        self.config = config

        # Training configuration
        self.num_rounds = config.get('num_rounds', 100)
        self.client_fraction = config.get('client_fraction', 0.1)

        # Learning rate scheduling (optional)
        self.initial_lr = config.get('learning_rate', 0.01)
        self.min_lr = config.get('min_learning_rate', None)
        self.decay_rate = config.get('decay_rate', None)

        # Device
        self.device = get_device()
        self.model.to(self.device)

        # Training state
        self.round = 0
        self.global_weights = serialize_weights(self.model)

        # History
        self.metrics_history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }

    def select_clients(
        self,
        all_clients: List[FederatedClient],
        fraction: Optional[float] = None,
        round_num: Optional[int] = None
    ) -> List[FederatedClient]:
        """
        Select subset of clients for training round.

        Implements random client sampling as per FedAvg paper.
        With m = max(C * K, 1) where C is fraction and K is total clients.

        Args:
            all_clients: List of all available clients
            fraction: Fraction of clients to select (defaults to self.client_fraction)
            round_num: Current round number (for reproducibility)

        Returns:
            List[FederatedClient]: Selected clients for this round
        """
        if fraction is None:
            fraction = self.client_fraction

        num_clients = len(all_clients)
        num_selected = max(int(fraction * num_clients), 1)

        # Use round number as seed for reproducibility
        if round_num is not None:
            random.seed(round_num)
        else:
            random.seed()

        selected_indices = random.sample(range(num_clients), num_selected)
        selected_clients = [all_clients[i] for i in selected_indices]

        return selected_clients

    def aggregate_weights(
        self,
        client_updates: ClientUpdates
    ) -> StateDict:
        """
        Aggregate client updates using FedAvg weighted averaging.

        Implements the core FedAvg algorithm:
        w_new = sum(n_k / n_total * w_k) for all clients k

        where n_k is the number of samples on client k.

        Args:
            client_updates: List of (client_weights, num_samples) tuples

        Returns:
            StateDict: Aggregated global weights
        """
        if not client_updates:
            return self.global_weights

        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in client_updates)

        # Initialize aggregated weights
        aggregated = {}
        first_weights = client_updates[0][0]

        for key in first_weights.keys():
            # Weighted average: sum(n_k / n_total * w_k)
            aggregated[key] = torch.zeros_like(first_weights[key])

            for client_weights, num_samples in client_updates:
                weight_fraction = num_samples / total_samples
                aggregated[key] += weight_fraction * client_weights[key]

        return aggregated

    def federated_round(
        self,
        clients: List[FederatedClient],
        round_num: int,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Execute one round of federated learning.

        Steps:
        1. Send current global weights to selected clients
        2. Clients perform local training
        3. Aggregate client updates
        4. Update global model
        5. Evaluate on test set (if provided)

        Args:
            clients: Selected clients for this round
            round_num: Current round number
            test_loader: Optional test data for evaluation
            verbose: Whether to print progress

        Returns:
            Dict[str, float]: Round metrics
        """
        self.round = round_num

        # Collect client updates
        client_updates = []
        train_losses = []
        train_accuracies = []

        if verbose:
            print(f"\n=== Round {round_num + 1}/{self.num_rounds} ===")
            print(f"Training with {len(clients)} clients")

        for client in clients:
            # Local training
            local_weights, num_samples = client.local_train(
                self.global_weights,
                verbose=False
            )

            # Store update
            client_updates.append((local_weights, num_samples))

            # Collect training metrics
            client_metrics = client.evaluate(client.train_loader, verbose=False)
            train_losses.append(client_metrics['loss'])
            train_accuracies.append(client_metrics['accuracy'])

        # Aggregate updates
        self.global_weights = self.aggregate_weights(client_updates)

        # Update global model
        deserialize_weights(self.model, self.global_weights)

        # Calculate average training metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        # Evaluate on test set
        test_loss = None
        test_accuracy = None

        if test_loader is not None:
            test_metrics = self.evaluate(test_loader, verbose=False)
            test_loss = test_metrics['loss']
            test_accuracy = test_metrics['accuracy']

            if verbose:
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        # Update learning rate (optional decay)
        if self.decay_rate is not None and self.min_lr is not None:
            new_lr = max(self.min_lr, self.initial_lr * (self.decay_rate ** round_num))
            for client in clients:
                for param_group in client.optimizer.param_groups:
                    param_group['lr'] = new_lr

        # Record history
        self.metrics_history['round'].append(round_num)
        self.metrics_history['train_loss'].append(avg_train_loss)
        self.metrics_history['train_accuracy'].append(avg_train_accuracy)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_accuracy'].append(test_accuracy)

        metrics = {
            'round': round_num,
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

        return metrics

    def evaluate(
        self,
        test_loader: DataLoader,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.

        Args:
            test_loader: Test data DataLoader
            verbose: Whether to print results

        Returns:
            Dict[str, float]: Test metrics
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        }

        if verbose:
            print(f"Global Model | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        return metrics

    def get_metrics_history(self) -> Dict[str, List]:
        """
        Get training history across all rounds.

        Returns:
            Dict with keys: 'round', 'train_loss', 'train_accuracy',
                           'test_loss', 'test_accuracy'
        """
        return self.metrics_history

    def save_global_model(self, path: str) -> None:
        """
        Save global model weights.

        Args:
            path: Path to save model
        """
        torch.save({
            'round': self.round,
            'model_state_dict': self.global_weights,
            'config': self.config
        }, path)

        if hasattr(self, 'metrics_history'):
            # Also save metrics
            import json
            metrics_path = path.replace('.pt', '_metrics.json')
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, values in self.metrics_history.items():
                serializable_metrics[key] = [float(v) if v is not None else None for v in values]

            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
