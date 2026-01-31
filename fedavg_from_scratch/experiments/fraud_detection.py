"""
Federated Learning for Credit Card Fraud Detection.

Applies FedAvg to fraud detection dataset, demonstrating real-world application.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from src.models import MLP
from src.data import load_fraud_data, partition_data, create_test_loader
from src.client import FederatedClientBinary
from src.server import FederatedServer
from src.metrics import ConvergenceTracker
from src.utils import set_seed, get_device


def run_fraud_experiment(
    data_path: str,
    num_clients: int = 5,
    num_rounds: int = 30,
    client_fraction: float = 0.8,
    local_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    distribution: str = 'non-iid',
    seed: int = 42,
    output_dir: str = './results/fraud'
):
    """
    Run federated learning experiment for fraud detection.

    Args:
        data_path: Path to credit card fraud CSV file
        num_clients: Number of federated clients (banks/institutions)
        num_rounds: Number of communication rounds
        client_fraction: Fraction of clients to select each round
        local_epochs: Number of local training epochs per client
        batch_size: Batch size for data loaders
        learning_rate: Learning rate for local optimization
        distribution: 'iid' or 'non-iid' data distribution
        seed: Random seed for reproducibility
        output_dir: Directory to save results
    """
    # Set random seeds
    set_seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load fraud detection data
    print(f"\nLoading fraud data from: {data_path}")
    train_dataset, test_dataset = load_fraud_data(data_path, train_ratio=0.8)
    test_loader = create_test_loader(test_dataset, batch_size=256)

    # Print dataset statistics
    train_labels = train_dataset.labels.numpy().flatten()
    test_labels = test_dataset.labels.numpy().flatten()

    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Train fraud cases: {train_labels.sum()} ({train_labels.mean()*100:.2f}%)")
    print(f"  Test fraud cases: {test_labels.sum()} ({test_labels.mean()*100:.2f}%)")

    # Partition data among clients
    print(f"\nPartitioning data among {num_clients} clients ({distribution} distribution)...")
    client_loaders, sample_counts = partition_data(
        train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        distribution=distribution,
        alpha=0.5,  # Dirichlet concentration for non-iid
        seed=seed
    )

    print(f"Client sample counts: {sample_counts}")
    print(f"Total samples: {sum(sample_counts)}")

    # Print fraud distribution per client
    print("\nFraud distribution per client:")
    for i, loader in enumerate(client_loaders):
        labels = []
        for _, target in loader:
            labels.extend(target.numpy().flatten())
        labels = np.array(labels)
        fraud_count = labels.sum()
        fraud_rate = labels.mean() * 100
        print(f"  Client {i}: {fraud_count} fraud cases ({fraud_rate:.2f}%)")

    # Create global model
    print("\nInitializing global model...")
    input_dim = train_dataset.features.shape[1]
    global_model = MLP(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        use_batch_norm=True
    )

    # Create clients
    print(f"Creating {num_clients} federated clients...")
    clients = []
    client_config = {
        'local_epochs': local_epochs,
        'learning_rate': learning_rate,
        'optimizer_type': 'adam',
        'weight_decay': 1e-5,
        'device': device
    }

    for client_id in range(num_clients):
        client = FederatedClientBinary(
            client_id=client_id,
            model=MLP(input_dim=input_dim, hidden_dim=64, num_layers=3, dropout=0.3),
            train_loader=client_loaders[client_id],
            config=client_config
        )
        clients.append(client)

    # Create server
    server_config = {
        'num_rounds': num_rounds,
        'client_fraction': client_fraction,
        'learning_rate': learning_rate
    }
    server = FederatedServer(global_model, server_config)

    # Initialize tracker
    tracker = ConvergenceTracker()

    # Initial evaluation
    print("\n" + "="*60)
    print("INITIAL EVALUATION")
    print("="*60)
    initial_metrics = evaluate_fraud_model(global_model, test_loader, device)
    print(f"Initial Test Loss: {initial_metrics['loss']:.4f}")
    print(f"Initial Test Accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"Initial Test F1: {initial_metrics['f1']:.4f}")
    print(f"Initial Test AUC-PR: {initial_metrics['auc_pr']:.4f}")

    tracker.update(-1, initial_metrics)

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING FEDERATED TRAINING: {num_rounds} rounds")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Clients per round: {max(int(client_fraction * num_clients), 1)} / {num_clients}")
    print(f"  - Local epochs: {local_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Distribution: {distribution}")
    print(f"{'='*60}\n")

    for round_num in range(num_rounds):
        # Select clients
        selected_clients = server.select_clients(clients, fraction=client_fraction)

        # Run federated round (without test evaluation every round for speed)
        round_metrics = server.federated_round(
            clients=selected_clients,
            round_num=round_num,
            test_loader=None,  # Evaluate separately
            verbose=False
        )

        # Evaluate on test set
        test_metrics = evaluate_fraud_model(global_model, test_loader, device)

        # Track metrics
        tracker.update(
            round_num=round_num,
            metrics={
                'train_loss': round_metrics['train_loss'],
                'train_accuracy': round_metrics['train_accuracy'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy']
            },
            num_clients=len(selected_clients),
            total_samples=sum(clients[c.client_id].num_samples for c in selected_clients)
        )

        # Print progress
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num + 1:3d}/{num_rounds} | "
                  f"Train Loss: {round_metrics['train_loss']:.4f} | "
                  f"Test Loss: {test_metrics['loss']:.4f} | "
                  f"Test F1: {test_metrics['f1']:.4f} | "
                  f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_metrics = evaluate_fraud_model(global_model, test_loader, device, verbose=True)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Final Test Loss:      {final_metrics['loss']:.4f}")
    print(f"Final Test Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Final Test Precision: {final_metrics['precision']:.4f}")
    print(f"Final Test Recall:    {final_metrics['recall']:.4f}")
    print(f"Final Test F1:        {final_metrics['f1']:.4f}")
    print(f"Final Test AUC-PR:    {final_metrics['auc_pr']:.4f}")

    # Save results
    print("\nSaving results...")
    convergence_plot = str(output_path / f"fraud_convergence_{distribution}.png")
    tracker.plot_convergence(convergence_plot, show_plot=False)
    print(f"Saved convergence plot to: {convergence_plot}")

    # Save metrics
    metrics_file = str(output_path / f"fraud_metrics_{distribution}.json")
    tracker.save_metrics(metrics_file)
    print(f"Saved metrics to: {metrics_file}")

    # Save model
    model_file = str(output_path / f"fraud_model_{distribution}.pt")
    server.save_global_model(model_file)
    print(f"Saved model to: {model_file}")

    # Save configuration
    config = {
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'client_fraction': client_fraction,
        'local_epochs': local_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'distribution': distribution,
        'seed': seed,
        'input_dim': input_dim,
        'final_metrics': {
            'loss': float(final_metrics['loss']),
            'accuracy': float(final_metrics['accuracy']),
            'f1': float(final_metrics['f1']),
            'auc_pr': float(final_metrics['auc_pr'])
        }
    }

    import json
    config_file = str(output_path / f"fraud_config_{distribution}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to: {config_file}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    return tracker, server


def evaluate_fraud_model(
    model,
    test_loader,
    device,
    verbose: bool = False
) -> dict:
    """
    Evaluate fraud detection model with multiple metrics.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        verbose: Whether to print detailed metrics

    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Binary predictions at 0.5 threshold
    binary_preds = (all_preds >= 0.5).astype(int)

    # Calculate metrics
    accuracy = (binary_preds == all_targets).mean()

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        confusion_matrix
    )

    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    auc_pr = average_precision_score(all_targets, all_preds)

    # Binary cross-entropy loss
    bce_loss = -np.mean(
        all_targets * np.log(all_preds + 1e-10) +
        (1 - all_targets) * np.log(1 - all_preds + 1e-10)
    )

    metrics = {
        'loss': bce_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': auc_pr,
        'num_samples': len(all_targets)
    }

    if verbose:
        cm = confusion_matrix(all_targets, binary_preds)
        print(f"Test Loss:      {metrics['loss']:.4f}")
        print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall:    {metrics['recall']:.4f}")
        print(f"Test F1:        {metrics['f1']:.4f}")
        print(f"Test AUC-PR:    {metrics['auc_pr']:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

    return metrics


if __name__ == "__main__":
    print("="*60)
    print("FEDERATED FRAUD DETECTION")
    print("="*60)

    # Check if data file exists
    # You can download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
    data_path = "./data/creditcard.csv"

    if not Path(data_path).exists():
        print("\nError: Credit card fraud dataset not found!")
        print(f"Expected path: {data_path}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("\nOr provide a custom path:")
        print("python experiments/fraud_detection.py --data_path /path/to/creditcard.csv")
        sys.exit(1)

    # Run fraud detection experiment
    tracker, server = run_fraud_experiment(
        data_path=data_path,
        num_clients=5,
        num_rounds=30,
        client_fraction=0.8,
        local_epochs=10,
        batch_size=64,
        learning_rate=0.001,
        distribution='non-iid',
        seed=42,
        output_dir='./results/fraud'
    )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
