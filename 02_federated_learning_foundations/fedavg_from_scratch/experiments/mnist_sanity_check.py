"""
MNIST Sanity Check for FedAvg Implementation.

Runs federated learning on MNIST to verify correct implementation.
This serves as a baseline before applying to fraud detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from src.models import SimpleCNN
from src.data import load_mnist, partition_data, create_test_loader
from src.client import FederatedClient
from src.server import FederatedServer
from src.metrics import ConvergenceTracker
from src.utils import set_seed, get_device


def run_mnist_experiment(
    num_clients: int = 10,
    num_rounds: int = 20,
    client_fraction: float = 0.5,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    distribution: str = 'iid',
    seed: int = 42,
    output_dir: str = './results/mnist'
):
    """
    Run federated learning experiment on MNIST.

    Args:
        num_clients: Number of federated clients
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

    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_dataset, test_dataset = load_mnist(data_dir=str(output_path / 'data'))
    test_loader = create_test_loader(test_dataset, batch_size=100)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Partition data among clients
    print(f"\nPartitioning data among {num_clients} clients ({distribution} distribution)...")
    client_loaders, sample_counts = partition_data(
        train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        distribution=distribution,
        seed=seed
    )

    print(f"Client sample counts: {sample_counts}")
    print(f"Total samples: {sum(sample_counts)}")

    # Create global model
    print("\nInitializing global model...")
    global_model = SimpleCNN(num_classes=10)

    # Create clients
    print(f"Creating {num_clients} federated clients...")
    clients = []
    client_config = {
        'local_epochs': local_epochs,
        'learning_rate': learning_rate,
        'optimizer_type': 'sgd',
        'momentum': 0.9,
        'device': device
    }

    for client_id in range(num_clients):
        client = FederatedClient(
            client_id=client_id,
            model=SimpleCNN(num_classes=10),
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
    initial_metrics = server.evaluate(test_loader, verbose=True)
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

        # Run federated round
        round_metrics = server.federated_round(
            clients=selected_clients,
            round_num=round_num,
            test_loader=test_loader,
            verbose=False
        )

        # Track metrics
        tracker.update(
            round_num=round_num,
            metrics={
                'train_loss': round_metrics['train_loss'],
                'train_accuracy': round_metrics['train_accuracy'],
                'test_loss': round_metrics['test_loss'],
                'test_accuracy': round_metrics['test_accuracy']
            },
            num_clients=len(selected_clients),
            total_samples=sum(c.num_samples for c in selected_clients)
        )

        # Print progress every 5 rounds
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num + 1:3d}/{num_rounds} | "
                  f"Train Acc: {round_metrics['train_accuracy']:.4f} | "
                  f"Test Acc: {round_metrics['test_accuracy']:.4f} | "
                  f"Test Loss: {round_metrics['test_loss']:.4f}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_metrics = server.evaluate(test_loader, verbose=True)

    # Print summary
    tracker.print_summary()

    # Save results
    print("\nSaving results...")
    convergence_plot = str(output_path / f"mnist_convergence_{distribution}.png")
    tracker.plot_convergence(convergence_plot, show_plot=False)
    print(f"Saved convergence plot to: {convergence_plot}")

    # Save metrics
    metrics_file = str(output_path / f"mnist_metrics_{distribution}.json")
    tracker.save_metrics(metrics_file)
    print(f"Saved metrics to: {metrics_file}")

    # Save model
    model_file = str(output_path / f"mnist_model_{distribution}.pt")
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
        'final_test_accuracy': final_metrics['accuracy'],
        'final_test_loss': final_metrics['loss']
    }

    import json
    config_file = str(output_path / f"mnist_config_{distribution}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to: {config_file}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    return tracker, server


if __name__ == "__main__":
    # Run MNIST sanity check
    print("="*60)
    print("FEDAVG MNIST SANITY CHECK")
    print("="*60)

    # Run IID experiment first
    print("\n[1/2] Running IID experiment...")
    tracker_iid, server_iid = run_mnist_experiment(
        num_clients=10,
        num_rounds=20,
        client_fraction=0.5,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.01,
        distribution='iid',
        seed=42,
        output_dir='./results/mnist'
    )

    # Run Non-IID experiment
    print("\n\n[2/2] Running Non-IID experiment...")
    tracker_non_iid, server_non_iid = run_mnist_experiment(
        num_clients=10,
        num_rounds=20,
        client_fraction=0.5,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.01,
        distribution='non-iid',
        alpha=0.5,
        seed=42,
        output_dir='./results/mnist'
    )

    # Compare IID vs Non-IID
    print("\nGenerating comparison plot...")
    tracker_iid.compare_trackers(
        tracker_non_iid,
        save_path='./results/mnist/mnist_comparison.png',
        label1='IID',
        label2='Non-IID'
    )
    print("Saved comparison plot to: ./results/mnist/mnist_comparison.png")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
