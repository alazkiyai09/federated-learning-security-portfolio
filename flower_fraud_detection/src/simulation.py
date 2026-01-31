"""
Flower Simulation Runner

Orchestrates federated learning simulation using Flower's
start_simulation API for local development and testing.
"""

import os
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from src.client import create_client
from src.data import prepare_federated_data
from src.model import FraudDetectionModel
from src.server import get_initial_parameters, get_strategy
from src.utils import TensorBoardLogger, set_seed


def client_fn(
    cid: str,
    partitioned_data: Dict[int, Any],
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    device: str = "cpu",
) -> Any:
    """
    Client factory function for Flower simulation.

    Args:
        cid: Client ID (as string)
        partitioned_data: Dictionary of client data loaders
        model_cfg: Model configuration
        train_cfg: Training configuration
        device: Device to use

    Returns:
        Flower Client instance
    """
    # Get client data
    client_id = int(cid)
    train_loader, test_loader = partitioned_data[client_id]

    # Create model instance
    model = FraudDetectionModel(
        input_dim=model_cfg.input_dim,
        hidden_dims=model_cfg.hidden_dims,
    )

    # Merge configurations
    config = OmegaConf.merge(model_cfg, train_cfg)

    # Create and return client
    return create_client(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )


def main(cfg: DictConfig) -> None:
    """
    Main simulation entry point.

    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)

    # Set device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data
    print("Preparing federated data...")
    partitioned_data, input_dim = prepare_federated_data(cfg)

    # Update input_dim in config
    cfg.input_dim = input_dim

    print(f"Data prepared for {len(partitioned_data)} clients")
    for client_id, (train_loader, _) in partitioned_data.items():
        print(f"  Client {client_id}: {len(train_loader.dataset)} training samples")

    # Create strategy
    print(f"Creating strategy: {cfg.strategy.strategy_name}")
    logger = TensorBoardLogger(
        log_dir=cfg.log_dir,
        experiment_name=f"{cfg.experiment_name}_{cfg.strategy.strategy_name}_{cfg.data.partition_type}",
    )

    strategy = get_strategy(
        strategy_name=cfg.strategy.strategy_name,
        config=cfg,
        logger=logger,
    )

    # Set initial parameters if needed
    if strategy.initial_parameters is None:
        print("Initializing model parameters...")
        initial_params = get_initial_parameters(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            device=device,
        )
        strategy.initial_parameters = ndarrays_to_parameters(initial_params)

    # Create client wrapper for simulation
    def simulation_client_fn(cid: str) -> Any:
        return client_fn(
            cid=cid,
            partitioned_data=partitioned_data,
            model_cfg=cfg,
            train_cfg=cfg,
            device=device,
        )

    # Run simulation
    print(f"\nStarting simulation for {cfg.num_rounds} rounds...")
    print(f"Strategy: {cfg.strategy.strategy_name}")
    print(f"Data partition: {cfg.data.partition_type}")
    print(f"Number of clients: {cfg.num_clients}")
    print("-" * 50)

    history = start_simulation(
        client_fn=simulation_client_fn,
        num_clients=cfg.num_clients,
        config=ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0 if device == "cpu" else 1},
        ray_init_args={"num_cpus": 4, "num_gpus": int(device == "cuda")},
    )

    # Close logger
    logger.close()

    # Print final results
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)

    if history.losses_distributed:
        print("\nTraining Losses (Round, Value):")
        for round_num, loss in history.losses_distributed:
            print(f"  Round {round_num}: {loss:.4f}")

    if history.metrics_distributed_fit:
        print("\nTraining Metrics (Round, Metric):")
        for metric_name in history.metrics_distributed_fit.keys():
            print(f"  {metric_name}:")
            for round_num, value in history.metrics_distributed_fit[metric_name][-5:]:
                print(f"    Round {round_num}: {value:.4f}")

    if history.metrics_distributed:
        print("\nEvaluation Metrics (Round, Metric):")
        for metric_name in history.metrics_distributed.keys():
            print(f"  {metric_name}:")
            for round_num, value in history.metrics_distributed[metric_name][-5:]:
                print(f"    Round {round_num}: {value:.4f}")

    print(f"\nResults logged to: {logger.log_path}")
    print(f"View with: tensorboard --logdir {cfg.log_dir}")


if __name__ == "__main__":
    # This is for direct execution (not recommended, use Hydra)
    from hydra import compose, initialize
    from omegaconf import DictConfig

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base")
    main(cfg)
