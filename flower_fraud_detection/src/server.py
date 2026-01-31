"""
Flower Server Setup

Provides strategy factory and server start functions.
"""

from typing import Optional

import torch
from flwr.server import ServerConfig, ServerDriver, start_server
from flwr.server.strategy import Strategy
from omegaconf import DictConfig

from src.model import FraudDetectionModel
from src.strategy import FedAdamCustom, FedAvgCustom, FedProxCustom
from src.utils import TensorBoardLogger


def get_strategy(
    strategy_name: str,
    config: DictConfig,
    logger: Optional[TensorBoardLogger] = None,
) -> Strategy:
    """
    Create a strategy instance based on configuration.

    Args:
        strategy_name: Name of the strategy (fedavg, fedprox, fedadam)
        config: Hydra configuration object
        logger: Optional TensorBoard logger

    Returns:
        Flower Strategy instance
    """
    strategy_config = config.strategy

    # Common parameters
    common_params = {
        "fraction_fit": config.fraction_fit,
        "fraction_evaluate": config.fraction_evaluate,
        "min_fit_clients": config.min_fit_clients,
        "min_evaluate_clients": config.min_evaluate_clients,
        "min_available_clients": config.min_available_clients,
        "accept_failures": strategy_config.get("accept_failures", True),
        "fit_metrics_aggregation": strategy_config.get("fit_metrics_aggregation", None),
        "evaluate_metrics_aggregation": strategy_config.get("evaluate_metrics_aggregation", None),
    }

    if strategy_name == "fedavg":
        strategy = FedAvgCustom(**common_params)

    elif strategy_name == "fedprox":
        proximal_mu = strategy_config.get("proximal_mu", 0.01)
        strategy = FedProxCustom(proximal_mu=proximal_mu, **common_params)

    elif strategy_name == "fedadam":
        tau = strategy_config.get("tau", 0.9)
        eta = strategy_config.get("eta", 0.01)
        strategy = FedAdamCustom(tau=tau, eta=eta, **common_params)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategy


def start_server_with_strategy(
    strategy: Strategy,
    num_rounds: int,
    server_address: str = "[::]:8080",
    logger: Optional[TensorBoardLogger] = None,
) -> None:
    """
    Start Flower server with given strategy.

    Args:
        strategy: Flower strategy instance
        num_rounds: Number of federated rounds
        server_address: Server address (default: [::]:8080)
        logger: Optional TensorBoard logger
    """
    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    # Start server
    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


def get_initial_parameters(
    input_dim: int,
    hidden_dims: list,
    device: str = "cpu",
) -> list:
    """
    Get initial model parameters for server.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes
        device: Device to create model on

    Returns:
        List of numpy arrays with initial model parameters
    """
    # Create model
    model = FraudDetectionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    # Get parameters as numpy arrays
    parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    return parameters
