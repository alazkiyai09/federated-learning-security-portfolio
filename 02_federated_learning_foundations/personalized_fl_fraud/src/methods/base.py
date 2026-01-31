"""
Base Class for Personalization Methods

Defines the abstract interface that all personalization methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ..models.base import FraudDetectionModel
from ..models.utils import (
    get_parameters_by_layer_type,
    set_parameters_by_layer_type,
    count_parameters
)


@dataclass
class PersonalizationResult:
    """
    Result of personalization training.

    Attributes:
        global_metrics: Metrics of global model on client data
        personalized_metrics: Metrics of personalized model on client data
        personalization_delta: Delta (personalized - global) for each metric
        training_time: Time taken for personalization (seconds)
        flops: FLOPs consumed during personalization
        communication_cost: Bytes transferred (if applicable)
    """
    global_metrics: Dict[str, float]
    personalized_metrics: Dict[str, float]
    personalization_delta: Dict[str, float]
    training_time: float
    flops: int
    communication_cost: int

    def get_summary(self) -> Dict[str, Any]:
        """Get summary dictionary for logging."""
        return {
            'global_metrics': self.global_metrics,
            'personalized_metrics': self.personalized_metrics,
            'personalization_delta': self.personalization_delta,
            'training_time': self.training_time,
            'flops': self.flops,
            'communication_cost': self.communication_cost
        }


class PersonalizationMethod(ABC):
    """
    Abstract base class for personalization methods.

    All personalization methods must implement:
    1. get_client_strategy() - Return client with method-specific training logic
    2. get_server_strategy() - Return server aggregation strategy
    3. compute_personalization_benefit() - Calculate benefit over global model

    The base class provides common utilities for parameter manipulation,
    metrics computation, and compute tracking.
    """

    def __init__(
        self,
        name: str,
        config: DictConfig,
        random_state: int = 42
    ):
        """
        Initialize personalization method.

        Args:
            name: Method name (e.g., "Local Fine-Tuning", "FedPer")
            config: Method configuration from methods.yaml
            random_state: Random seed
        """
        self.name = name
        self.config = config
        self.random_state = random_state

        # Extract method-specific config
        self.method_config = config.get(self._get_config_key(), {})

    @abstractmethod
    def _get_config_key(self) -> str:
        """
        Get the config key for this method in methods.yaml.

        Returns:
            Config key (e.g., 'local_finetuning', 'fedper')
        """
        pass

    @abstractmethod
    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """
        Return client instance for this personalization method.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test/validation data loader
            device: Device to use

        Returns:
            Client instance (must implement fit() and evaluate() methods)
        """
        pass

    @abstractmethod
    def get_server_strategy(
        self,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int
    ):
        """
        Return server strategy for this personalization method.

        Args:
            fraction_fit: Fraction of clients to sample each round
            min_fit_clients: Minimum clients for training
            min_available_clients: Minimum available clients

        Returns:
            Server strategy instance
        """
        pass

    def compute_personalization_benefit(
        self,
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute personalization benefit (delta from global).

        Args:
            global_metrics: Metrics from global model
            personalized_metrics: Metrics from personalized model

        Returns:
            Dictionary of metric deltas (personalized - global)
        """
        delta = {}

        for metric_name in global_metrics.keys():
            if metric_name in personalized_metrics:
                delta[metric_name] = (
                    personalized_metrics[metric_name] - global_metrics[metric_name]
                )

        return delta

    def get_num_personalizable_params(
        self,
        model: nn.Module
    ) -> int:
        """
        Count number of personalized parameters.

        Args:
            model: PyTorch model

        Returns:
            Number of personalized parameters
        """
        return count_parameters(model, count_only_trainable=True)

    def get_num_shared_params(
        self,
        model: nn.Module
    ) -> int:
        """
        Count number of shared (global) parameters.

        For methods without personalization layers, this equals total params.

        Args:
            model: PyTorch model

        Returns:
            Number of shared parameters
        """
        return count_parameters(model, count_only_trainable=True)

    def is_enabled(self) -> bool:
        """Check if this method is enabled in config."""
        return self.method_config.get('enabled', True)

    def get_description(self) -> str:
        """Get method description from config."""
        return self.method_config.get('description', '')

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class FedAvgBaseline(PersonalizationMethod):
    """
    Standard FedAvg baseline (no personalization).

    Serves as the baseline for comparison against personalization methods.
    """

    def _get_config_key(self) -> str:
        return "fedavg"

    def get_client_strategy(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu"
    ):
        """Return standard FedAvg client."""
        from ..clients.wrappers import create_fedavg_client
        return create_fedavg_client(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=self.config,
            device=device
        )

    def get_server_strategy(
        self,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int
    ):
        """Return FedAvg server strategy."""
        from flwr.server.strategy import FedAvg
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def compute_personalization_benefit(
        self,
        global_metrics: Dict[str, float],
        personalized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        For FedAvg, personalization benefit is zero (no personalization).
        """
        return {k: 0.0 for k in global_metrics.keys()}
