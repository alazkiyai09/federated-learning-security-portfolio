#!/usr/bin/env python3
"""
Main Entry Point for Flower Fraud Detection Experiments

Uses Hydra for configuration management.
Run with: python main.py [overrides]
"""

import os
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.simulation import main as run_simulation


def main() -> None:
    """Main entry point for federated learning experiments."""
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="base")

    # Print configuration
    print("\n" + "=" * 60)
    print("FLOWER FRAUD DETECTION - FEDERATED LEARNING")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Strategy: {cfg.strategy.strategy_name}")
    print(f"  Data Partition: {cfg.data.partition_type}")
    print(f"  Number of Clients: {cfg.num_clients}")
    print(f"  Number of Rounds: {cfg.num_rounds}")
    print(f"  Local Epochs: {cfg.local_epochs}")
    print(f"  Learning Rate: {cfg.learning_rate}")
    print(f"  Device: {cfg.device}")

    # Print strategy-specific config
    if cfg.strategy.strategy_name == "fedprox":
        print(f"  Proximal Mu: {cfg.strategy.proximal_mu}")
    elif cfg.strategy.strategy_name == "fedadam":
        print(f"  Tau: {cfg.strategy.tau}")
        print(f"  Eta: {cfg.strategy.eta}")

    print(f"  Seed: {cfg.seed}")
    print("=" * 60 + "\n")

    # Run simulation
    try:
        run_simulation(cfg)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        raise


if __name__ == "__main__":
    main()
