"""
Reproducibility Utilities for Personalized FL

Provides:
1. Random seed management
2. Model checkpointing
3. Experiment configuration tracking
4. Deterministic execution
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import pickle
import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For deterministic CUDA operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CheckpointManager:
    """
    Manage model checkpoints for FL experiments.

    Handles saving/loading of:
    - Global model parameters
    - Client-specific personalized models
    - Training state
    - Experiment configuration
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        max_to_keep: int = 5
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints
            experiment_name: Name of experiment (subdirectory)
            max_to_keep: Maximum number of checkpoints to keep per client
        """
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.max_to_keep = max_to_keep

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir = self.checkpoint_dir / "global"
        self.clients_dir = self.checkpoint_dir / "clients"

        self.global_dir.mkdir(exist_ok=True)
        self.clients_dir.mkdir(exist_ok=True)

    def save_global_model(
        self,
        model: torch.nn.Module,
        round_num: int,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save global model checkpoint.

        Args:
            model: Global model
            round_num: Current round number
            metrics: Optional metrics to save with checkpoint
            config: Optional experiment configuration

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.global_dir / f"global_round_{round_num}.pt"

        checkpoint = {
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }

        if config:
            checkpoint['config'] = config

        torch.save(checkpoint, checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(self.global_dir, "global_round_")

        return str(checkpoint_path)

    def load_global_model(
        self,
        round_num: int,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Load global model checkpoint.

        Args:
            round_num: Round number to load
            model: Model to load parameters into

        Returns:
            Checkpoint dictionary with metrics and config
        """
        checkpoint_path = self.global_dir / f"global_round_{round_num}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def save_client_model(
        self,
        model: torch.nn.Module,
        client_id: int,
        round_num: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save client-specific personalized model.

        Args:
            model: Client model (personalized)
            client_id: Client identifier
            round_num: Current round number
            metrics: Optional client metrics

        Returns:
            Path to saved checkpoint
        """
        client_dir = self.clients_dir / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)

        checkpoint_path = client_dir / f"client_{client_id}_round_{round_num}.pt"

        checkpoint = {
            'client_id': client_id,
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)

        # Clean up old client checkpoints
        self._cleanup_old_checkpoints(client_dir, f"client_{client_id}_round_")

        return str(checkpoint_path)

    def load_client_model(
        self,
        client_id: int,
        round_num: int,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Load client-specific checkpoint.

        Args:
            client_id: Client identifier
            round_num: Round number to load
            model: Model to load parameters into

        Returns:
            Checkpoint dictionary
        """
        client_dir = self.clients_dir / f"client_{client_id}"
        checkpoint_path = client_dir / f"client_{client_id}_round_{round_num}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Client checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def save_experiment_config(
        self,
        config: Dict[str, Any]
    ) -> str:
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Path to saved config
        """
        config_path = self.checkpoint_dir / "experiment_config.json"

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        return str(config_path)

    def load_experiment_config(self) -> Dict[str, Any]:
        """
        Load experiment configuration.

        Returns:
            Configuration dictionary
        """
        config_path = self.checkpoint_dir / "experiment_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def list_available_rounds(self) -> list:
        """
        List available global model checkpoint rounds.

        Returns:
            Sorted list of round numbers
        """
        rounds = []
        for file in self.global_dir.glob("global_round_*.pt"):
            round_str = file.stem.replace("global_round_", "")
            try:
                rounds.append(int(round_str))
            except ValueError:
                continue

        return sorted(rounds)

    def _cleanup_old_checkpoints(
        self,
        directory: Path,
        prefix: str
    ) -> None:
        """
        Remove old checkpoints exceeding max_to_keep.

        Args:
            directory: Directory containing checkpoints
            prefix: File prefix (e.g., 'global_round_')
        """
        checkpoints = sorted(
            directory.glob(f"{prefix}*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove excess checkpoints
        for checkpoint in checkpoints[self.max_to_keep:]:
            checkpoint.unlink()


class ExperimentTracker:
    """
    Track experiment metrics and metadata for reproducibility.

    Records:
    - Hyperparameters
    - Per-round metrics
    - Per-client metrics
    - Compute budget usage
    """

    def __init__(
        self,
        results_dir: str,
        experiment_name: str
    ):
        """
        Initialize experiment tracker.

        Args:
            results_dir: Base results directory
            experiment_name: Experiment name
        """
        self.results_dir = Path(results_dir) / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'config': {},
            'rounds': [],
            'final_metrics': {}
        }

    def log_config(
        self,
        config: Dict[str, Any]
    ) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.experiment_data['config'] = config

    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, float],
        per_client_metrics: Optional[Dict[int, Dict[str, float]]] = None
    ) -> None:
        """
        Log metrics for a single round.

        Args:
            round_num: Round number
            metrics: Aggregated metrics
            per_client_metrics: Optional per-client metrics
        """
        round_data = {
            'round': round_num,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if per_client_metrics:
            # Convert int keys to strings for JSON serialization
            round_data['per_client_metrics'] = {
                str(k): v for k, v in per_client_metrics.items()
            }

        self.experiment_data['rounds'].append(round_data)

    def log_final_metrics(
        self,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log final experiment metrics.

        Args:
            metrics: Final metrics dictionary
        """
        self.experiment_data['final_metrics'] = metrics
        self.experiment_data['end_time'] = datetime.now().isoformat()

    def save_results(self) -> str:
        """
        Save experiment results to JSON.

        Returns:
            Path to saved results
        """
        results_path = self.results_dir / "experiment_results.json"

        with open(results_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)

        return str(results_path)

    def save_per_client_metrics(
        self,
        metrics: Dict[int, Dict[str, float]],
        suffix: str = ""
    ) -> str:
        """
        Save per-client metrics to separate JSON file.

        Args:
            metrics: Per-client metrics dictionary
            suffix: Optional filename suffix

        Returns:
            Path to saved metrics
        """
        filename = f"per_client_metrics{suffix}.json"
        metrics_path = self.results_dir / filename

        # Convert int keys to strings
        serializable_metrics = {
            str(k): v for k, v in metrics.items()
        }

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)

        return str(metrics_path)


def save_model_parameters(
    model: torch.nn.Module,
    output_path: str
) -> None:
    """
    Save model parameters to disk.

    Args:
        model: PyTorch model
        output_path: Path to save parameters
    """
    state_dict = model.state_dict()

    # Convert to numpy for serialization
    numpy_params = {
        k: v.cpu().numpy() for k, v in state_dict.items()
    }

    with open(output_path, 'wb') as f:
        pickle.dump(numpy_params, f)


def load_model_parameters(
    model: torch.nn.Module,
    input_path: str
) -> None:
    """
    Load model parameters from disk.

    Args:
        model: PyTorch model to load into
        input_path: Path to load parameters from
    """
    with open(input_path, 'rb') as f:
        numpy_params = pickle.load(f)

    # Convert back to torch tensors
    state_dict = {
        k: torch.from_numpy(v)
        for k, v in numpy_params.items()
    }

    model.load_state_dict(state_dict, strict=True)
