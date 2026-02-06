"""
Flower client implementation for federated fraud detection.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

from ..models.fraud_nn import create_model
from ..models.training_utils import Trainer


class FraudClient(fl.client.NumPyClient):
    """
    Flower client for fraud detection model training.

    Each bank runs one client instance with its local data.
    """

    def __init__(
        self,
        bank_id: str,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        training_config: Dict
    ):
        """
        Initialize Flower client.

        Args:
            bank_id: Bank identifier
            model: PyTorch model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            training_config: Training hyperparameters
        """
        self.bank_id = bank_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.training_config = training_config

        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            learning_rate=training_config.get('learning_rate', 0.001),
            verbose=False  # Suppress verbose output in FL
        )

        # Track training history
        self.history = {
            'loss': [],
            'auc': [],
            'f1': []
        }

        # Track round number
        self.current_round = 0

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Configuration from server

        Returns:
            List of numpy arrays containing model parameters
        """
        # Get model state dict
        state_dict = self.model.state_dict()

        # Convert to list of numpy arrays
        params = [param.cpu().numpy() for param in state_dict.values()]

        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of numpy arrays from server
        """
        # Get current state dict
        state_dict = self.model.state_dict()

        # Convert numpy arrays to torch tensors
        params_dict = zip(state_dict.keys(), parameters)
        state_dict_update = {
            k: torch.from_numpy(v).to(self.model.device)
            for k, v in params_dict
        }

        # Update model
        self.model.load_state_dict(state_dict_update, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.

        Args:
            parameters: Initial model parameters from server
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, n_samples, metrics)
        """
        # Set model parameters from server
        self.set_parameters(parameters)

        # Get current round
        self.current_round = config.get('rnd', 0)
        local_epochs = config.get('local_epochs', self.training_config.get('local_epochs', 3))

        # Train locally
        history = self.trainer.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            n_epochs=local_epochs,
            batch_size=self.training_config.get('batch_size', 256),
            early_stopping_patience=100,  # Disable early stopping in FL
            verbose=False
        )

        # Get updated parameters
        updated_params = self.get_parameters(config={})

        # Calculate metrics on validation set
        val_predictions = self.trainer.predict(self.X_val)
        val_metrics = compute_validation_metrics(self.y_val, val_predictions)

        # Store history
        self.history['loss'].append(history['train_loss'][-1])
        self.history['auc'].append(val_metrics['auc_roc'])
        self.history['f1'].append(val_metrics['f1'])

        # Return metrics to server
        metrics = {
            'bank_id': self.bank_id,
            'auc_roc': val_metrics['auc_roc'],
            'f1': val_metrics['f1'],
            'loss': history['train_loss'][-1],
            'n_train_samples': len(self.X_train),
            'n_val_samples': len(self.X_val)
        }

        return updated_params, len(self.X_train), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, n_samples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate on validation set (using as test)
        val_predictions = self.trainer.predict(self.X_val)
        val_metrics = compute_validation_metrics(self.y_val, val_predictions)

        # Note: In FL, evaluate() is called with validation data
        # True testing happens after FL completes

        return 0.0, len(self.X_val), val_metrics


def compute_validation_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for validation.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Metrics dictionary
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except (ValueError, IndexError) as e:
        # Handle edge cases like single class in y_true
        auc = 0.0

    # F1 score
    y_pred = (y_pred_proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {
        'auc_roc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def client_fn(
    client_id: str,
    bank_data: Dict[str, Dict],
    model_config: Dict,
    training_config: Dict,
    device: str = "cpu"
) -> FraudClient:
    """
    Client function for Flower simulation.

    Args:
        client_id: Client identifier
        bank_data: Dictionary with bank data
        model_config: Model configuration
        training_config: Training configuration
        device: Device to use

    Returns:
        FraudClient instance
    """
    # Extract bank_id from client_id
    bank_id = client_id.replace("client_", "")

    # Get bank data
    bank_info = bank_data[bank_id]

    # Create model
    input_dim = bank_info['input_dim']
    model = create_model(
        input_dim=input_dim,
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.3)
    )

    # Move to device
    model = model.to(device)

    # Create client
    client = FraudClient(
        bank_id=bank_id,
        model=model,
        X_train=bank_info['X_train'],
        y_train=bank_info['y_train'],
        X_val=bank_info['X_val'],
        y_val=bank_info['y_val'],
        training_config=training_config
    )

    return client
