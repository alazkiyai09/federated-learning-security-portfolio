"""
Vertical Federated Learning Trainer.

Orchestrates training of split neural network across multiple parties
and a server in Vertical Federated Learning.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field

from ..models.bottom_model import PartyABottomModel, PartyBBottomModel
from ..models.top_model import TopModel
from ..models.split_nn import SplitNN
from .forward_pass import secure_forward, compute_loss
from .backward_pass import secure_backward, analyze_gradient_leakage


@dataclass
class TrainingConfig:
    """Configuration for VFL training."""
    num_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    log_interval: int = 10
    save_checkpoints: bool = True
    checkpoint_dir: str = "results/checkpoints"
    analyze_gradient_leakage: bool = True
    leakage_check_interval: int = 5  # epochs


@dataclass
class TrainingHistory:
    """Training history tracking."""
    train_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    val_aucs: List[float] = field(default_factory=list)
    leakage_metrics: List[Dict] = field(default_factory=list)


class VerticalFLTrainer:
    """
    Trainer for Vertical Federated Learning.

    Manages training loop across Party A, Party B, and Server.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_config: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize VFL trainer.

        Args:
            config: Training configuration
            model_config: Model architecture configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        self.history = TrainingHistory()

        # Initialize models
        if model_config is None:
            model_config = self._get_default_model_config()

        self.bottom_model_a = PartyABottomModel(
            input_dim=model_config['party_a']['input_dim'],
            embedding_dim=model_config['party_a']['embedding_dim'],
            hidden_dims=model_config['party_a']['hidden_dims'],
            activation=model_config['party_a']['activation'],
            dropout=model_config['party_a']['dropout']
        )

        self.bottom_model_b = PartyBBottomModel(
            input_dim=model_config['party_b']['input_dim'],
            embedding_dim=model_config['party_b']['embedding_dim'],
            hidden_dims=model_config['party_b']['hidden_dims'],
            activation=model_config['party_b']['activation'],
            dropout=model_config['party_b']['dropout']
        )

        total_emb_dim = (
            model_config['party_a']['embedding_dim'] +
            model_config['party_b']['embedding_dim']
        )

        self.top_model = TopModel(
            embedding_dim_total=total_emb_dim,
            output_dim=model_config['server']['output_dim'],
            hidden_dims=model_config['server']['hidden_dims'],
            activation=model_config['server']['activation'],
            output_activation='None',  # Use logits for CrossEntropyLoss
            dropout=model_config['server']['dropout']
        )

        # Create SplitNN wrapper
        self.split_nn = SplitNN(
            self.bottom_model_a,
            self.bottom_model_b,
            self.top_model,
            device
        )

        # Optimizers (each party has their own optimizer)
        self.optimizer_a = optim.Adam(
            self.bottom_model_a.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.optimizer_b = optim.Adam(
            self.bottom_model_b.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.optimizer_server = optim.Adam(
            self.top_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _get_default_model_config(self) -> Dict:
        """Get default model configuration."""
        return {
            'party_a': {
                'input_dim': 7,
                'embedding_dim': 16,
                'hidden_dims': [32, 24],
                'activation': 'ReLU',
                'dropout': 0.2
            },
            'party_b': {
                'input_dim': 3,
                'embedding_dim': 8,
                'hidden_dims': [16, 12],
                'activation': 'ReLU',
                'dropout': 0.2
            },
            'server': {
                'output_dim': 2,
                'hidden_dims': [32, 16],
                'activation': 'ReLU',
                'dropout': 0.3
            }
        }

    def train_epoch(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        y: np.ndarray,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            X_a: Party A features (num_samples, feat_a_dim)
            X_b: Party B features (num_samples, feat_b_dim)
            y: Labels (num_samples,)
            epoch: Current epoch number

        Returns:
            Dictionary with epoch statistics
        """
        self.split_nn.train_mode()

        num_samples = len(y)
        num_batches = num_samples // self.config.batch_size

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_leakage = None

        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size

            batch_x_a = torch.FloatTensor(X_a[start_idx:end_idx]).to(self.device)
            batch_x_b = torch.FloatTensor(X_b[start_idx:end_idx]).to(self.device)
            batch_y = torch.LongTensor(y[start_idx:end_idx]).to(self.device)

            # Forward pass
            predictions, emb_a, emb_b = secure_forward(
                self.bottom_model_a,
                self.bottom_model_b,
                self.top_model,
                batch_x_a,
                batch_x_b
            )

            # Use logits (no softmax) for CrossEntropyLoss
            logits = self.top_model.forward_logits(
                torch.cat([emb_a, emb_b], dim=1)
            )
            loss = self.criterion(logits, batch_y)

            # Backward pass
            grad_stats = secure_backward(
                self.top_model,
                emb_a,
                emb_b,
                loss,
                self.bottom_model_a,
                self.bottom_model_b,
                batch_x_a,
                batch_x_b
            )

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.bottom_model_a.parameters(),
                    self.config.gradient_clip
                )
                torch.nn.utils.clip_grad_norm_(
                    self.bottom_model_b.parameters(),
                    self.config.gradient_clip
                )
                torch.nn.utils.clip_grad_norm_(
                    self.top_model.parameters(),
                    self.config.gradient_clip
                )

            # Update parameters
            self.optimizer_a.step()
            self.optimizer_b.step()
            self.optimizer_server.step()

            # Track statistics
            epoch_loss += loss.item()
            pred_labels = predictions.argmax(dim=1)
            epoch_correct += (pred_labels == batch_y).sum().item()

            # Analyze gradient leakage periodically
            if (self.config.analyze_gradient_leakage and
                batch_idx == 0 and
                epoch % self.config.leakage_check_interval == 0):
                leakage_a = analyze_gradient_leakage(emb_a, emb_a.grad)
                leakage_b = analyze_gradient_leakage(emb_b, emb_b.grad)
                epoch_leakage = {
                    'party_a': leakage_a,
                    'party_b': leakage_b,
                    'epoch': epoch
                }

            # Log progress
            if batch_idx % self.config.log_interval == 0:
                print(f"  Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        accuracy = epoch_correct / num_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'leakage': epoch_leakage
        }

    def validate(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            X_a: Party A features
            X_b: Party B features
            y: Labels

        Returns:
            Dictionary with validation metrics
        """
        self.split_nn.eval_mode()

        num_samples = len(y)
        num_batches = num_samples // self.config.batch_size + 1

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, num_samples)

                if start_idx >= num_samples:
                    break

                batch_x_a = torch.FloatTensor(X_a[start_idx:end_idx]).to(self.device)
                batch_x_b = torch.FloatTensor(X_b[start_idx:end_idx]).to(self.device)
                batch_y = torch.LongTensor(y[start_idx:end_idx]).to(self.device)

                # Forward pass
                predictions, emb_a, emb_b = secure_forward(
                    self.bottom_model_a,
                    self.bottom_model_b,
                    self.top_model,
                    batch_x_a,
                    batch_x_b
                )

                logits = self.top_model.forward_logits(
                    torch.cat([emb_a, emb_b], dim=1)
                )
                loss = self.criterion(logits, batch_y)

                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_labels.append(batch_y.cpu())

        # Compute metrics
        avg_loss = total_loss / num_batches
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        pred_labels = all_predictions.argmax(dim=1)
        accuracy = (pred_labels == all_labels).float().mean().item()

        # AUC-ROC
        from sklearn.metrics import roc_auc_score
        probs = all_predictions[:, 1]  # Probability of class 1
        auc = roc_auc_score(all_labels.numpy(), probs.numpy())

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }

    def train(
        self,
        X_a_train: np.ndarray,
        X_b_train: np.ndarray,
        y_train: np.ndarray,
        X_a_val: np.ndarray,
        X_b_val: np.ndarray,
        y_val: np.ndarray
    ) -> TrainingHistory:
        """
        Full training loop.

        Args:
            X_a_train: Training features for Party A
            X_b_train: Training features for Party B
            y_train: Training labels
            X_a_val: Validation features for Party A
            X_b_val: Validation features for Party B
            y_val: Validation labels

        Returns:
            TrainingHistory object
        """
        print(f"\n=== Starting Vertical FL Training ===")
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.device}\n")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")

            # Train
            train_stats = self.train_epoch(
                X_a_train, X_b_train, y_train, epoch
            )

            # Validate
            val_stats = self.validate(
                X_a_val, X_b_val, y_val
            )

            # Log history
            self.history.train_losses.append(train_stats['loss'])
            self.history.train_accuracies.append(train_stats['accuracy'])
            self.history.val_losses.append(val_stats['loss'])
            self.history.val_accuracies.append(val_stats['accuracy'])
            self.history.val_aucs.append(val_stats['auc'])

            if train_stats['leakage']:
                self.history.leakage_metrics.append(train_stats['leakage'])

            # Print summary
            print(f"\n  Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.4f}")
            print(f"  Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.4f}, Val AUC: {val_stats['auc']:.4f}")

            if train_stats['leakage']:
                print(f"\n  Gradient Leakage Analysis:")
                print(f"    Party A risk: {train_stats['leakage']['party_a']['leakage_risk_percent']:.1f}%")
                print(f"    Party B risk: {train_stats['leakage']['party_b']['leakage_risk_percent']:.1f}%")

            # Early stopping
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                self.patience_counter = 0

                # Save checkpoint
                if self.config.save_checkpoints:
                    self._save_checkpoint(epoch, val_stats)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print(f"\n=== Training Complete ===")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, epoch: int, val_stats: Dict) -> None:
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch+1}.pth'
        )

        torch.save({
            'epoch': epoch,
            'bottom_model_a_state_dict': self.bottom_model_a.state_dict(),
            'bottom_model_b_state_dict': self.bottom_model_b.state_dict(),
            'top_model_state_dict': self.top_model.state_dict(),
            'optimizer_a_state_dict': self.optimizer_a.state_dict(),
            'optimizer_b_state_dict': self.optimizer_b.state_dict(),
            'optimizer_server_state_dict': self.optimizer_server.state_dict(),
            'val_loss': val_stats['loss'],
            'val_accuracy': val_stats['accuracy'],
            'val_auc': val_stats['auc'],
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def save_results(self, save_dir: str) -> None:
        """Save training results."""
        os.makedirs(save_dir, exist_ok=True)

        # Save history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': self.history.train_losses,
                'train_accuracies': self.history.train_accuracies,
                'val_losses': self.history.val_losses,
                'val_accuracies': self.history.val_accuracies,
                'val_aucs': self.history.val_aucs,
                'leakage_metrics': self.history.leakage_metrics,
            }, f, indent=2)

        # Save models
        self.split_nn.save_models(save_dir)

        print(f"Results saved to: {save_dir}")
