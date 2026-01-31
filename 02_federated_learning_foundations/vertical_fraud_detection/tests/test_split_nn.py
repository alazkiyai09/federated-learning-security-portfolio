"""
Integration tests for Split Neural Network.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import tempfile

from models.bottom_model import PartyABottomModel, PartyBBottomModel
from models.top_model import TopModel
from models.split_nn import SplitNN
from training.vertical_fl_trainer import TrainingConfig, VerticalFLTrainer


def test_split_nn_initialization():
    """Test SplitNN initialization."""
    print("\n=== Testing SplitNN Initialization ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    split_nn = SplitNN(bottom_a, bottom_b, top)

    assert split_nn.bottom_model_a is bottom_a, "Bottom model A not set"
    assert split_nn.bottom_model_b is bottom_b, "Bottom model B not set"
    assert split_nn.top_model is top, "Top model not set"

    print("✓ SplitNN initialization successful")
    return True


def test_split_nn_forward_pass():
    """Test SplitNN forward pass."""
    print("\n=== Testing SplitNN Forward Pass ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    split_nn = SplitNN(bottom_a, bottom_b, top)

    batch_size = 16
    x_a = torch.randn(batch_size, 7)
    x_b = torch.randn(batch_size, 3)

    predictions, emb_a, emb_b = split_nn.forward_pass(x_a, x_b)

    assert predictions.shape == (batch_size, 2), f"Predictions shape incorrect: {predictions.shape}"
    assert emb_a.shape == (batch_size, 8), f"Embedding A shape incorrect: {emb_a.shape}"
    assert emb_b.shape == (batch_size, 4), f"Embedding B shape incorrect: {emb_b.shape}"

    # Check that predictions sum to 1 (softmax)
    pred_sums = predictions.sum(dim=1)
    assert torch.allclose(pred_sums, torch.ones(batch_size)), "Predictions don't sum to 1"

    print("✓ SplitNN forward pass working correctly")
    return True


def test_split_nn_backward_pass():
    """Test SplitNN backward pass."""
    print("\n=== Testing SplitNN Backward Pass ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2, output_activation='None')

    split_nn = SplitNN(bottom_a, bottom_b, top)

    batch_size = 16
    x_a = torch.randn(batch_size, 7)
    x_b = torch.randn(batch_size, 3)
    labels = torch.randint(0, 2, (batch_size,))

    # Forward pass
    predictions, emb_a, emb_b = split_nn.forward_pass(x_a, x_b)

    # Compute loss (use logits)
    logits = top.forward_logits(torch.cat([emb_a, emb_b], dim=1))
    loss = nn.CrossEntropyLoss()(logits, labels)

    # Backward pass
    stats = split_nn.backward_pass(loss, emb_a, emb_b, x_a, x_b)

    # Check gradients exist
    for name, param in bottom_a.named_parameters():
        assert param.grad is not None, f"Bottom A {name} has no gradient"

    for name, param in bottom_b.named_parameters():
        assert param.grad is not None, f"Bottom B {name} has no gradient"

    for name, param in top.named_parameters():
        assert param.grad is not None, f"Top {name} has no gradient"

    # Check stats
    assert 'loss' in stats, "Stats missing loss"
    assert 'grad_norm_a' in stats, "Stats missing grad_norm_a"
    assert 'grad_norm_b' in stats, "Stats missing grad_norm_b"
    assert 'grad_norm_top' in stats, "Stats missing grad_norm_top"

    print("✓ SplitNN backward pass working correctly")
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Grad norm A: {stats['grad_norm_a']:.4f}")
    print(f"  Grad norm B: {stats['grad_norm_b']:.4f}")
    print(f"  Grad norm top: {stats['grad_norm_top']:.4f}")

    return True


def test_split_nn_train_eval_modes():
    """Test train and eval mode switching."""
    print("\n=== Testing Train/Eval Mode Switching ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    split_nn = SplitNN(bottom_a, bottom_b, top)

    # Test train mode
    split_nn.train_mode()
    assert bottom_a.training, "Bottom A not in training mode"
    assert bottom_b.training, "Bottom B not in training mode"
    assert top.training, "Top not in training mode"

    # Test eval mode
    split_nn.eval_mode()
    assert not bottom_a.training, "Bottom A not in eval mode"
    assert not bottom_b.training, "Bottom B not in eval mode"
    assert not top.training, "Top not in eval mode"

    print("✓ Train/Eval mode switching working correctly")
    return True


def test_split_nn_predict():
    """Test prediction method."""
    print("\n=== Testing Predict Method ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    split_nn = SplitNN(bottom_a, bottom_b, top)

    batch_size = 16
    x_a = torch.randn(batch_size, 7)
    x_b = torch.randn(batch_size, 3)

    predictions = split_nn.predict(x_a, x_b)

    assert predictions.shape == (batch_size, 2), f"Predictions shape incorrect: {predictions.shape}"
    assert torch.allclose(predictions.sum(dim=1), torch.ones(batch_size)), "Predictions don't sum to 1"

    # Check model is in eval mode after predict
    assert not bottom_a.training, "Bottom A not in eval mode after predict"
    assert not bottom_b.training, "Bottom B not in eval mode after predict"
    assert not top.training, "Top not in eval mode after predict"

    print("✓ Predict method working correctly")
    return True


def test_split_nn_save_load():
    """Test saving and loading models."""
    print("\n=== Testing Save/Load ===")

    bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
    bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
    top = TopModel(embedding_dim_total=12, output_dim=2)

    split_nn = SplitNN(bottom_a, bottom_b, top)

    # Get initial parameters
    initial_params = {
        'bottom_a': list(bottom_a.parameters())[0].clone(),
        'bottom_b': list(bottom_b.parameters())[0].clone(),
        'top': list(top.parameters())[0].clone(),
    }

    # Save models
    with tempfile.TemporaryDirectory() as tmpdir:
        split_nn.save_models(tmpdir)

        # Create new models
        new_bottom_a = PartyABottomModel(input_dim=7, embedding_dim=8)
        new_bottom_b = PartyBBottomModel(input_dim=3, embedding_dim=4)
        new_top = TopModel(embedding_dim_total=12, output_dim=2)

        new_split_nn = SplitNN(new_bottom_a, new_bottom_b, new_top)

        # Load models
        new_split_nn.load_models(tmpdir)

        # Check parameters match
        assert torch.allclose(
            initial_params['bottom_a'],
            list(new_bottom_a.parameters())[0]
        ), "Bottom A parameters don't match after load"

        assert torch.allclose(
            initial_params['bottom_b'],
            list(new_bottom_b.parameters())[0]
        ), "Bottom B parameters don't match after load"

        assert torch.allclose(
            initial_params['top'],
            list(new_top.parameters())[0]
        ), "Top parameters don't match after load"

    print("✓ Save/Load working correctly")
    return True


def test_split_nn_integration_with_trainer():
    """Test SplitNN integration with VerticalFLTrainer."""
    print("\n=== Testing Integration with Trainer ===")

    # Create synthetic data
    np.random.seed(42)
    n_train, n_val = 200, 50

    X_a_train = np.random.randn(n_train, 7)
    X_b_train = np.random.randn(n_train, 3)
    y_train = np.random.randint(0, 2, n_train)

    X_a_val = np.random.randn(n_val, 7)
    X_b_val = np.random.randn(n_val, 3)
    y_val = np.random.randint(0, 2, n_val)

    # Create trainer with minimal config
    config = TrainingConfig(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10
    )

    trainer = VerticalFLTrainer(config=config, device='cpu')

    # Train for a few epochs
    history = trainer.train(
        X_a_train, X_b_train, y_train,
        X_a_val, X_b_val, y_val
    )

    # Check that training occurred
    assert len(history.train_losses) > 0, "No training losses recorded"
    assert len(history.val_losses) > 0, "No validation losses recorded"

    # Test prediction
    test_preds = trainer.split_nn.predict(
        torch.FloatTensor(X_a_val[:10]),
        torch.FloatTensor(X_b_val[:10])
    )

    assert test_preds.shape == (10, 2), f"Predictions shape incorrect: {test_preds.shape}"

    print("✓ Integration with trainer working correctly")
    print(f"  Epochs completed: {len(history.train_losses)}")
    print(f"  Final train loss: {history.train_losses[-1]:.4f}")
    print(f"  Final val loss: {history.val_losses[-1]:.4f}")

    return True


def run_all_tests():
    """Run all SplitNN tests."""
    print("\n" + "="*80)
    print("SPLITNN INTEGRATION TESTS")
    print("="*80)

    tests = [
        test_split_nn_initialization,
        test_split_nn_forward_pass,
        test_split_nn_backward_pass,
        test_split_nn_train_eval_modes,
        test_split_nn_predict,
        test_split_nn_save_load,
        test_split_nn_integration_with_trainer,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except AssertionError as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
        except Exception as e:
            print(f"\n✗ {test.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
