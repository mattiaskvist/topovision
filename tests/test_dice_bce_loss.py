"""Tests for DiceBCELoss."""

import pytest
import torch

from training.losses import DiceBCELoss


@pytest.fixture
def loss_fn():
    """Create default DiceBCELoss."""
    return DiceBCELoss()


@pytest.fixture
def loss_fn_weighted():
    """Create DiceBCELoss with custom weights."""
    return DiceBCELoss(dice_weight=2.0, bce_weight=0.5)


# --- Basic Forward Pass Tests ---


def test_forward_returns_tuple(loss_fn):
    """Test that forward returns (loss, dict) tuple."""
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    result = loss_fn(logits, targets)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], dict)


def test_forward_loss_dict_contains_components(loss_fn):
    """Test that loss dict contains dice and bce components."""
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    _, loss_dict = loss_fn(logits, targets)

    assert "dice" in loss_dict
    assert "bce" in loss_dict
    assert isinstance(loss_dict["dice"], float)
    assert isinstance(loss_dict["bce"], float)


def test_forward_loss_is_scalar(loss_fn):
    """Test that total loss is a scalar tensor."""
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, _ = loss_fn(logits, targets)

    assert total_loss.dim() == 0  # Scalar


def test_forward_loss_is_positive(loss_fn):
    """Test that loss is non-negative."""
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, loss_dict = loss_fn(logits, targets)

    assert total_loss.item() >= 0
    assert loss_dict["dice"] >= 0
    assert loss_dict["bce"] >= 0


# --- Edge Cases ---


def test_perfect_prediction(loss_fn):
    """Test loss when prediction perfectly matches target."""
    # High logits where target is 1, low where target is 0
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0

    # Create logits that will produce perfect prediction after sigmoid
    logits = torch.full_like(targets, -10.0)  # Very low = 0 after sigmoid
    logits[targets == 1] = 10.0  # Very high = 1 after sigmoid

    _total_loss, loss_dict = loss_fn(logits, targets)

    # With perfect prediction, dice loss should be close to 0
    assert loss_dict["dice"] < 0.01
    # BCE should also be very low
    assert loss_dict["bce"] < 0.01


def test_all_zeros_prediction(loss_fn):
    """Test loss with all-zero predictions against mixed targets."""
    logits = torch.full((1, 1, 32, 32), -10.0)  # All zeros after sigmoid
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0  # Some ones

    total_loss, loss_dict = loss_fn(logits, targets)

    # Dice should be high (close to 1) since no intersection
    assert loss_dict["dice"] > 0.9
    # Total loss should be positive
    assert total_loss.item() > 0


def test_all_ones_prediction(loss_fn):
    """Test loss with all-one predictions against mixed targets."""
    logits = torch.full((1, 1, 32, 32), 10.0)  # All ones after sigmoid
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0  # Some ones

    total_loss, _loss_dict = loss_fn(logits, targets)

    # Loss should be moderate (not perfect match)
    assert total_loss.item() > 0


def test_all_zeros_target(loss_fn):
    """Test loss with all-zero targets."""
    logits = torch.randn(1, 1, 32, 32)
    targets = torch.zeros(1, 1, 32, 32)

    total_loss, _loss_dict = loss_fn(logits, targets)

    # Should still compute without errors
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)


def test_all_ones_target(loss_fn):
    """Test loss with all-one targets."""
    logits = torch.randn(1, 1, 32, 32)
    targets = torch.ones(1, 1, 32, 32)

    total_loss, _loss_dict = loss_fn(logits, targets)

    # Should still compute without errors
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)


def test_batch_size_one(loss_fn):
    """Test with batch size of 1."""
    logits = torch.randn(1, 1, 64, 64)
    targets = torch.randint(0, 2, (1, 1, 64, 64)).float()

    total_loss, _loss_dict = loss_fn(logits, targets)

    assert not torch.isnan(total_loss)


def test_large_batch_size(loss_fn):
    """Test with larger batch size."""
    logits = torch.randn(16, 1, 64, 64)
    targets = torch.randint(0, 2, (16, 1, 64, 64)).float()

    total_loss, _loss_dict = loss_fn(logits, targets)

    assert not torch.isnan(total_loss)


def test_different_spatial_sizes(loss_fn):
    """Test with various spatial dimensions."""
    for size in [32, 64, 128, 256]:
        logits = torch.randn(2, 1, size, size)
        targets = torch.randint(0, 2, (2, 1, size, size)).float()

        total_loss, _ = loss_fn(logits, targets)

        assert not torch.isnan(total_loss)


# --- Weight Combinations ---


def test_custom_weights(loss_fn_weighted):
    """Test that custom weights are applied correctly."""
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, _loss_dict = loss_fn_weighted(logits, targets)

    # Verify the loss is computed (weights affect total, not components)
    assert not torch.isnan(total_loss)
    assert loss_fn_weighted.dice_weight == 2.0
    assert loss_fn_weighted.bce_weight == 0.5


def test_zero_dice_weight():
    """Test with zero dice weight (BCE only)."""
    loss_fn = DiceBCELoss(dice_weight=0.0, bce_weight=1.0)
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, loss_dict = loss_fn(logits, targets)

    # Total loss should equal BCE component
    assert abs(total_loss.item() - loss_dict["bce"]) < 1e-5


def test_zero_bce_weight():
    """Test with zero BCE weight (Dice only)."""
    loss_fn = DiceBCELoss(dice_weight=1.0, bce_weight=0.0)
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, loss_dict = loss_fn(logits, targets)

    # Total loss should equal Dice component
    assert abs(total_loss.item() - loss_dict["dice"]) < 1e-5


def test_equal_weights_same_as_default():
    """Test that equal weights produce expected behavior."""
    loss_fn_default = DiceBCELoss()
    loss_fn_explicit = DiceBCELoss(dice_weight=1.0, bce_weight=1.0)

    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss1, _ = loss_fn_default(logits, targets)
    loss2, _ = loss_fn_explicit(logits, targets)

    assert torch.allclose(loss1, loss2)


# --- Gradient Flow ---


def test_gradient_flow(loss_fn):
    """Test that gradients flow back through the loss."""
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, _ = loss_fn(logits, targets)
    total_loss.backward()

    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)


def test_gradient_not_nan(loss_fn):
    """Test that gradients are not NaN."""
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    total_loss, _ = loss_fn(logits, targets)
    total_loss.backward()

    assert not torch.any(torch.isnan(logits.grad))


def test_gradient_with_extreme_logits(loss_fn):
    """Test gradients with extreme logit values."""
    # Very large positive and negative logits
    logits = torch.tensor([[[[100.0, -100.0], [50.0, -50.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    total_loss, _ = loss_fn(logits, targets)
    total_loss.backward()

    # Gradients should exist and not be NaN
    assert logits.grad is not None
    assert not torch.any(torch.isnan(logits.grad))


# --- Numerical Stability ---


def test_numerical_stability_sparse_target(loss_fn):
    """Test numerical stability with very sparse targets."""
    logits = torch.randn(1, 1, 128, 128)
    targets = torch.zeros(1, 1, 128, 128)
    targets[0, 0, 64, 64] = 1.0  # Single pixel

    total_loss, _loss_dict = loss_fn(logits, targets)

    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)


def test_numerical_stability_smooth_factor():
    """Test that smooth factor prevents division by zero."""
    loss_fn = DiceBCELoss()

    # All zeros in both prediction and target
    logits = torch.full((1, 1, 32, 32), -100.0)  # All zeros after sigmoid
    targets = torch.zeros(1, 1, 32, 32)

    total_loss, _loss_dict = loss_fn(logits, targets)

    # Should not produce NaN or Inf due to smooth factor
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)
