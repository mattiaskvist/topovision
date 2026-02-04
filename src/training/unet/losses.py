"""Loss functions for contour segmentation training."""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross-Entropy loss.

    Dice loss helps with class imbalance (contour lines are sparse),
    while BCE provides stable gradients.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        """Initialize the loss function."""
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute the combined loss.

        Args:
            logits: Model output logits [B, 1, H, W].
            targets: Ground truth masks [B, 1, H, W].

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # BCE loss
        bce_loss = self.bce(logits, targets)

        # Dice loss (on sigmoid probabilities)
        probs = torch.sigmoid(logits)
        smooth = 1e-6

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (
            probs_flat.sum() + targets_flat.sum() + smooth
        )
        dice_loss = 1.0 - dice_score

        # Combined
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return total_loss, {"dice": dice_loss.item(), "bce": bce_loss.item()}
