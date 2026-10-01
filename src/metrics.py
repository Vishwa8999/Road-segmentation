"""Losses and thresholded binary segmentation metrics."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 4.0/3.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    
    tp = (probabilities * targets).sum(dim=dims)
    fp = (probabilities * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - probabilities) * targets).sum(dim=dims)
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.pow((1.0 - tversky_index), gamma).mean()


class HybridLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, focal_tversky_weight: float = 0.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * dice_loss(logits, targets)
        if self.focal_tversky_weight > 0:
            loss = loss + self.focal_tversky_weight * focal_tversky_loss(logits, targets)
        return loss


class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = torch.sigmoid(logits) >= self.threshold
        truth = targets >= 0.5
        self.tp += torch.logical_and(predictions, truth).sum().item()
        self.fp += torch.logical_and(predictions, ~truth).sum().item()
        self.fn += torch.logical_and(~predictions, truth).sum().item()
        self.tn += torch.logical_and(~predictions, ~truth).sum().item()

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "dice": (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
            "pixel_accuracy": (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps),
        }
