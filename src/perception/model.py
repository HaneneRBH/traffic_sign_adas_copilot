"""
Perception model for the Traffic-Sign ADAS Copilot.

We fine-tune a MobileNetV3-Small (ImageNet-pretrained) on GTSRB. MobileNetV3-Small
is chosen because it is compact and deployment-friendly (few parameters,
INT8-quantizable), which keeps the perception stage fast on CPU.

Freeze strategies:
    'all'        -> freeze the whole feature extractor, train only the classifier
                    head (fastest on CPU; good warm-up).
    'last_block' -> also unfreeze the last few feature blocks (better accuracy,
                    still CPU-tractable). Default.
    'none'       -> full fine-tuning (slowest on CPU).
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 43


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = True,
                freeze: str = "last_block") -> nn.Module:
    """Create a MobileNetV3-Small with a GTSRB classification head."""
    # torchvision weights API (torch 2.5 / torchvision 0.20). Fallback for old API.
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:  # very old torchvision
        model = models.mobilenet_v3_small(pretrained=pretrained)

    # Replace the final Linear (classifier is a small Sequential ending in Linear).
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    _apply_freeze(model, freeze)
    return model


def _apply_freeze(model: nn.Module, freeze: str) -> None:
    if freeze == "none":
        for p in model.parameters():
            p.requires_grad = True
        return

    # Freeze the whole feature extractor first.
    for p in model.features.parameters():
        p.requires_grad = False
    # Classifier head is always trainable.
    for p in model.classifier.parameters():
        p.requires_grad = True

    if freeze == "last_block":
        # Unfreeze the last 3 feature blocks for extra capacity.
        for block in list(model.features.children())[-3:]:
            for p in block.parameters():
                p.requires_grad = True
    elif freeze != "all":
        raise ValueError(f"Unknown freeze strategy: {freeze!r}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    """Approximate on-disk size in MB from parameter + buffer byte counts."""
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)
