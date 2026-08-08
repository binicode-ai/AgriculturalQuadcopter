"""
ai/training.py

Training utilities for the AgriculturalQuadcopter
crop disease classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ai.cnn import (
    create_transfer_learning_model,
)
from ai.dataset import (
    create_dataset,
    CLASS_NAMES,
)

from ai.augmentation import (
    get_train_transform,
    get_validation_transform,
)

from ai.cnn import create_model


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Class weights
# ============================================================

CLASS_WEIGHTS = torch.tensor(
    [
        0.9662,  # Blight
        0.3762,  # Healthy
        1.6940,  # LeafSpot
        1.9717,  # Mildew
        4.7748,  # Rust
    ],
    dtype=torch.float32,
)


# ============================================================
# Create DataLoader
# ============================================================

def create_dataloader(
    split,
    batch_size=BATCH_SIZE,
):

    if split == "train":

        transform = (
            get_train_transform()
        )

        shuffle = True

    elif split == "validation":

        transform = (
            get_validation_transform()
        )

        shuffle = False

    else:

        raise ValueError(
            "split must be "
            "'train' or 'validation'"
        )

    dataset = create_dataset(
        split,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return dataset, loader


# ============================================================
# Create loss function
# ============================================================

def create_loss_function():

    weights = CLASS_WEIGHTS.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    return criterion


# ============================================================
# Create optimizer
# ============================================================

def create_optimizer(model):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return optimizer


# ============================================================
# Verify one training batch
# ============================================================

def verify_training_batch():

    print("=" * 60)
    print(
        "AgriculturalQuadcopter"
    )
    print(
        "Training Pipeline Verification"
    )
    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset, loader = (
        create_dataloader(
            "train"
        )
    )

    print(
        f"Training images: "
        f"{len(dataset)}"
    )

    print(
        f"Training batches: "
        f"{len(loader)}"
    )

    print()

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    images, labels = next(
        iter(loader)
    )

    print(
        "Input batch shape: "
        f"{tuple(images.shape)}"
    )

    print(
        "Label batch shape: "
        f"{tuple(labels.shape)}"
    )

    print(
        "Labels:"
    )

    print(labels.tolist())

    # --------------------------------------------------------
    # Verify batch
    # --------------------------------------------------------

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224
    )

    assert labels.shape == (
        BATCH_SIZE,
    )

    assert images.dtype == (
        torch.float32
    )

    assert labels.dtype == (
        torch.int64
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        pretrained=False
    )

    model = model.to(
        DEVICE
    )

    images = images.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model(
        images
    )

    print()

    print(
        "Model output shape: "
        f"{tuple(outputs.shape)}"
    )

    assert outputs.shape == (
        BATCH_SIZE,
        len(CLASS_NAMES)
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        create_loss_function()
    )

    loss = criterion(
        outputs,
        labels
    )

    print(
        f"Initial loss: "
        f"{loss.item():.4f}"
    )

    assert torch.isfinite(
        loss
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    optimizer = (
        create_optimizer(
            model
        )
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print()

    print(
        "Forward pass:  PASSED"
    )

    print(
        "Loss function: PASSED"
    )

    print(
        "Backward pass: PASSED"
    )

    print(
        "Optimizer step: PASSED"
    )

    print()

    print("=" * 60)
    print(
        "TRAINING PIPELINE VERIFICATION PASSED"
    )
    print("=" * 60)


def main():

    verify_training_batch()


if __name__ == "__main__":

    main()