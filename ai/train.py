"""
ai/train.py

Transfer-learning training for the
AgriculturalQuadcopter crop disease classifier.

This script:

1. Loads the training and validation datasets.
2. Creates an ImageNet-pretrained MobileNetV3-Small.
3. Freezes the pretrained backbone.
4. Trains the 5-class classifier.
5. Calculates training and validation metrics.
6. Saves the best model based on validation accuracy.
7. Saves a checkpoint containing model metadata.

Classes:
    0 - Blight
    1 - Healthy
    2 - LeafSpot
    3 - Mildew
    4 - Rust
"""

import os

import torch


from ai.training import (
    create_dataloader,
    create_loss_function,
    DEVICE,
)

from ai.cnn import (
    create_transfer_learning_model,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 2

BATCH_SIZE = 16

LEARNING_RATE = 0.001

MODEL_DIR = "trained_models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "crop_disease_mobilenet_best.pth"
)


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    outputs,
    labels
):
    """
    Calculate classification accuracy
    for one batch.
    """

    predictions = outputs.argmax(
        dim=1
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    if total == 0:
        return 0.0

    return correct / total


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    """
    Train the model for one epoch.

    Returns:
        average_loss
        accuracy
    """

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        current_batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        total_correct += (
            (
                predictions == labels
            )
            .sum()
            .item()
        )

        total_samples += (
            current_batch_size
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            batch_accuracy = (
                calculate_accuracy(
                    outputs,
                    labels
                )
            )

            print(
                f"  Batch "
                f"{batch_index + 1:4d}"
                f"/{len(loader)}"
                f" | Loss: "
                f"{loss.item():.4f}"
                f" | Accuracy: "
                f"{batch_accuracy:.4f}"
            )

    # --------------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------------

    if total_samples == 0:

        return 0.0, 0.0

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
):
    """
    Evaluate the model on the validation dataset.

    No gradients are calculated.

    Returns:
        average_loss
        accuracy
    """

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            current_batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * current_batch_size
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            total_correct += (
                (
                    predictions == labels
                )
                .sum()
                .item()
            )

            total_samples += (
                current_batch_size
            )

    if total_samples == 0:

        return 0.0, 0.0

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# Save best model
# ============================================================

def save_best_model(
    model,
    epoch,
    validation_loss,
    validation_accuracy,
):
    """
    Save the current model as the best model.
    """

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "num_classes":
            len(CLASS_NAMES),

        "epoch":
            epoch,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "architecture":
            "MobileNetV3-Small",

        "pretrained":
            True,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print()

    print(
        "Best model saved:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"  Validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"  Validation loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"  Epoch: "
        f"{epoch}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Transfer Learning Training"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Configuration information
    # --------------------------------------------------------

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print()

    print(
        "Classes:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index}: {class_name}"
        )

    print()

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create datasets and loaders
    # --------------------------------------------------------

    train_dataset, train_loader = (
        create_dataloader(
            "train",
            batch_size=BATCH_SIZE
        )
    )

    validation_dataset, validation_loader = (
        create_dataloader(
            "validation",
            batch_size=BATCH_SIZE
        )
    )

    print()

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    print()

    # --------------------------------------------------------
    # Create pretrained model
    # --------------------------------------------------------

    print(
        "Creating pretrained "
        "MobileNetV3-Small..."
    )

    model = (
        create_transfer_learning_model()
    )

    model = model.to(
        DEVICE
    )

    print(
        "Model created successfully."
    )

    print()

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print(
        "Model parameters:"
    )

    print(
        f"  Total:     "
        f"{total_parameters:,}"
    )

    print(
        f"  Trainable: "
        f"{trainable_parameters:,}"
    )

    print(
        f"  Frozen:    "
        f"{frozen_parameters:,}"
    )

    print()

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = (
        create_loss_function()
    )

    print(
        "Weighted CrossEntropyLoss: "
        "READY"
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    #
    # Only parameters with
    # requires_grad=True are trained.
    # --------------------------------------------------------

    trainable_parameters_list = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters_list,
        lr=LEARNING_RATE
    )

    print(
        "Adam optimizer: READY"
    )

    print()

    # --------------------------------------------------------
    # Best-model tracking
    # --------------------------------------------------------

    best_validation_accuracy = 0.0

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        current_epoch = (
            epoch + 1
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"Epoch "
            f"{current_epoch}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
            )
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print()

        print(
            "Epoch results:"
        )

        print(
            f"  Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train accuracy: "
            f"{train_accuracy:.4f}"
            f" ({train_accuracy * 100:.2f}%)"
        )

        print(
            f"  Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"  Validation accuracy: "
            f"{validation_accuracy:.4f}"
            f" ({validation_accuracy * 100:.2f}%)"
        )

        # ----------------------------------------------------
        # Check whether this is the best model
        # ----------------------------------------------------

        is_better = (
            validation_accuracy
            > best_validation_accuracy
        )

        if is_better:

            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = (
                current_epoch
            )

            save_best_model(
                model=model,
                epoch=current_epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )

        else:

            print()

            print(
                "Best model unchanged."
            )

            print(
                f"  Best validation accuracy: "
                f"{best_validation_accuracy:.4f}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "TRANSFER LEARNING TRAINING COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Best model:"
    )

    print(
        f"  File: "
        f"{MODEL_PATH}"
    )

    print(
        f"  Epoch: "
        f"{best_epoch}"
    )

    print(
        f"  Validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"  Validation accuracy: "
        f"{best_validation_accuracy:.4f}"
        f" ({best_validation_accuracy * 100:.2f}%)"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "  python -m ai.evaluate"
    )

    print()

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()