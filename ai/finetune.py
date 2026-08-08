"""
ai/finetune.py

Targeted fine-tuning experiment for
AgriculturalQuadcopter.

Purpose:
    Improve disease classification after
    the initial transfer-learning baseline.

Baseline:
    MobileNetV3-Small
    Test accuracy: 97.28%

Target:
    Improve LeafSpot recall while maintaining
    overall model performance.
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

EPOCHS = 3

BATCH_SIZE = 16

LEARNING_RATE = 0.0001

BASELINE_MODEL = os.path.join(
    "trained_models",
    "crop_disease_mobilenet_best.pth"
)

FINETUNED_MODEL = os.path.join(
    "trained_models",
    "crop_disease_mobilenet_finetuned.pth"
)


# ============================================================
# Load baseline checkpoint
# ============================================================

def load_baseline_model():

    print(
        "Loading baseline model..."
    )

    checkpoint = torch.load(
        BASELINE_MODEL,
        map_location=DEVICE
    )

    model = (
        create_transfer_learning_model()
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    print(
        "Baseline model loaded."
    )

    print(
        f"Baseline validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    print(
        f"Baseline epoch: "
        f"{checkpoint['epoch']}"
    )

    return model


# ============================================================
# Unfreeze later layers
# ============================================================

def prepare_for_finetuning(model):

    print()

    print(
        "Preparing model for fine-tuning..."
    )

    # --------------------------------------------------------
    # First freeze everything
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # MobileNetV3-Small classifier
    #
    # The classifier must remain trainable.
    # --------------------------------------------------------

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True

    # --------------------------------------------------------
    # Unfreeze the final feature blocks.
    #
    # MobileNetV3-Small contains a Sequential
    # feature extractor.
    #
    # We deliberately unfreeze only the final
    # portion rather than the entire network.
    # --------------------------------------------------------

    feature_count = len(
        model.features
    )

    number_to_unfreeze = 3

    start_index = max(
        0,
        feature_count
        - number_to_unfreeze
    )

    for block_index in range(
        start_index,
        feature_count
    ):

        for parameter in (
            model.features[
                block_index
            ].parameters()
        ):

            parameter.requires_grad = True

    print(
        f"Total feature blocks: "
        f"{feature_count}"
    )

    print(
        f"Unfrozen final blocks: "
        f"{start_index} "
        f"to "
        f"{feature_count - 1}"
    )

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

    print()

    print(
        "Fine-tuning parameters:"
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

    return model


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
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
            batch_size
        )

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1:4d}"
                f"/{len(loader)}"
                f" | Loss: "
                f"{loss.item():.4f}"
            )

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

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
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
                batch_size
            )

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
# Save model
# ============================================================

def save_model(
    model,
    epoch,
    validation_loss,
    validation_accuracy,
):

    os.makedirs(
        "trained_models",
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

        "training_type":
            "targeted_fine_tuning",

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,
    }

    torch.save(
        checkpoint,
        FINETUNED_MODEL
    )

    print()

    print(
        "Fine-tuned model saved:"
    )

    print(
        f"  {FINETUNED_MODEL}"
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
        "Targeted MobileNetV3 Fine-Tuning"
    )

    print("=" * 60)

    print()

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

    # --------------------------------------------------------
    # Check baseline model
    # --------------------------------------------------------

    if not os.path.exists(
        BASELINE_MODEL
    ):

        raise FileNotFoundError(
            "Baseline model not found:\n"
            f"{BASELINE_MODEL}\n\n"
            "Run first:\n"
            "python -m ai.train"
        )

    # --------------------------------------------------------
    # Dataset
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

    print()

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    model = (
        load_baseline_model()
    )

    # --------------------------------------------------------
    # Prepare fine-tuning
    # --------------------------------------------------------

    model = (
        prepare_for_finetuning(
            model
        )
    )

    model = model.to(
        DEVICE
    )

    print()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        create_loss_function()
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE
    )

    print(
        "Optimizer: Adam"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Best model tracking
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
            f"Fine-tuning Epoch "
            f"{current_epoch}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
            )
        )

        print()

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.4f}"
            f" ({train_accuracy * 100:.2f}%)"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
            f" ({validation_accuracy * 100:.2f}%)"
        )

        # ----------------------------------------------------
        # Save best fine-tuned model
        # ----------------------------------------------------

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = (
                current_epoch
            )

            save_model(
                model,
                current_epoch,
                validation_loss,
                validation_accuracy,
            )

        else:

            print()

            print(
                "Fine-tuned best model "
                "unchanged."
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "FINE-TUNING EXPERIMENT COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
        f" ({best_validation_accuracy * 100:.2f}%)"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print()

    print(
        "Model:"
    )

    print(
        f"  {FINETUNED_MODEL}"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "  Evaluate the fine-tuned model "
        "on the test set."
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()