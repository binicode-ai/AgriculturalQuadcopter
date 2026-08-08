"""
ai/targeted_blspot_finetune.py

Targeted fine-tuning experiment for the five-class
AgriculturalQuadcopter crop disease classifier.

Goal:
    Reduce Blight <-> LeafSpot confusion while preserving
    performance on Healthy, Mildew, and Rust.

Five classes ONLY:
    Blight
    Healthy
    LeafSpot
    Mildew
    Rust

Important:
    Any extra directory inside train/validation, such as
    "crop_disease", is ignored and is NOT treated as a class.
"""

import os
import copy

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from ai.cnn import (
    create_transfer_learning_model,
)

from ai.training import DEVICE


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = os.path.join(
    "datasets",
    "crop_disease",
)

TRAIN_DIRECTORY = os.path.join(
    DATASET_ROOT,
    "train",
)

VALIDATION_DIRECTORY = os.path.join(
    DATASET_ROOT,
    "validation",
)

MODEL_DIRECTORY = "trained_models"

BASELINE_MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "crop_disease_mobilenet_finetuned.pth",
)

OUTPUT_MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "crop_disease_mobilenet_blspot_targeted.pth",
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 3

LEARNING_RATE = 0.00005

NUM_WORKERS = 0


# ============================================================
# Five-class configuration
# ============================================================

EXPECTED_CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ------------------------------------------------------------
# Only these two classes receive extra loss weight.
# ------------------------------------------------------------

TARGET_CLASSES = [
    "Blight",
    "LeafSpot",
]

TARGET_WEIGHT = 1.5

NORMAL_WEIGHT = 1.0


# ============================================================
# Utility
# ============================================================

def normalize_path(path):
    """
    Convert a path into an absolute normalized path.
    """

    return os.path.abspath(
        os.path.normpath(
            path
        )
    )


# ============================================================
# Dataset directory verification
# ============================================================

def verify_dataset_directory(
    directory,
    split_name,
):
    """
    Verify that the required five disease directories exist.

    Extra directories are allowed and ignored.

    Example:

        train/
            Blight/
            Healthy/
            LeafSpot/
            Mildew/
            Rust/
            crop_disease/   <-- ignored
    """

    directory = normalize_path(
        directory
    )

    if not os.path.isdir(
        directory
    ):

        raise FileNotFoundError(
            f"{split_name.capitalize()} dataset directory "
            f"does not exist:\n\n"
            f"{directory}"
        )

    print(
        f"Using {split_name} dataset directory:"
    )

    print(
        f"  {directory}"
    )

    print()

    # --------------------------------------------------------
    # Find required classes.
    # --------------------------------------------------------

    missing_classes = []

    for class_name in EXPECTED_CLASSES:

        class_directory = os.path.join(
            directory,
            class_name,
        )

        if not os.path.isdir(
            class_directory
        ):

            missing_classes.append(
                class_name
            )

    if missing_classes:

        raise RuntimeError(
            f"Invalid {split_name} dataset structure.\n\n"
            f"Missing required classes:\n"
            f"  {missing_classes}\n\n"
            f"Expected five classes:\n"
            f"  {EXPECTED_CLASSES}\n\n"
            f"Dataset directory:\n"
            f"  {directory}"
        )

    # --------------------------------------------------------
    # Detect extra directories.
    # --------------------------------------------------------

    actual_directories = []

    for name in os.listdir(
        directory
    ):

        full_path = os.path.join(
            directory,
            name,
        )

        if os.path.isdir(
            full_path
        ):

            actual_directories.append(
                name
            )

    extra_directories = [
        name
        for name in actual_directories
        if name not in EXPECTED_CLASSES
    ]

    if extra_directories:

        print(
            "Ignoring extra directories:"
        )

        for name in extra_directories:

            print(
                f"  {name}"
            )

        print()

    print(
        f"{split_name.capitalize()} dataset contains "
        f"all five required classes."
    )

    print()


# ============================================================
# Five-class ImageFolder
# ============================================================

class FiveClassImageFolder(
    datasets.ImageFolder
):
    """
    ImageFolder restricted to exactly five classes.

    torchvision.ImageFolder normally treats every directory
    as a class.

    This custom implementation prevents accidental folders
    such as:

        crop_disease/

    from becoming a sixth class.
    """

    def find_classes(
        self,
        directory,
    ):

        directory = normalize_path(
            directory
        )

        # ----------------------------------------------------
        # Require all five classes.
        # ----------------------------------------------------

        missing_classes = []

        for class_name in EXPECTED_CLASSES:

            class_directory = os.path.join(
                directory,
                class_name,
            )

            if not os.path.isdir(
                class_directory
            ):

                missing_classes.append(
                    class_name
                )

        if missing_classes:

            raise RuntimeError(
                "Five-class dataset verification failed.\n\n"
                f"Directory:\n"
                f"  {directory}\n\n"
                f"Missing classes:\n"
                f"  {missing_classes}\n\n"
                f"Expected:\n"
                f"  {EXPECTED_CLASSES}"
            )

        # ----------------------------------------------------
        # Explicit class order.
        #
        # IMPORTANT:
        #
        # 0 = Blight
        # 1 = Healthy
        # 2 = LeafSpot
        # 3 = Mildew
        # 4 = Rust
        # ----------------------------------------------------

        classes = list(
            EXPECTED_CLASSES
        )

        class_to_idx = {
            class_name: index
            for index, class_name in enumerate(
                classes
            )
        }

        return (
            classes,
            class_to_idx,
        )


# ============================================================
# Transformations
# ============================================================

def create_train_transform():

    return transforms.Compose([

        # ----------------------------------------------------
        # Random crop first.
        # ----------------------------------------------------

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=20
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


def create_validation_transform():

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# ============================================================
# Dataset loading
# ============================================================

def load_datasets():

    print(
        "Loading datasets..."
    )

    print()

    # --------------------------------------------------------
    # Verify directories before ImageFolder.
    # --------------------------------------------------------

    train_directory = normalize_path(
        TRAIN_DIRECTORY
    )

    validation_directory = normalize_path(
        VALIDATION_DIRECTORY
    )

    verify_dataset_directory(
        train_directory,
        "training",
    )

    verify_dataset_directory(
        validation_directory,
        "validation",
    )

    # --------------------------------------------------------
    # Create five-class datasets.
    # --------------------------------------------------------

    train_dataset = FiveClassImageFolder(
        root=train_directory,
        transform=create_train_transform(),
    )

    validation_dataset = FiveClassImageFolder(
        root=validation_directory,
        transform=create_validation_transform(),
    )

    # --------------------------------------------------------
    # Verify class order.
    # --------------------------------------------------------

    if train_dataset.classes != EXPECTED_CLASSES:

        raise RuntimeError(
            "Unexpected training class order.\n\n"
            f"Expected:\n"
            f"  {EXPECTED_CLASSES}\n\n"
            f"Found:\n"
            f"  {train_dataset.classes}"
        )

    if validation_dataset.classes != EXPECTED_CLASSES:

        raise RuntimeError(
            "Unexpected validation class order.\n\n"
            f"Expected:\n"
            f"  {EXPECTED_CLASSES}\n\n"
            f"Found:\n"
            f"  {validation_dataset.classes}"
        )

    # --------------------------------------------------------
    # Print dataset statistics.
    # --------------------------------------------------------

    print(
        "Training class distribution:"
    )

    print()

    for class_name in EXPECTED_CLASSES:

        class_index = (
            train_dataset.class_to_idx[
                class_name
            ]
        )

        count = sum(
            1
            for _, label in train_dataset.samples
            if label == class_index
        )

        print(
            f"  {class_name:<10}: "
            f"{count:6d}"
        )

    print()

    print(
        "Validation class distribution:"
    )

    print()

    for class_name in EXPECTED_CLASSES:

        class_index = (
            validation_dataset.class_to_idx[
                class_name
            ]
        )

        count = sum(
            1
            for _, label in validation_dataset.samples
            if label == class_index
        )

        print(
            f"  {class_name:<10}: "
            f"{count:6d}"
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

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# DataLoaders
# ============================================================

def create_loaders(
    train_dataset,
    validation_dataset,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# Load baseline model
# ============================================================

def load_baseline_model():

    if not os.path.exists(
        BASELINE_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Baseline fine-tuned model was not found:\n"
            f"{BASELINE_MODEL_PATH}\n\n"
            "Run the normal fine-tuning experiment first:\n"
            "  python -m ai.finetune"
        )

    print(
        "Loading baseline fine-tuned model..."
    )

    checkpoint = torch.load(
        BASELINE_MODEL_PATH,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Validate checkpoint metadata.
    # --------------------------------------------------------

    checkpoint_classes = checkpoint.get(
        "class_names",
        EXPECTED_CLASSES,
    )

    if list(checkpoint_classes) != EXPECTED_CLASSES:

        raise RuntimeError(
            "Baseline checkpoint class order does not match "
            "the five-class project configuration.\n\n"
            f"Expected:\n"
            f"  {EXPECTED_CLASSES}\n\n"
            f"Checkpoint:\n"
            f"  {checkpoint_classes}"
        )

    checkpoint_num_classes = checkpoint.get(
        "num_classes",
        len(EXPECTED_CLASSES),
    )

    if checkpoint_num_classes != len(
        EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Baseline checkpoint does not contain "
            "exactly five classes.\n\n"
            f"Expected: 5\n"
            f"Found:    {checkpoint_num_classes}"
        )

    # --------------------------------------------------------
    # Create model architecture.
    # --------------------------------------------------------

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

    model.train()

    print(
        "Baseline model loaded."
    )

    print()

    print(
        "Baseline architecture: "
        f"{checkpoint.get('architecture', 'MobileNetV3-Small')}"
    )

    print(
        "Baseline training type: "
        f"{checkpoint.get('training_type', 'targeted_fine_tuning')}"
    )

    print(
        "Baseline validation accuracy: "
        f"{checkpoint.get('validation_accuracy', 0.0):.4f}"
    )

    print(
        "Baseline validation loss: "
        f"{checkpoint.get('validation_loss', 0.0):.4f}"
    )

    print(
        "Baseline epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    print()

    return model


# ============================================================
# Prepare targeted fine-tuning
# ============================================================

def prepare_model(
    model
):

    # --------------------------------------------------------
    # Freeze everything first.
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # MobileNetV3-Small feature blocks.
    #
    # Previous successful experiment unfroze the final
    # three blocks.
    # --------------------------------------------------------

    features = model.features

    total_blocks = len(
        features
    )

    start_block = max(
        0,
        total_blocks - 3,
    )

    for index in range(
        start_block,
        total_blocks,
    ):

        for parameter in features[
            index
        ].parameters():

            parameter.requires_grad = True

    # --------------------------------------------------------
    # Classifier must remain trainable.
    # --------------------------------------------------------

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True

    print(
        f"Total feature blocks: "
        f"{total_blocks}"
    )

    print(
        f"Unfrozen final blocks: "
        f"{start_block} to "
        f"{total_blocks - 1}"
    )

    print()

    return model


# ============================================================
# Targeted loss
# ============================================================

def create_targeted_loss():

    weights = []

    for class_name in EXPECTED_CLASSES:

        if class_name in TARGET_CLASSES:

            weights.append(
                TARGET_WEIGHT
            )

        else:

            weights.append(
                NORMAL_WEIGHT
            )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        "Targeted class weights:"
    )

    for class_name, weight in zip(
        EXPECTED_CLASSES,
        weights,
    ):

        print(
            f"  {class_name:<10}: "
            f"{weight.item():.2f}"
        )

    print()

    return nn.CrossEntropyLoss(
        weight=weights
    )


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
        labels,
    ) in enumerate(
        loader
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Clear gradients.
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass.
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Targeted loss.
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels,
        )

        # ----------------------------------------------------
        # Backward pass.
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics.
        # ----------------------------------------------------

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
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

        # ----------------------------------------------------
        # Progress.
        # ----------------------------------------------------

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
        accuracy,
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
                labels,
            )

            batch_size = labels.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = outputs.argmax(
                dim=1
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
        accuracy,
    )


# ============================================================
# Parameter information
# ============================================================

def print_parameter_counts(
    model
):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    print(
        "Fine-tuning parameters:"
    )

    print(
        f"  Total:     {total:,}"
    )

    print(
        f"  Trainable: {trainable:,}"
    )

    print(
        f"  Frozen:    {frozen:,}"
    )

    print()


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    model,
    epoch,
    validation_loss,
    validation_accuracy,
):

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True,
    )

    checkpoint = {

        # ----------------------------------------------------
        # Model.
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        # ----------------------------------------------------
        # Five-class configuration.
        # ----------------------------------------------------

        "class_names":
            list(EXPECTED_CLASSES),

        "num_classes":
            len(EXPECTED_CLASSES),

        # ----------------------------------------------------
        # Training information.
        # ----------------------------------------------------

        "epoch":
            epoch,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        # ----------------------------------------------------
        # Architecture metadata.
        # ----------------------------------------------------

        "architecture":
            "MobileNetV3-Small",

        "training_type":
            "targeted_blspot_fine_tuning",

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        # ----------------------------------------------------
        # Targeted fine-tuning metadata.
        # ----------------------------------------------------

        "target_classes":
            list(TARGET_CLASSES),

        "target_weight":
            TARGET_WEIGHT,

        "normal_weight":
            NORMAL_WEIGHT,

        "unfrozen_feature_blocks":
            3,
    }

    torch.save(
        checkpoint,
        OUTPUT_MODEL_PATH,
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
        "Targeted Blight ↔ LeafSpot Fine-Tuning"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Configuration.
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

    print(
        "Target classes: "
        f"{', '.join(TARGET_CLASSES)}"
    )

    print(
        f"Target loss weight: "
        f"{TARGET_WEIGHT}"
    )

    print()

    print(
        "Five-class configuration:"
    )

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        print(
            f"  {index} = {class_name}"
        )

    print()

    print(
        "Dataset root:"
    )

    print(
        f"  {normalize_path(DATASET_ROOT)}"
    )

    print()

    # --------------------------------------------------------
    # Load datasets.
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
    ) = load_datasets()

    # --------------------------------------------------------
    # Create loaders.
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
    ) = create_loaders(
        train_dataset,
        validation_dataset,
    )

    # --------------------------------------------------------
    # Load baseline model.
    # --------------------------------------------------------

    print(
        "Loading baseline model..."
    )

    print()

    model = load_baseline_model()

    # --------------------------------------------------------
    # Prepare model.
    # --------------------------------------------------------

    print(
        "Preparing model for targeted "
        "fine-tuning..."
    )

    print()

    model = prepare_model(
        model
    )

    model = model.to(
        DEVICE
    )

    print_parameter_counts(
        model
    )

    # --------------------------------------------------------
    # Targeted loss.
    # --------------------------------------------------------

    criterion = (
        create_targeted_loss()
    )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    print(
        "Optimizer: Adam"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print()

    # --------------------------------------------------------
    # Best model tracking.
    # --------------------------------------------------------

    best_validation_accuracy = 0.0

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    # --------------------------------------------------------
    # Training loop.
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        print()

        print(
            "=" * 60
        )

        print(
            f"Targeted Fine-Tuning "
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Training.
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
        # Validation.
        # ----------------------------------------------------

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
            )
        )

        # ----------------------------------------------------
        # Results.
        # ----------------------------------------------------

        print()

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.4f} "
            f"({train_accuracy * 100:.2f}%)"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f} "
            f"({validation_accuracy * 100:.2f}%)"
        )

        # ----------------------------------------------------
        # Check for improvement.
        # ----------------------------------------------------

        is_better = False

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            is_better = True

        elif (
            validation_accuracy
            == best_validation_accuracy
            and validation_loss
            < best_validation_loss
        ):

            is_better = True

        # ----------------------------------------------------
        # Save best model.
        # ----------------------------------------------------

        if is_better:

            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = (
                epoch + 1
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                model,
                best_epoch,
                best_validation_loss,
                best_validation_accuracy,
            )

            print()

            print(
                "New best targeted model saved:"
            )

            print(
                f"  {OUTPUT_MODEL_PATH}"
            )

        else:

            print()

            print(
                "No validation improvement."
            )

    # --------------------------------------------------------
    # Restore best model in memory.
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Final results.
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "TARGETED FINE-TUNING COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f} "
        f"({best_validation_accuracy * 100:.2f}%)"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print()

    print(
        "Five classes:"
    )

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        print(
            f"  {index} = {class_name}"
        )

    print()

    print(
        "Model:"
    )

    print(
        f"  {OUTPUT_MODEL_PATH}"
    )

    print()

    print(
        "Targeted classes:"
    )

    print(
        "  Blight"
    )

    print(
        "  LeafSpot"
    )

    print()

    print(
        "Other classes preserved:"
    )

    print(
        "  Healthy"
    )

    print(
        "  Mildew"
    )

    print(
        "  Rust"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "  Evaluate the targeted model on "
        "the untouched five-class test set."
    )

    print()

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()