"""
ai/error_analysis.py

Five-class crop disease error analysis
for AgriculturalQuadcopter.

Classes:

    Blight
    Healthy
    LeafSpot
    Mildew
    Rust

The script:

1. Loads the best fine-tuned MobileNetV3-Small model.
2. Scans the test dataset recursively.
3. Evaluates all test images.
4. Finds incorrect predictions.
5. Creates confusion pairs.
6. Saves misclassified images.
7. Creates a CSV report.
"""

import os
import csv
import shutil
from collections import Counter

import torch
from PIL import Image
from torchvision import transforms

from ai.cnn import create_transfer_learning_model
from ai.training import DEVICE


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.join(
    "trained_models",
    "crop_disease_mobilenet_blspot_targeted.pth"
)

TEST_DIRECTORY = os.path.join(
    "datasets",
    "crop_disease",
    "test",
)

OUTPUT_DIRECTORY = os.path.join(
    "error_analysis",
)

MISCLASSIFIED_DIRECTORY = os.path.join(
    OUTPUT_DIRECTORY,
    "misclassified",
)

CSV_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "misclassifications.csv",
)

IMAGE_SIZE = 224


# ============================================================
# Five-class configuration
# ============================================================

CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


# ============================================================
# Test transformation
# ============================================================

def create_test_transform():

    return transforms.Compose([

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
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
# Find test images
# ============================================================

def find_test_images():

    if not os.path.isdir(
        TEST_DIRECTORY
    ):

        raise FileNotFoundError(
            "Test dataset directory not found:\n"
            f"{TEST_DIRECTORY}"
        )

    samples = []

    # --------------------------------------------------------
    # Recursively search the test directory.
    #
    # We only accept images whose parent directory is one
    # of our five disease classes.
    # --------------------------------------------------------

    for root, directories, files in os.walk(
        TEST_DIRECTORY
    ):

        parent_directory = os.path.basename(
            root
        )

        if parent_directory not in CLASS_NAMES:
            continue

        actual_class = (
            parent_directory
        )

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            image_path = os.path.join(
                root,
                filename,
            )

            samples.append(
                (
                    image_path,
                    actual_class,
                )
            )

    # --------------------------------------------------------
    # Sort for reproducibility
    # --------------------------------------------------------

    samples.sort(
        key=lambda item: item[0].lower()
    )

    return samples


# ============================================================
# Print dataset information
# ============================================================

def inspect_test_dataset(
    samples,
):

    counts = Counter()

    for image_path, class_name in samples:

        counts[class_name] += 1

    print(
        f"Test images: {len(samples)}"
    )

    print()

    print(
        "Test dataset:"
    )

    print("-" * 60)

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<10}: "
            f"{counts[class_name]:5d}"
        )

    print("-" * 60)

    print(
        f"  {'TOTAL':<10}: "
        f"{len(samples):5d}"
    )

    print()

    # --------------------------------------------------------
    # Verify expected five classes
    # --------------------------------------------------------

    missing = []

    for class_name in CLASS_NAMES:

        if counts[class_name] == 0:

            missing.append(
                class_name
            )

    if missing:

        raise ValueError(
            "Missing disease classes in test dataset:\n"
            + "\n".join(
                f"  - {name}"
                for name in missing
            )
        )

    print(
        "Five-class test dataset verified."
    )

    print()


# ============================================================
# Create output directories
# ============================================================

def create_output_directories():

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    os.makedirs(
        MISCLASSIFIED_DIRECTORY,
        exist_ok=True,
    )


# ============================================================
# Load model
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Fine-tuned model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run:\n"
            "python -m ai.finetune"
        )

    print(
        "Loading fine-tuned model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = (
        create_transfer_learning_model()
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Read class names
    # --------------------------------------------------------

    checkpoint_class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )

    # --------------------------------------------------------
    # Verify exactly five classes
    # --------------------------------------------------------

    if len(
        checkpoint_class_names
    ) != 5:

        raise ValueError(
            "Model checkpoint does not contain "
            "exactly five classes.\n"
            f"Found: {checkpoint_class_names}"
        )

    # --------------------------------------------------------
    # Verify order
    # --------------------------------------------------------

    if list(
        checkpoint_class_names
    ) != CLASS_NAMES:

        raise ValueError(
            "Class order mismatch.\n\n"
            f"Expected:\n"
            f"{CLASS_NAMES}\n\n"
            f"Checkpoint:\n"
            f"{checkpoint_class_names}"
        )

    print(
        "Model loaded successfully."
    )

    print()

    print(
        f"Architecture: "
        f"{checkpoint.get('architecture', 'Unknown')}"
    )

    print(
        f"Training type: "
        f"{checkpoint.get('training_type', 'Unknown')}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    validation_accuracy = checkpoint.get(
        "validation_accuracy"
    )

    if validation_accuracy is not None:

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

    print(
        f"Number of classes: "
        f"{checkpoint.get('num_classes', 5)}"
    )

    print()

    return model


# ============================================================
# Save misclassified image
# ============================================================

def save_misclassified_image(
    image_path,
    actual_class,
    predicted_class,
    index,
):

    directory = os.path.join(
        MISCLASSIFIED_DIRECTORY,
        f"{actual_class}_to_{predicted_class}",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    filename = os.path.basename(
        image_path
    )

    destination = os.path.join(
        directory,
        f"{index:05d}_{filename}",
    )

    shutil.copy2(
        image_path,
        destination,
    )


# ============================================================
# Analyze test set
# ============================================================

def analyze(
    model,
    samples,
):

    transform = (
        create_test_transform()
    )

    total = 0

    correct = 0

    errors = []

    confusion_pairs = Counter()

    confusion_matrix = [
        [0 for _ in CLASS_NAMES]
        for _ in CLASS_NAMES
    ]

    print(
        "Running test-set error analysis..."
    )

    print()

    # --------------------------------------------------------
    # Class-to-index mapping
    # --------------------------------------------------------

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(CLASS_NAMES)
    }

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    with torch.no_grad():

        for index, (
            image_path,
            actual_class,
        ) in enumerate(samples):

            # ------------------------------------------------
            # Open image
            # ------------------------------------------------

            try:

                image = Image.open(
                    image_path
                ).convert(
                    "RGB"
                )

            except Exception as error:

                print(
                    f"WARNING: Could not read image:"
                )

                print(
                    f"  {image_path}"
                )

                print(
                    f"  {error}"
                )

                continue

            # ------------------------------------------------
            # Transform
            # ------------------------------------------------

            image_tensor = transform(
                image
            )

            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predicted_index = (
                probabilities
                .argmax(
                    dim=1
                )
                .item()
            )

            predicted_class = (
                CLASS_NAMES[
                    predicted_index
                ]
            )

            confidence = (
                probabilities[
                    0,
                    predicted_index
                ]
                .item()
            )

            actual_index = (
                class_to_index[
                    actual_class
                ]
            )

            # ------------------------------------------------
            # Update confusion matrix
            # ------------------------------------------------

            confusion_matrix[
                actual_index
            ][
                predicted_index
            ] += 1

            total += 1

            # ------------------------------------------------
            # Correct
            # ------------------------------------------------

            if (
                predicted_class
                == actual_class
            ):

                correct += 1

            # ------------------------------------------------
            # Incorrect
            # ------------------------------------------------

            else:

                confusion_pairs[
                    (
                        actual_class,
                        predicted_class,
                    )
                ] += 1

                error_record = {
                    "image_path": image_path,
                    "actual": actual_class,
                    "predicted": predicted_class,
                    "confidence": confidence,
                }

                errors.append(
                    error_record
                )

                save_misclassified_image(
                    image_path,
                    actual_class,
                    predicted_class,
                    index,
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 500
                == 0
            ):

                current_accuracy = (
                    correct / total
                    if total > 0
                    else 0.0
                )

                print(
                    f"  Processed "
                    f"{index + 1:4d}/"
                    f"{len(samples)}"
                    f" | Accuracy: "
                    f"{current_accuracy:.4f}"
                )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        total,
        correct,
        accuracy,
        errors,
        confusion_pairs,
        confusion_matrix,
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    errors,
):

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "actual",
                "predicted",
                "confidence",
            ],
        )

        writer.writeheader()

        for error in errors:

            writer.writerow(
                error
            )


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    matrix,
):

    print(
        "CONFUSION MATRIX"
    )

    print("-" * 80)

    print(
        "Rows = actual"
    )

    print(
        "Columns = predicted"
    )

    print()

    print(
        f"{'':12}"
        + "".join(
            f"{name:>12}"
            for name in CLASS_NAMES
        )
    )

    for index, row in enumerate(
        matrix
    ):

        print(
            f"{CLASS_NAMES[index]:12}"
            + "".join(
                f"{value:12d}"
                for value in row
            )
        )

    print()


# ============================================================
# Print summary
# ============================================================

def print_summary(
    total,
    correct,
    accuracy,
    errors,
    confusion_pairs,
    confusion_matrix,
):

    print()

    print("=" * 60)

    print(
        "ERROR ANALYSIS RESULTS"
    )

    print("=" * 60)

    print()

    print(
        f"Total test images: "
        f"{total}"
    )

    print(
        f"Correct predictions: "
        f"{correct}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(errors)}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    print(
        "CONFUSION PAIRS"
    )

    print("-" * 60)

    if not confusion_pairs:

        print(
            "No misclassifications found."
        )

    else:

        sorted_pairs = sorted(
            confusion_pairs.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for (
            pair,
            count,
        ) in sorted_pairs:

            actual, predicted = pair

            print(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": "
                f"{count:4d}"
            )

    print()

    # --------------------------------------------------------
    # Full matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        confusion_matrix
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "OUTPUT FILES"
    )

    print("-" * 60)

    print(
        f"CSV report:"
    )

    print(
        f"  {CSV_PATH}"
    )

    print()

    print(
        "Misclassified images:"
    )

    print(
        f"  {MISCLASSIFIED_DIRECTORY}"
    )

    print()

    print("=" * 60)

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Five-Class Disease Error Analysis"
    )

    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Find test images
    # --------------------------------------------------------

    samples = find_test_images()

    # --------------------------------------------------------
    # Verify dataset
    # --------------------------------------------------------

    inspect_test_dataset(
        samples
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    (
        total,
        correct,
        accuracy,
        errors,
        confusion_pairs,
        confusion_matrix,
    ) = analyze(
        model,
        samples,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_csv(
        errors
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_summary(
        total,
        correct,
        accuracy,
        errors,
        confusion_pairs,
        confusion_matrix,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()