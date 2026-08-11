"""
ai/confidence_threshold.py

AgriculturalQuadcopter
Five-Class Confidence Threshold Analysis

Purpose
-------
Evaluate model confidence on the untouched five-class test set.

The experiment answers:

    "When the model is not sufficiently confident,
     should the system request further inspection?"

Five classes:

    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Important
---------
This script DOES NOT use torchvision.datasets.ImageFolder.

The dataset is loaded manually so that extra directories such as:

    crop_disease

are ignored.

Current model:
    trained_models/crop_disease_mobilenet_blspot_targeted.pth
"""

import csv
import os
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from ai.cnn import create_transfer_learning_model
from ai.training import DEVICE


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "crop_disease"
)

TEST_DIRECTORY = (
    DATASET_ROOT
    / "test"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "trained_models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "crop_disease_mobilenet_blspot_targeted.pth"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "confidence_analysis"
)

CSV_OUTPUT = (
    OUTPUT_DIRECTORY
    / "confidence_predictions.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIRECTORY
    / "confidence_summary.txt"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

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


CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
    in enumerate(EXPECTED_CLASSES)
}


# ============================================================
# Confidence thresholds
# ============================================================

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
]


# ============================================================
# Image extensions
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Transform
# ============================================================

def create_test_transform():

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
# Dataset
# ============================================================

class FiveClassTestDataset(Dataset):
    """
    Manually loaded five-class dataset.

    This avoids ImageFolder interpreting unwanted directories
    as additional classes.
    """

    def __init__(
        self,
        samples,
        transform=None,
    ):

        self.samples = samples

        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                f"Unable to open image:\n"
                f"{image_path}\n\n"
                f"Error: {exc}"
            ) from exc

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# Locate five-class directory
# ============================================================

def resolve_test_directory():

    search_root = Path(
        TEST_DIRECTORY
    ).resolve()

    print(
        "Test dataset search root:"
    )

    print(
        search_root
    )

    print()

    if not search_root.exists():

        raise FileNotFoundError(
            "Test dataset directory does not exist:\n"
            f"{search_root}"
        )

    if not search_root.is_dir():

        raise RuntimeError(
            "Test dataset path is not a directory:\n"
            f"{search_root}"
        )

    # --------------------------------------------------------
    # First try the directory itself.
    # --------------------------------------------------------

    direct_matches = []

    for class_name in EXPECTED_CLASSES:

        candidate = (
            search_root
            / class_name
        )

        if candidate.is_dir():

            direct_matches.append(
                class_name
            )

    if len(direct_matches) == len(
        EXPECTED_CLASSES
    ):

        return search_root

    # --------------------------------------------------------
    # Search one level below.
    #
    # This handles cases where the actual dataset is:
    #
    # test/
    #     crop_disease/
    #         Blight/
    #         Healthy/
    #         ...
    # --------------------------------------------------------

    for child in sorted(
        search_root.iterdir()
    ):

        if not child.is_dir():

            continue

        matches = []

        for class_name in EXPECTED_CLASSES:

            candidate = (
                child
                / class_name
            )

            if candidate.is_dir():

                matches.append(
                    class_name
                )

        if len(matches) == len(
            EXPECTED_CLASSES
        ):

            return child

    # --------------------------------------------------------
    # Nothing valid was found.
    # --------------------------------------------------------

    found_directories = [
        item.name
        for item in search_root.iterdir()
        if item.is_dir()
    ]

    raise RuntimeError(
        "Could not locate the five-class test dataset.\n\n"
        f"Search root:\n"
        f"  {search_root}\n\n"
        "Required classes:\n"
        f"  {EXPECTED_CLASSES}\n\n"
        "Directories found:\n"
        f"  {found_directories}"
    )


# ============================================================
# Load test dataset
# ============================================================

def load_test_dataset():

    resolved_directory = (
        resolve_test_directory()
    )

    print(
        "Resolved five-class test directory:"
    )

    print(
        resolved_directory
    )

    print()

    class_directories = {}

    # --------------------------------------------------------
    # Verify classes.
    # --------------------------------------------------------

    for class_name in EXPECTED_CLASSES:

        class_directory = (
            resolved_directory
            / class_name
        )

        if not class_directory.is_dir():

            raise RuntimeError(
                "Missing required class directory:\n"
                f"{class_directory}"
            )

        class_directories[
            class_name
        ] = class_directory

    print(
        "Five-class test dataset verified."
    )

    print()

    # --------------------------------------------------------
    # Build samples manually.
    # --------------------------------------------------------

    samples = []

    class_counts = {}

    for class_name in EXPECTED_CLASSES:

        class_directory = (
            class_directories[
                class_name
            ]
        )

        label = CLASS_TO_INDEX[
            class_name
        ]

        count = 0

        for file_path in sorted(
            class_directory.rglob("*")
        ):

            if not file_path.is_file():

                continue

            if (
                file_path.suffix.lower()
                not in VALID_EXTENSIONS
            ):

                continue

            samples.append(
                (
                    str(file_path),
                    label,
                )
            )

            count += 1

        class_counts[
            class_name
        ] = count

    # --------------------------------------------------------
    # Verify every class has images.
    # --------------------------------------------------------

    for class_name in EXPECTED_CLASSES:

        if class_counts[
            class_name
        ] == 0:

            raise RuntimeError(
                "No valid images found for class:\n"
                f"{class_name}\n\n"
                f"Directory:\n"
                f"{class_directories[class_name]}"
            )

    # --------------------------------------------------------
    # Print distribution.
    # --------------------------------------------------------

    print(
        "Test dataset distribution:"
    )

    print(
        "------------------------------------------------------------"
    )

    for class_name in EXPECTED_CLASSES:

        print(
            f"  {class_name:<10}: "
            f"{class_counts[class_name]:6d}"
        )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"  {'TOTAL':<10}: "
        f"{len(samples):6d}"
    )

    print()

    dataset = FiveClassTestDataset(
        samples=samples,
        transform=create_test_transform(),
    )

    return dataset


# ============================================================
# Create DataLoader
# ============================================================

def create_loader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


# ============================================================
# Load model
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Targeted model was not found:\n"
            f"{MODEL_PATH}\n\n"
            "Expected model:\n"
            "crop_disease_mobilenet_blspot_targeted.pth"
        )

    print(
        "Loading targeted model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Verify checkpoint classes.
    # --------------------------------------------------------

    checkpoint_classes = checkpoint.get(
        "class_names",
        EXPECTED_CLASSES,
    )

    if list(
        checkpoint_classes
    ) != EXPECTED_CLASSES:

        raise RuntimeError(
            "Model class configuration does not match "
            "the five-class project.\n\n"
            f"Expected:\n"
            f"  {EXPECTED_CLASSES}\n\n"
            f"Model contains:\n"
            f"  {checkpoint_classes}"
        )

    model = create_transfer_learning_model()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

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
        f"Validation accuracy: "
        f"{checkpoint.get('validation_accuracy', 0.0):.4f}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    print()

    return model, checkpoint


# ============================================================
# Run predictions
# ============================================================

def collect_predictions(
    model,
    loader,
):

    print(
        "Running confidence analysis..."
    )

    print()

    records = []

    total = len(
        loader.dataset
    )

    processed = 0

    correct = 0

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

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, predictions = (
                probabilities.max(
                    dim=1
                )
            )

            for index in range(
                labels.size(0)
            ):

                actual_index = (
                    labels[index]
                    .item()
                )

                predicted_index = (
                    predictions[index]
                    .item()
                )

                confidence_value = (
                    confidence[index]
                    .item()
                )

                probability_vector = (
                    probabilities[index]
                    .cpu()
                    .tolist()
                )

                is_correct = (
                    actual_index
                    == predicted_index
                )

                if is_correct:

                    correct += 1

                records.append({

                    "image_path":
                        loader.dataset.samples[
                            processed
                        ][0],

                    "actual":
                        EXPECTED_CLASSES[
                            actual_index
                        ],

                    "predicted":
                        EXPECTED_CLASSES[
                            predicted_index
                        ],

                    "confidence":
                        confidence_value,

                    "correct":
                        is_correct,

                    "blight_probability":
                        probability_vector[0],

                    "healthy_probability":
                        probability_vector[1],

                    "leafspot_probability":
                        probability_vector[2],

                    "mildew_probability":
                        probability_vector[3],

                    "rust_probability":
                        probability_vector[4],
                })

                processed += 1

            if (
                processed % 500 == 0
                or processed == total
            ):

                running_accuracy = (
                    correct / processed
                )

                print(
                    f"  Processed "
                    f"{processed:4d}/{total}"
                    f" | Accuracy: "
                    f"{running_accuracy:.4f}"
                )

    print()

    return records


# ============================================================
# Overall metrics
# ============================================================

def calculate_overall_metrics(
    records
):

    total = len(records)

    correct = sum(
        1
        for record in records
        if record["correct"]
    )

    incorrect = (
        total - correct
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    confidences = [
        record["confidence"]
        for record in records
    ]

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    correct_confidences = [
        record["confidence"]
        for record in records
        if record["correct"]
    ]

    incorrect_confidences = [
        record["confidence"]
        for record in records
        if not record["correct"]
    ]

    average_correct_confidence = (
        sum(correct_confidences)
        / len(correct_confidences)
        if correct_confidences
        else 0.0
    )

    average_incorrect_confidence = (
        sum(incorrect_confidences)
        / len(incorrect_confidences)
        if incorrect_confidences
        else 0.0
    )

    maximum_incorrect_confidence = (
        max(incorrect_confidences)
        if incorrect_confidences
        else 0.0
    )

    minimum_correct_confidence = (
        min(correct_confidences)
        if correct_confidences
        else 0.0
    )

    return {

        "total": total,

        "correct": correct,

        "incorrect": incorrect,

        "accuracy": accuracy,

        "average_confidence":
            average_confidence,

        "average_correct_confidence":
            average_correct_confidence,

        "average_incorrect_confidence":
            average_incorrect_confidence,

        "maximum_incorrect_confidence":
            maximum_incorrect_confidence,

        "minimum_correct_confidence":
            minimum_correct_confidence,
    }


# ============================================================
# Threshold analysis
# ============================================================

def analyze_thresholds(
    records
):

    results = []

    total = len(records)

    for threshold in THRESHOLDS:

        accepted = [
            record
            for record in records
            if record["confidence"]
            >= threshold
        ]

        rejected = [
            record
            for record in records
            if record["confidence"]
            < threshold
        ]

        accepted_count = len(
            accepted
        )

        rejected_count = len(
            rejected
        )

        accepted_correct = sum(
            1
            for record in accepted
            if record["correct"]
        )

        accepted_incorrect = (
            accepted_count
            - accepted_correct
        )

        if accepted_count > 0:

            accepted_accuracy = (
                accepted_correct
                / accepted_count
            )

        else:

            accepted_accuracy = 0.0

        coverage = (
            accepted_count / total
            if total > 0
            else 0.0
        )

        rejection_rate = (
            rejected_count / total
            if total > 0
            else 0.0
        )

        results.append({

            "threshold":
                threshold,

            "accepted":
                accepted_count,

            "rejected":
                rejected_count,

            "coverage":
                coverage,

            "rejection_rate":
                rejection_rate,

            "accepted_correct":
                accepted_correct,

            "accepted_incorrect":
                accepted_incorrect,

            "accepted_accuracy":
                accepted_accuracy,
        })

    return results


# ============================================================
# Per-class confidence analysis
# ============================================================

def analyze_classes(records):

    results = {}

    for class_name in EXPECTED_CLASSES:

        class_records = [
            record
            for record in records
            if record["actual"]
            == class_name
        ]

        correct_records = [
            record
            for record in class_records
            if record["correct"]
        ]

        incorrect_records = [
            record
            for record in class_records
            if not record["correct"]
        ]

        confidences = [
            record["confidence"]
            for record in class_records
        ]

        correct_confidences = [
            record["confidence"]
            for record in correct_records
        ]

        incorrect_confidences = [
            record["confidence"]
            for record in incorrect_records
        ]

        count = len(
            class_records
        )

        correct_count = len(
            correct_records
        )

        results[class_name] = {

            "total":
                count,

            "correct":
                correct_count,

            "incorrect":
                len(
                    incorrect_records
                ),

            "accuracy":
                (
                    correct_count / count
                    if count > 0
                    else 0.0
                ),

            "average_confidence":
                (
                    sum(confidences)
                    / len(confidences)
                    if confidences
                    else 0.0
                ),

            "correct_confidence":
                (
                    sum(correct_confidences)
                    / len(correct_confidences)
                    if correct_confidences
                    else 0.0
                ),

            "incorrect_confidence":
                (
                    sum(incorrect_confidences)
                    / len(incorrect_confidences)
                    if incorrect_confidences
                    else 0.0
                ),
        }

    return results


# ============================================================
# Confusion pairs
# ============================================================

def calculate_confusion_pairs(
    records
):

    pairs = Counter()

    for record in records:

        if not record["correct"]:

            pair = (
                record["actual"],
                record["predicted"],
            )

            pairs[pair] += 1

    return pairs


# ============================================================
# Save CSV
# ============================================================

def save_csv(records):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [

        "image_path",

        "actual",

        "predicted",

        "confidence",

        "correct",

        "blight_probability",

        "healthy_probability",

        "leafspot_probability",

        "mildew_probability",

        "rust_probability",
    ]

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    print(
        "Prediction CSV saved:"
    )

    print(
        f"  {CSV_OUTPUT}"
    )

    print()


# ============================================================
# Print report
# ============================================================

def print_report(
    records,
    overall,
    threshold_results,
    class_results,
    confusion_pairs,
):

    print()
    print("=" * 60)
    print(
        "CONFIDENCE THRESHOLD ANALYSIS"
    )
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print(
        "OVERALL PERFORMANCE"
    )

    print(
        "-" * 60
    )

    print(
        f"Total test images: "
        f"{overall['total']}"
    )

    print(
        f"Correct predictions: "
        f"{overall['correct']}"
    )

    print(
        f"Incorrect predictions: "
        f"{overall['incorrect']}"
    )

    print(
        f"Test accuracy: "
        f"{overall['accuracy']:.4f} "
        f"({overall['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Average confidence: "
        f"{overall['average_confidence']:.4f}"
    )

    print(
        f"Average confidence "
        f"(correct): "
        f"{overall['average_correct_confidence']:.4f}"
    )

    print(
        f"Average confidence "
        f"(incorrect): "
        f"{overall['average_incorrect_confidence']:.4f}"
    )

    print(
        f"Minimum confidence "
        f"(correct): "
        f"{overall['minimum_correct_confidence']:.4f}"
    )

    print(
        f"Maximum confidence "
        f"(incorrect): "
        f"{overall['maximum_incorrect_confidence']:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Per class
    # --------------------------------------------------------

    print(
        "PER-CLASS CONFIDENCE"
    )

    print(
        "-" * 60
    )

    print(
        f"{'Class':<12}"
        f"{'Images':>8}"
        f"{'Accuracy':>12}"
        f"{'Avg Conf.':>12}"
        f"{'Wrong Conf.':>12}"
    )

    for class_name in EXPECTED_CLASSES:

        result = class_results[
            class_name
        ]

        print(
            f"{class_name:<12}"
            f"{result['total']:>8}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['average_confidence']:>12.4f}"
            f"{result['incorrect_confidence']:>12.4f}"
        )

    print()

    # --------------------------------------------------------
    # Threshold table
    # --------------------------------------------------------

    print(
        "CONFIDENCE THRESHOLD RESULTS"
    )

    print(
        "-" * 60
    )

    print(
        f"{'Threshold':>10}"
        f"{'Accepted':>10}"
        f"{'Rejected':>10}"
        f"{'Coverage':>11}"
        f"{'Accuracy':>11}"
    )

    for result in threshold_results:

        print(
            f"{result['threshold']:>10.2f}"
            f"{result['accepted']:>10}"
            f"{result['rejected']:>10}"
            f"{result['coverage'] * 100:>10.2f}%"
            f"{result['accepted_accuracy'] * 100:>10.2f}%"
        )

    print()

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    print(
        "CONFUSION PAIRS"
    )

    print(
        "-" * 60
    )

    if not confusion_pairs:

        print(
            "No incorrect predictions."
        )

    else:

        for (
            actual,
            predicted,
        ), count in confusion_pairs.most_common():

            print(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": {count:4d}"
            )

    print()


# ============================================================
# Save summary
# ============================================================

def save_summary(
    overall,
    threshold_results,
    class_results,
    confusion_pairs,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "AgriculturalQuadcopter\n"
        )

        file.write(
            "Five-Class Confidence Threshold Analysis\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "CLASSES\n"
        )

        for index, class_name in enumerate(
            EXPECTED_CLASSES
        ):

            file.write(
                f"{index} = {class_name}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "OVERALL PERFORMANCE\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Total: {overall['total']}\n"
        )

        file.write(
            f"Correct: {overall['correct']}\n"
        )

        file.write(
            f"Incorrect: {overall['incorrect']}\n"
        )

        file.write(
            f"Accuracy: "
            f"{overall['accuracy']:.6f}\n"
        )

        file.write(
            f"Average confidence: "
            f"{overall['average_confidence']:.6f}\n"
        )

        file.write(
            f"Average correct confidence: "
            f"{overall['average_correct_confidence']:.6f}\n"
        )

        file.write(
            f"Average incorrect confidence: "
            f"{overall['average_incorrect_confidence']:.6f}\n"
        )

        file.write(
            f"Minimum correct confidence: "
            f"{overall['minimum_correct_confidence']:.6f}\n"
        )

        file.write(
            f"Maximum incorrect confidence: "
            f"{overall['maximum_incorrect_confidence']:.6f}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "PER-CLASS CONFIDENCE\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for class_name in EXPECTED_CLASSES:

            result = class_results[
                class_name
            ]

            file.write(
                f"\n{class_name}\n"
            )

            file.write(
                f"  Total: "
                f"{result['total']}\n"
            )

            file.write(
                f"  Correct: "
                f"{result['correct']}\n"
            )

            file.write(
                f"  Incorrect: "
                f"{result['incorrect']}\n"
            )

            file.write(
                f"  Accuracy: "
                f"{result['accuracy']:.6f}\n"
            )

            file.write(
                f"  Average confidence: "
                f"{result['average_confidence']:.6f}\n"
            )

            file.write(
                f"  Average incorrect confidence: "
                f"{result['incorrect_confidence']:.6f}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "THRESHOLD ANALYSIS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for result in threshold_results:

            file.write(
                f"\nThreshold: "
                f"{result['threshold']:.2f}\n"
            )

            file.write(
                f"  Accepted: "
                f"{result['accepted']}\n"
            )

            file.write(
                f"  Rejected: "
                f"{result['rejected']}\n"
            )

            file.write(
                f"  Coverage: "
                f"{result['coverage']:.6f}\n"
            )

            file.write(
                f"  Rejection rate: "
                f"{result['rejection_rate']:.6f}\n"
            )

            file.write(
                f"  Accepted correct: "
                f"{result['accepted_correct']}\n"
            )

            file.write(
                f"  Accepted incorrect: "
                f"{result['accepted_incorrect']}\n"
            )

            file.write(
                f"  Accepted accuracy: "
                f"{result['accepted_accuracy']:.6f}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "CONFUSION PAIRS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for (
            actual,
            predicted,
        ), count in confusion_pairs.most_common():

            file.write(
                f"{actual} -> {predicted}: "
                f"{count}\n"
            )

    print(
        "Summary saved:"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Five-Class Confidence Threshold Analysis"
    )

    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print()

    print(
        "Five-class configuration:"
    )

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        print(
            f"{index} = {class_name}"
        )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_test_dataset()

    loader = create_loader(
        dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = (
        load_model()
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    records = collect_predictions(
        model,
        loader,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    overall = (
        calculate_overall_metrics(
            records
        )
    )

    threshold_results = (
        analyze_thresholds(
            records
        )
    )

    class_results = (
        analyze_classes(
            records
        )
    )

    confusion_pairs = (
        calculate_confusion_pairs(
            records
        )
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        records,
        overall,
        threshold_results,
        class_results,
        confusion_pairs,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_csv(
        records
    )

    save_summary(
        overall,
        threshold_results,
        class_results,
        confusion_pairs,
    )

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "CONFIDENCE ANALYSIS COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Important:"
    )

    print(
        "The threshold does NOT change the classifier."
    )

    print(
        "It determines when a prediction should be"
    )

    print(
        "accepted automatically versus sent for"
    )

    print(
        "additional inspection."
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {CSV_OUTPUT}"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print()

    print(
        "Next lesson:"
    )

    print(
        "Select an operating confidence threshold"
    )

    print(
        "for the agricultural quadcopter."
    )

    print()

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()