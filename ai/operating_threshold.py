"""
ai/operating_threshold.py

AgriculturalQuadcopter
Operating Confidence Threshold Selection

Purpose
-------
Select an operating confidence threshold for the
five-class crop disease classifier.

The threshold is selected ONLY using the validation set.

The untouched test set must NOT be used to choose the threshold.

Classes:
    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Operating principle
-------------------
If model confidence >= operating threshold:
    ACCEPT prediction automatically

If model confidence < operating threshold:
    SEND prediction for additional inspection

This module:
    1. Finds the validation dataset robustly.
    2. Ignores accidental extra directories.
    3. Loads the targeted Blight/LeafSpot model.
    4. Evaluates confidence on validation images.
    5. Tests multiple confidence thresholds.
    6. Selects the highest-coverage threshold that reaches
       the required accepted accuracy.
    7. Saves the operating threshold.
    8. Produces a deployment-oriented report.

IMPORTANT
---------
The test dataset is intentionally NOT used here.

The threshold selected here should later be evaluated once
on the untouched test set.
"""

import os
import json
import copy

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ai.cnn import create_transfer_learning_model
from ai.training import DEVICE


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    )
)


DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "crop_disease",
)


VALIDATION_SEARCH_ROOT = os.path.join(
    DATASET_ROOT,
    "validation",
)


MODEL_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "trained_models",
)


MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "crop_disease_mobilenet_blspot_targeted.pth",
)


OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "confidence_analysis",
)


THRESHOLD_JSON_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "operating_threshold.json",
)


THRESHOLD_REPORT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "operating_threshold_report.txt",
)


IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# Operating policy
# ============================================================

# We want the highest possible coverage while requiring
# accepted predictions to reach at least this accuracy.

TARGET_ACCEPTED_ACCURACY = 0.995


# Thresholds to evaluate.

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
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
]


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


SUPPORTED_EXTENSIONS = {
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
# Image transformation
# ============================================================

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
# Dataset directory discovery
# ============================================================

def directory_contains_required_classes(
    directory,
):

    if not os.path.isdir(directory):
        return False

    try:
        entries = os.listdir(directory)
    except OSError:
        return False

    folders = {
        name
        for name in entries
        if os.path.isdir(
            os.path.join(directory, name)
        )
    }

    return all(
        class_name in folders
        for class_name in EXPECTED_CLASSES
    )


def find_five_class_directory(
    search_root,
):

    search_root = os.path.abspath(
        search_root
    )

    print(
        "Searching recursively for the "
        "five-class validation dataset..."
    )

    print()

    print(
        "Search root:"
    )

    print(
        f"  {search_root}"
    )

    print()

    if not os.path.isdir(search_root):

        raise FileNotFoundError(
            "Validation search directory does not exist:\n"
            f"{search_root}\n\n"
            "Expected structure:\n"
            "validation/\n"
            "    Blight/\n"
            "    Healthy/\n"
            "    LeafSpot/\n"
            "    Mildew/\n"
            "    Rust/"
        )

    candidates = []

    for current_root, directories, _files in os.walk(
        search_root
    ):

        directory_set = set(
            directories
        )

        if all(
            class_name in directory_set
            for class_name in EXPECTED_CLASSES
        ):

            candidates.append(
                current_root
            )

    if not candidates:

        raise RuntimeError(
            "Could not find a directory containing "
            "all five required classes.\n\n"
            f"Search root:\n{search_root}\n\n"
            "Required classes:\n"
            f"{EXPECTED_CLASSES}\n\n"
            "Expected structure:\n"
            "validation/\n"
            "    Blight/\n"
            "    Healthy/\n"
            "    LeafSpot/\n"
            "    Mildew/\n"
            "    Rust/"
        )

    # Prefer the shallowest valid directory.
    candidates.sort(
        key=lambda path: (
            path.count(os.sep),
            len(path),
        )
    )

    selected = candidates[0]

    print(
        "Found valid five-class dataset:"
    )

    print(
        f"  {selected}"
    )

    print()

    return selected


# ============================================================
# Explicit five-class dataset
# ============================================================

class FiveClassImageDataset(
    Dataset
):

    """
    Explicit five-class image dataset.

    Unlike torchvision.datasets.ImageFolder, this class
    deliberately ignores extra directories such as:

        crop_disease

    Only the five required class directories are scanned.
    """

    def __init__(
        self,
        root,
        transform=None,
    ):

        self.root = os.path.abspath(
            root
        )

        self.transform = transform

        self.samples = []

        self.class_names = list(
            EXPECTED_CLASSES
        )

        self.class_to_idx = dict(
            CLASS_TO_INDEX
        )

        self._build_samples()

    def _build_samples(self):

        if not os.path.isdir(
            self.root
        ):

            raise FileNotFoundError(
                "Dataset directory does not exist:\n"
                f"{self.root}"
            )

        actual_directories = set(
            name
            for name in os.listdir(
                self.root
            )
            if os.path.isdir(
                os.path.join(
                    self.root,
                    name,
                )
            )
        )

        missing_classes = [
            class_name
            for class_name in EXPECTED_CLASSES
            if class_name not in actual_directories
        ]

        if missing_classes:

            raise RuntimeError(
                "Validation dataset is missing required "
                "class directories.\n\n"
                f"Dataset:\n{self.root}\n\n"
                f"Missing classes:\n{missing_classes}\n\n"
                f"Available directories:\n"
                f"{sorted(actual_directories)}"
            )

        extra_directories = sorted(
            actual_directories
            - set(EXPECTED_CLASSES)
        )

        if extra_directories:

            print(
                "Ignoring extra directories:"
            )

            for directory in extra_directories:

                print(
                    f"  {directory}"
                )

            print()

        for class_name in EXPECTED_CLASSES:

            class_directory = os.path.join(
                self.root,
                class_name,
            )

            class_index = self.class_to_idx[
                class_name
            ]

            for current_root, _directories, files in os.walk(
                class_directory
            ):

                for filename in sorted(files):

                    extension = os.path.splitext(
                        filename
                    )[1].lower()

                    if extension not in SUPPORTED_EXTENSIONS:
                        continue

                    image_path = os.path.join(
                        current_root,
                        filename,
                    )

                    self.samples.append(
                        (
                            image_path,
                            class_index,
                        )
                    )

        if not self.samples:

            raise RuntimeError(
                "No valid images were found in the "
                "five required classes.\n\n"
                f"Dataset:\n{self.root}"
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        image_path, label = self.samples[
            index
        ]

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not read image:\n"
                f"{image_path}\n\n"
                f"Error: {exc}"
            ) from exc

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            label,
        )


# ============================================================
# Load validation dataset
# ============================================================

def load_validation_dataset():

    print(
        "Validation dataset search root:"
    )

    print(
        f"  {VALIDATION_SEARCH_ROOT}"
    )

    print()

    resolved_directory = find_five_class_directory(
        VALIDATION_SEARCH_ROOT
    )

    print(
        "Resolved five-class validation directory:"
    )

    print(
        f"  {resolved_directory}"
    )

    print()

    dataset = FiveClassImageDataset(
        root=resolved_directory,
        transform=create_validation_transform(),
    )

    print(
        "Five-class validation dataset verified."
    )

    print()

    print(
        "## Validation dataset distribution:"
    )

    print()

    class_counts = {
        class_name: 0
        for class_name in EXPECTED_CLASSES
    }

    for _path, label in dataset.samples:

        class_name = EXPECTED_CLASSES[
            label
        ]

        class_counts[
            class_name
        ] += 1

    total = len(
        dataset
    )

    print(
        f"TOTAL     : {total:6d}"
    )

    for class_name in EXPECTED_CLASSES:

        print(
            f"{class_name:<10}: "
            f"{class_counts[class_name]:6d}"
        )

    print()

    return dataset


# ============================================================
# DataLoader
# ============================================================

def create_loader(
    dataset,
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


# ============================================================
# Model loading
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Targeted model was not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run the targeted Blight/LeafSpot "
            "fine-tuning lesson first."
        )

    print(
        "Loading targeted model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = create_transfer_learning_model()

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        raise RuntimeError(
            "The checkpoint does not contain "
            "'model_state_dict'.\n\n"
            f"Model:\n{MODEL_PATH}"
        )

    model.load_state_dict(
        state_dict
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
        "Architecture: "
        f"{checkpoint.get('architecture', 'Unknown')}"
    )

    print(
        "Training type: "
        f"{checkpoint.get('training_type', 'Unknown')}"
    )

    print(
        "Validation accuracy: "
        f"{checkpoint.get('validation_accuracy', 0.0):.4f}"
    )

    print(
        "Best epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    print()

    return model, checkpoint


# ============================================================
# Confidence analysis
# ============================================================

def collect_validation_predictions(
    model,
    loader,
):

    all_labels = []

    all_predictions = []

    all_confidences = []

    total_samples = len(
        loader.dataset
    )

    processed = 0

    print(
        "Running validation confidence analysis..."
    )

    print()

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

            confidences, predictions = (
                probabilities.max(
                    dim=1
                )
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_confidences.extend(
                confidences.cpu().tolist()
            )

            processed += labels.size(0)

            if (
                processed % 1000 == 0
                or processed == total_samples
            ):

                correct = sum(
                    int(
                        prediction == label
                    )
                    for prediction, label
                    in zip(
                        all_predictions,
                        all_labels,
                    )
                )

                accuracy = (
                    correct
                    / processed
                )

                print(
                    f"Processed "
                    f"{processed}/{total_samples}"
                    f" | Accuracy: "
                    f"{accuracy:.4f}"
                )

    return (
        all_labels,
        all_predictions,
        all_confidences,
    )


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_threshold(
    labels,
    predictions,
    confidences,
    threshold,
):

    total = len(
        labels
    )

    accepted_indices = [
        index
        for index, confidence
        in enumerate(confidences)
        if confidence >= threshold
    ]

    rejected = (
        total
        - len(accepted_indices)
    )

    accepted = len(
        accepted_indices
    )

    if accepted == 0:

        accepted_accuracy = 0.0

    else:

        correct = sum(
            int(
                predictions[index]
                == labels[index]
            )
            for index in accepted_indices
        )

        accepted_accuracy = (
            correct
            / accepted
        )

    coverage = (
        accepted
        / total
        if total > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "accepted": accepted,
        "rejected": rejected,
        "coverage": coverage,
        "accuracy": accepted_accuracy,
    }


def evaluate_all_thresholds(
    labels,
    predictions,
    confidences,
):

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            labels,
            predictions,
            confidences,
            threshold,
        )

        results.append(
            result
        )

    return results


# ============================================================
# Select operating threshold
# ============================================================

def select_operating_threshold(
    results,
):

    qualified = [
        result
        for result in results
        if result["accuracy"]
        >= TARGET_ACCEPTED_ACCURACY
    ]

    if qualified:

        # Primary objective:
        # maximum coverage.
        #
        # Secondary objective:
        # highest accepted accuracy.

        selected = max(
            qualified,
            key=lambda result: (
                result["coverage"],
                result["accuracy"],
                -result["threshold"],
            ),
        )

        selection_reason = (
            "Selected the lowest practical threshold "
            "that achieves the required accepted "
            "accuracy while maximizing coverage."
        )

        return (
            selected,
            selection_reason,
        )

    # No threshold reached the desired target.
    #
    # Fall back to highest accepted accuracy,
    # then highest coverage.

    selected = max(
        results,
        key=lambda result: (
            result["accuracy"],
            result["coverage"],
            -result["threshold"],
        ),
    )

    selection_reason = (
        "WARNING: No tested threshold reached the "
        "required accepted accuracy. Selected the "
        "threshold with the highest accepted accuracy, "
        "then highest coverage."
    )

    return (
        selected,
        selection_reason,
    )


# ============================================================
# Overall validation performance
# ============================================================

def calculate_overall_performance(
    labels,
    predictions,
    confidences,
):

    total = len(
        labels
    )

    correct = sum(
        int(
            prediction == label
        )
        for prediction, label
        in zip(
            predictions,
            labels,
        )
    )

    incorrect = (
        total
        - correct
    )

    accuracy = (
        correct
        / total
        if total > 0
        else 0.0
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    correct_confidences = [
        confidence
        for confidence, prediction, label
        in zip(
            confidences,
            predictions,
            labels,
        )
        if prediction == label
    ]

    incorrect_confidences = [
        confidence
        for confidence, prediction, label
        in zip(
            confidences,
            predictions,
            labels,
        )
        if prediction != label
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

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "average_confidence": average_confidence,
        "average_correct_confidence":
            average_correct_confidence,
        "average_incorrect_confidence":
            average_incorrect_confidence,
    }


# ============================================================
# Per-class confidence
# ============================================================

def calculate_per_class_statistics(
    labels,
    predictions,
    confidences,
):

    statistics = {}

    for class_index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        indices = [
            index
            for index, label
            in enumerate(labels)
            if label == class_index
        ]

        total = len(
            indices
        )

        correct_indices = [
            index
            for index in indices
            if predictions[index]
            == labels[index]
        ]

        incorrect_indices = [
            index
            for index in indices
            if predictions[index]
            != labels[index]
        ]

        correct = len(
            correct_indices
        )

        accuracy = (
            correct
            / total
            if total > 0
            else 0.0
        )

        class_confidences = [
            confidences[index]
            for index in indices
        ]

        wrong_confidences = [
            confidences[index]
            for index in incorrect_indices
        ]

        average_confidence = (
            sum(class_confidences)
            / len(class_confidences)
            if class_confidences
            else 0.0
        )

        average_wrong_confidence = (
            sum(wrong_confidences)
            / len(wrong_confidences)
            if wrong_confidences
            else 0.0
        )

        statistics[
            class_name
        ] = {
            "images": total,
            "accuracy": accuracy,
            "average_confidence":
                average_confidence,
            "average_wrong_confidence":
                average_wrong_confidence,
        }

    return statistics


# ============================================================
# Confusion pairs
# ============================================================

def calculate_confusion_pairs(
    labels,
    predictions,
):

    pairs = {}

    for actual, predicted in zip(
        labels,
        predictions,
    ):

        if actual == predicted:
            continue

        actual_name = EXPECTED_CLASSES[
            actual
        ]

        predicted_name = EXPECTED_CLASSES[
            predicted
        ]

        key = (
            actual_name,
            predicted_name,
        )

        pairs[key] = (
            pairs.get(key, 0)
            + 1
        )

    return dict(
        sorted(
            pairs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


# ============================================================
# Print threshold table
# ============================================================

def print_threshold_results(
    results,
):

    print(
        "## CONFIDENCE THRESHOLD RESULTS"
    )

    print(
        "-" * 80
    )

    print(
        "Threshold  Accepted  Rejected   "
        "Coverage   Accepted Accuracy"
    )

    for result in results:

        print(
            f"{result['threshold']:<10.3f}"
            f"{result['accepted']:>9}"
            f"{result['rejected']:>10}"
            f"{result['coverage'] * 100:>10.2f}%"
            f"{result['accuracy'] * 100:>18.2f}%"
        )

    print()


# ============================================================
# Save JSON configuration
# ============================================================

def save_threshold_json(
    selected,
    selection_reason,
    overall,
    per_class,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    data = {

        "operating_threshold":
            selected["threshold"],

        "target_accepted_accuracy":
            TARGET_ACCEPTED_ACCURACY,

        "accepted_accuracy":
            selected["accuracy"],

        "coverage":
            selected["coverage"],

        "accepted_images":
            selected["accepted"],

        "rejected_images":
            selected["rejected"],

        "selection_reason":
            selection_reason,

        "classes":
            EXPECTED_CLASSES,

        "model":
            os.path.basename(
                MODEL_PATH
            ),

        "validation_total":
            overall["total"],

        "validation_accuracy":
            overall["accuracy"],

        "validation_average_confidence":
            overall["average_confidence"],

        "per_class":
            per_class,

        "policy": {

            "accepted":
                "confidence >= operating threshold",

            "rejected":
                "confidence < operating threshold",

            "rejected_action":
                "additional inspection",
        },
    }

    with open(
        THRESHOLD_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# Save human-readable report
# ============================================================

def save_report(
    selected,
    selection_reason,
    overall,
    per_class,
    threshold_results,
    confusion_pairs,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    lines = []

    lines.append(
        "AgriculturalQuadcopter"
    )

    lines.append(
        "Operating Confidence Threshold Report"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "MODEL"
    )

    lines.append(
        f"Model: {MODEL_PATH}"
    )

    lines.append("")

    lines.append(
        "CLASSES"
    )

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        lines.append(
            f"{index} = {class_name}"
        )

    lines.append("")

    lines.append(
        "VALIDATION PERFORMANCE"
    )

    lines.append(
        f"Total images: {overall['total']}"
    )

    lines.append(
        f"Correct: {overall['correct']}"
    )

    lines.append(
        f"Incorrect: {overall['incorrect']}"
    )

    lines.append(
        f"Accuracy: "
        f"{overall['accuracy'] * 100:.4f}%"
    )

    lines.append(
        f"Average confidence: "
        f"{overall['average_confidence']:.4f}"
    )

    lines.append("")

    lines.append(
        "OPERATING POLICY"
    )

    lines.append(
        f"Required accepted accuracy: "
        f"{TARGET_ACCEPTED_ACCURACY * 100:.2f}%"
    )

    lines.append(
        f"Selected threshold: "
        f"{selected['threshold']:.3f}"
    )

    lines.append(
        f"Accepted accuracy: "
        f"{selected['accuracy'] * 100:.4f}%"
    )

    lines.append(
        f"Coverage: "
        f"{selected['coverage'] * 100:.4f}%"
    )

    lines.append(
        f"Accepted images: "
        f"{selected['accepted']}"
    )

    lines.append(
        f"Rejected images: "
        f"{selected['rejected']}"
    )

    lines.append("")

    lines.append(
        "SELECTION REASON"
    )

    lines.append(
        selection_reason
    )

    lines.append("")

    lines.append(
        "DEPLOYMENT RULE"
    )

    lines.append(
        "If confidence >= threshold:"
    )

    lines.append(
        "    ACCEPT prediction automatically"
    )

    lines.append(
        "If confidence < threshold:"
    )

    lines.append(
        "    SEND for additional inspection"
    )

    lines.append("")

    lines.append(
        "PER-CLASS CONFIDENCE"
    )

    lines.append(
        "-" * 70
    )

    for class_name in EXPECTED_CLASSES:

        stats = per_class[
            class_name
        ]

        lines.append(
            f"{class_name:<10}"
            f" Images: {stats['images']:5d}"
            f" Accuracy: {stats['accuracy'] * 100:7.2f}%"
            f" Avg Conf: {stats['average_confidence']:.4f}"
            f" Wrong Conf: "
            f"{stats['average_wrong_confidence']:.4f}"
        )

    lines.append("")

    lines.append(
        "THRESHOLD TABLE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Threshold | Accepted | Rejected | Coverage | Accuracy"
    )

    for result in threshold_results:

        lines.append(
            f"{result['threshold']:.3f}"
            f" | {result['accepted']:8d}"
            f" | {result['rejected']:8d}"
            f" | {result['coverage'] * 100:7.2f}%"
            f" | {result['accuracy'] * 100:7.2f}%"
        )

    lines.append("")

    lines.append(
        "CONFUSION PAIRS"
    )

    lines.append(
        "-" * 70
    )

    if confusion_pairs:

        for (
            actual,
            predicted,
        ), count in confusion_pairs.items():

            lines.append(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": {count:4d}"
            )

    else:

        lines.append(
            "No validation misclassifications."
        )

    lines.append("")

    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "The operating threshold was selected "
        "using validation data only."
    )

    lines.append(
        "The untouched test set must be used later "
        "for final evaluation of this selected threshold."
    )

    lines.append("")

    with open(
        THRESHOLD_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
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
        "Operating Confidence Threshold Selection"
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

    print(
        f"Required accepted accuracy: "
        f"{TARGET_ACCEPTED_ACCURACY * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    dataset = load_validation_dataset()

    loader = create_loader(
        dataset
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model()

    # --------------------------------------------------------
    # Collect confidence predictions
    # --------------------------------------------------------

    (
        labels,
        predictions,
        confidences,
    ) = collect_validation_predictions(
        model,
        loader,
    )

    print()

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    overall = calculate_overall_performance(
        labels,
        predictions,
        confidences,
    )

    print(
        "## OVERALL VALIDATION PERFORMANCE"
    )

    print()

    print(
        f"Total validation images: "
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
        f"Validation accuracy: "
        f"{overall['accuracy']:.4f} "
        f"({overall['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Average confidence: "
        f"{overall['average_confidence']:.4f}"
    )

    print(
        f"Average confidence (correct): "
        f"{overall['average_correct_confidence']:.4f}"
    )

    print(
        f"Average confidence (incorrect): "
        f"{overall['average_incorrect_confidence']:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Per-class confidence
    # --------------------------------------------------------

    per_class = calculate_per_class_statistics(
        labels,
        predictions,
        confidences,
    )

    print(
        "## PER-CLASS CONFIDENCE"
    )

    print()

    print(
        "Class         Images    Accuracy   Avg Conf. Wrong Conf."
    )

    for class_name in EXPECTED_CLASSES:

        stats = per_class[
            class_name
        ]

        print(
            f"{class_name:<12}"
            f"{stats['images']:>6}"
            f"{stats['accuracy'] * 100:>11.2f}%"
            f"{stats['average_confidence']:>11.4f}"
            f"{stats['average_wrong_confidence']:>12.4f}"
        )

    print()

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------

    threshold_results = evaluate_all_thresholds(
        labels,
        predictions,
        confidences,
    )

    print_threshold_results(
        threshold_results
    )

    # --------------------------------------------------------
    # Select threshold
    # --------------------------------------------------------

    (
        selected,
        selection_reason,
    ) = select_operating_threshold(
        threshold_results
    )

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    confusion_pairs = calculate_confusion_pairs(
        labels,
        predictions,
    )

    print(
        "## CONFUSION PAIRS"
    )

    print()

    if confusion_pairs:

        for (
            actual,
            predicted,
        ), count in confusion_pairs.items():

            print(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": {count:4d}"
            )

    else:

        print(
            "No validation misclassifications."
        )

    print()

    # --------------------------------------------------------
    # Selected operating threshold
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "RECOMMENDED OPERATING THRESHOLD"
    )

    print("=" * 60)

    print()

    print(
        f"Threshold: "
        f"{selected['threshold']:.3f}"
    )

    print(
        f"Accepted accuracy: "
        f"{selected['accuracy'] * 100:.2f}%"
    )

    print(
        f"Coverage: "
        f"{selected['coverage'] * 100:.2f}%"
    )

    print(
        f"Accepted predictions: "
        f"{selected['accepted']}"
    )

    print(
        f"Rejected predictions: "
        f"{selected['rejected']}"
    )

    print()

    print(
        "Operating rule:"
    )

    print(
        f"  Confidence >= "
        f"{selected['threshold']:.3f}"
        "  -> ACCEPT automatically"
    )

    print(
        f"  Confidence <  "
        f"{selected['threshold']:.3f}"
        "  -> SEND for additional inspection"
    )

    print()

    print(
        "Selection reason:"
    )

    print(
        f"  {selection_reason}"
    )

    print()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_threshold_json(
        selected,
        selection_reason,
        overall,
        per_class,
    )

    save_report(
        selected,
        selection_reason,
        overall,
        per_class,
        threshold_results,
        confusion_pairs,
    )

    print(
        "Operating threshold configuration saved:"
    )

    print(
        f"  {THRESHOLD_JSON_PATH}"
    )

    print()

    print(
        "Operating threshold report saved:"
    )

    print(
        f"  {THRESHOLD_REPORT_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Final instructions
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "OPERATING THRESHOLD SELECTION COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Important:"
    )

    print(
        "The threshold was selected using the "
        "validation dataset only."
    )

    print()

    print(
        "Do NOT tune this threshold using the "
        "test dataset."
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Evaluate the selected operating threshold "
        "once on the untouched test set."
    )

    print()

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()