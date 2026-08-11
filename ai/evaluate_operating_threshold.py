"""
ai/evaluate_operating_threshold.py

AgriculturalQuadcopter
Final Operating Threshold Test Evaluation

IMPORTANT
---------
The operating threshold was selected using the validation set.

This module evaluates that FIXED threshold exactly once
on the untouched test set.

The threshold must NOT be changed based on test results.

Classes:
    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Operating threshold:
    Loaded from confidence_analysis/operating_threshold.json

Policy:
    confidence >= threshold
        -> ACCEPT

    confidence < threshold
        -> ADDITIONAL INSPECTION
"""

import os
import json
from collections import Counter

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ai.cnn import create_transfer_learning_model
from ai.training import DEVICE


# ============================================================
# PROJECT PATHS
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


TEST_SEARCH_ROOT = os.path.join(
    DATASET_ROOT,
    "test",
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "crop_disease_mobilenet_blspot_targeted.pth",
)


THRESHOLD_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "confidence_analysis",
    "operating_threshold.json",
)


OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "confidence_analysis",
)


REPORT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "final_operating_threshold_test_report.txt",
)


RESULT_JSON_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "final_operating_threshold_test.json",
)


IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0


# ============================================================
# FIVE CLASS CONFIGURATION
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
# TRANSFORM
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
# DATASET DISCOVERY
# ============================================================

def find_five_class_directory(search_root):

    search_root = os.path.abspath(
        search_root
    )

    print(
        "Searching recursively for the "
        "five-class test dataset..."
    )

    print()

    print(
        f"Search root:\n  {search_root}"
    )

    print()

    if not os.path.isdir(search_root):

        raise FileNotFoundError(
            "Test search directory does not exist:\n"
            f"{search_root}"
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
            "Could not find a five-class test dataset.\n\n"
            f"Search root:\n{search_root}\n\n"
            "Required classes:\n"
            f"{EXPECTED_CLASSES}"
        )

    candidates.sort(
        key=lambda path: (
            path.count(os.sep),
            len(path),
        )
    )

    selected = candidates[0]

    print(
        "Found valid five-class test dataset:"
    )

    print(
        f"  {selected}"
    )

    print()

    return selected


# ============================================================
# DATASET
# ============================================================

class FiveClassImageDataset(Dataset):

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

        actual_directories = {
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
        }

        missing_classes = [
            class_name
            for class_name in EXPECTED_CLASSES
            if class_name not in actual_directories
        ]

        if missing_classes:

            raise RuntimeError(
                "Missing required classes:\n"
                f"{missing_classes}\n\n"
                f"Dataset:\n{self.root}"
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

            class_index = CLASS_TO_INDEX[
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
                "No valid test images found."
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

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            label,
        )


# ============================================================
# LOAD FIXED THRESHOLD
# ============================================================

def load_operating_threshold():

    if not os.path.exists(
        THRESHOLD_JSON_PATH
    ):

        raise FileNotFoundError(
            "Operating threshold configuration not found:\n"
            f"{THRESHOLD_JSON_PATH}\n\n"
            "Run:\n"
            "python -m ai.operating_threshold"
        )

    with open(
        THRESHOLD_JSON_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        configuration = json.load(
            file
        )

    threshold = configuration[
        "operating_threshold"
    ]

    print(
        "FIXED OPERATING THRESHOLD"
    )

    print(
        f"  {threshold:.3f}"
    )

    print()

    print(
        "This threshold was selected using "
        "validation data only."
    )

    print(
        "It will NOT be changed during test evaluation."
    )

    print()

    return threshold, configuration


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_dataset():

    resolved_directory = find_five_class_directory(
        TEST_SEARCH_ROOT
    )

    print(
        "Resolved five-class test directory:"
    )

    print(
        f"  {resolved_directory}"
    )

    print()

    dataset = FiveClassImageDataset(
        root=resolved_directory,
        transform=create_test_transform(),
    )

    print(
        "Five-class test dataset verified."
    )

    print()

    counts = Counter(
        label
        for _path, label
        in dataset.samples
    )

    print(
        "## TEST DATASET DISTRIBUTION"
    )

    print()

    print(
        f"TOTAL: {len(dataset)}"
    )

    for class_index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        print(
            f"{class_name:<10}: "
            f"{counts[class_index]:6d}"
        )

    print()

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "Loading targeted model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = create_transfer_learning_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
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

    print()

    return model


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test_set(
    model,
    dataset,
    threshold,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    labels = []
    predictions = []
    confidences = []

    print(
        "Running FINAL test evaluation..."
    )

    print()

    processed = 0

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            batch_confidences, batch_predictions = (
                probabilities.max(
                    dim=1
                )
            )

            labels.extend(
                batch_labels.tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            confidences.extend(
                batch_confidences.cpu().tolist()
            )

            processed += batch_labels.size(0)

            if (
                processed % 1000 == 0
                or processed == len(dataset)
            ):

                correct = sum(
                    int(
                        p == y
                    )
                    for p, y
                    in zip(
                        predictions,
                        labels,
                    )
                )

                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                    f" | Accuracy: "
                    f"{correct / processed:.4f}"
                )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    total = len(labels)

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

    overall_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # Threshold policy
    # --------------------------------------------------------

    accepted_indices = [
        i
        for i, confidence
        in enumerate(confidences)
        if confidence >= threshold
    ]

    rejected_indices = [
        i
        for i, confidence
        in enumerate(confidences)
        if confidence < threshold
    ]

    accepted = len(
        accepted_indices
    )

    rejected = len(
        rejected_indices
    )

    accepted_correct = sum(
        int(
            predictions[i]
            == labels[i]
        )
        for i in accepted_indices
    )

    accepted_incorrect = (
        accepted
        - accepted_correct
    )

    rejected_correct = sum(
        int(
            predictions[i]
            == labels[i]
        )
        for i in rejected_indices
    )

    rejected_incorrect = (
        rejected
        - rejected_correct
    )

    accepted_accuracy = (
        accepted_correct / accepted
        if accepted
        else 0.0
    )

    coverage = (
        accepted / total
        if total
        else 0.0
    )

    rejection_rate = (
        rejected / total
        if total
        else 0.0
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    confusion_pairs = {}

    for actual, predicted in zip(
        labels,
        predictions,
    ):

        if actual == predicted:
            continue

        key = (
            EXPECTED_CLASSES[actual],
            EXPECTED_CLASSES[predicted],
        )

        confusion_pairs[key] = (
            confusion_pairs.get(
                key,
                0,
            )
            + 1
        )

    confusion_pairs = dict(
        sorted(
            confusion_pairs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "overall_accuracy": overall_accuracy,
        "average_confidence": average_confidence,

        "threshold": threshold,

        "accepted": accepted,
        "rejected": rejected,

        "accepted_correct": accepted_correct,
        "accepted_incorrect": accepted_incorrect,

        "rejected_correct": rejected_correct,
        "rejected_incorrect": rejected_incorrect,

        "accepted_accuracy": accepted_accuracy,
        "coverage": coverage,
        "rejection_rate": rejection_rate,

        "confusion_pairs": confusion_pairs,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()

    print("=" * 70)

    print(
        "FINAL OPERATING THRESHOLD TEST RESULT"
    )

    print("=" * 70)

    print()

    print(
        f"Test images:          "
        f"{results['total']}"
    )

    print(
        f"Correct predictions:  "
        f"{results['correct']}"
    )

    print(
        f"Incorrect predictions:"
        f" {results['incorrect']}"
    )

    print(
        f"Test accuracy:        "
        f"{results['overall_accuracy'] * 100:.2f}%"
    )

    print(
        f"Average confidence:   "
        f"{results['average_confidence']:.4f}"
    )

    print()

    print(
        "FIXED OPERATING POLICY"
    )

    print()

    print(
        f"Threshold: "
        f"{results['threshold']:.3f}"
    )

    print()

    print(
        "ACCEPTED"
    )

    print(
        f"  Images: "
        f"{results['accepted']}"
    )

    print(
        f"  Correct: "
        f"{results['accepted_correct']}"
    )

    print(
        f"  Incorrect: "
        f"{results['accepted_incorrect']}"
    )

    print(
        f"  Accuracy: "
        f"{results['accepted_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Coverage: "
        f"{results['coverage'] * 100:.2f}%"
    )

    print()

    print(
        "ADDITIONAL INSPECTION"
    )

    print(
        f"  Images: "
        f"{results['rejected']}"
    )

    print(
        f"  Correct: "
        f"{results['rejected_correct']}"
    )

    print(
        f"  Incorrect: "
        f"{results['rejected_incorrect']}"
    )

    print(
        f"  Rejection rate: "
        f"{results['rejection_rate'] * 100:.2f}%"
    )

    print()

    print(
        "CONFUSION PAIRS"
    )

    print(
        "-" * 50
    )

    if results["confusion_pairs"]:

        for (
            actual,
            predicted,
        ), count in results["confusion_pairs"].items():

            print(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": {count}"
            )

    else:

        print(
            "No errors."
        )

    print()


# ============================================================
# SAVE JSON
# ============================================================

def save_json(results):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    data = {
        "evaluation": "final_test",
        "threshold_source": (
            "validation_only"
        ),

        "operating_threshold":
            results["threshold"],

        "test_total":
            results["total"],

        "test_correct":
            results["correct"],

        "test_incorrect":
            results["incorrect"],

        "test_accuracy":
            results["overall_accuracy"],

        "average_confidence":
            results["average_confidence"],

        "accepted":
            results["accepted"],

        "rejected":
            results["rejected"],

        "accepted_correct":
            results["accepted_correct"],

        "accepted_incorrect":
            results["accepted_incorrect"],

        "rejected_correct":
            results["rejected_correct"],

        "rejected_incorrect":
            results["rejected_incorrect"],

        "accepted_accuracy":
            results["accepted_accuracy"],

        "coverage":
            results["coverage"],

        "rejection_rate":
            results["rejection_rate"],

        "confusion_pairs": {
            f"{actual} -> {predicted}":
                count
            for (
                actual,
                predicted,
            ), count
            in results["confusion_pairs"].items()
        },

        "policy": {
            "accepted":
                "confidence >= threshold",

            "additional_inspection":
                "confidence < threshold",
        },
    }

    with open(
        RESULT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    lines = []

    lines.append(
        "AgriculturalQuadcopter"
    )

    lines.append(
        "FINAL OPERATING THRESHOLD TEST REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "The operating threshold was selected using "
        "validation data only."
    )

    lines.append(
        "The threshold was NOT tuned using the test set."
    )

    lines.append("")

    lines.append(
        f"Operating threshold: "
        f"{results['threshold']:.3f}"
    )

    lines.append("")

    lines.append(
        "FINAL TEST PERFORMANCE"
    )

    lines.append(
        f"Test images: {results['total']}"
    )

    lines.append(
        f"Correct: {results['correct']}"
    )

    lines.append(
        f"Incorrect: {results['incorrect']}"
    )

    lines.append(
        f"Accuracy: "
        f"{results['overall_accuracy'] * 100:.4f}%"
    )

    lines.append("")

    lines.append(
        "OPERATING THRESHOLD PERFORMANCE"
    )

    lines.append(
        f"Accepted: {results['accepted']}"
    )

    lines.append(
        f"Rejected: {results['rejected']}"
    )

    lines.append(
        f"Accepted correct: "
        f"{results['accepted_correct']}"
    )

    lines.append(
        f"Accepted incorrect: "
        f"{results['accepted_incorrect']}"
    )

    lines.append(
        f"Rejected correct: "
        f"{results['rejected_correct']}"
    )

    lines.append(
        f"Rejected incorrect: "
        f"{results['rejected_incorrect']}"
    )

    lines.append(
        f"Accepted accuracy: "
        f"{results['accepted_accuracy'] * 100:.4f}%"
    )

    lines.append(
        f"Coverage: "
        f"{results['coverage'] * 100:.4f}%"
    )

    lines.append(
        f"Rejection rate: "
        f"{results['rejection_rate'] * 100:.4f}%"
    )

    lines.append("")

    lines.append(
        "DEPLOYMENT RULE"
    )

    lines.append(
        f"confidence >= "
        f"{results['threshold']:.3f}"
        " -> ACCEPT"
    )

    lines.append(
        f"confidence < "
        f"{results['threshold']:.3f}"
        " -> ADDITIONAL INSPECTION"
    )

    lines.append("")

    lines.append(
        "CONFUSION PAIRS"
    )

    lines.append(
        "-" * 50
    )

    for (
        actual,
        predicted,
    ), count in results["confusion_pairs"].items():

        lines.append(
            f"{actual} -> {predicted}: {count}"
        )

    lines.append("")

    lines.append(
        "CONCLUSION"
    )

    lines.append(
        "This is a final evaluation of the fixed "
        "operating threshold on the untouched test set."
    )

    lines.append(
        "The threshold must not be changed based on "
        "this result."
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Final Operating Threshold Test Evaluation"
    )

    print("=" * 70)

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
    # Load fixed threshold
    # --------------------------------------------------------

    threshold, _configuration = (
        load_operating_threshold()
    )

    # --------------------------------------------------------
    # Load untouched test set
    # --------------------------------------------------------

    dataset = load_test_dataset()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_test_set(
        model,
        dataset,
        threshold,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        results
    )

    save_report(
        results
    )

    print(
        "Final test JSON saved:"
    )

    print(
        f"  {RESULT_JSON_PATH}"
    )

    print()

    print(
        "Final test report saved:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print()

    print("=" * 70)

    print(
        "FINAL TEST EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do not change the 0.650 threshold based "
        "on this test result."
    )

    print()

    print(
        "Next lesson:"
    )

    print(
        "Integrate the fixed confidence policy "
        "into the agricultural quadcopter inference pipeline."
    )


if __name__ == "__main__":

    main()