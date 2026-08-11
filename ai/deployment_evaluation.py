"""
ai/deployment_evaluation.py

AgriculturalQuadcopter
Final Deployment Evaluation

Purpose
-------
Evaluate the LOCKED agricultural disease classifier and
LOCKED operating confidence threshold on the untouched
test dataset.

IMPORTANT
---------
The threshold is NOT selected or modified here.

It is loaded from:

    confidence_analysis/operating_threshold.json

Current selected threshold:

    0.650

This script answers:

    How does the finalized deployment policy behave
    on the untouched test set?

Classes:
    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Decision:

    confidence >= 0.650
        ACCEPT

    confidence < 0.650
        NEEDS-REVIEW
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
# PROJECT CONFIGURATION
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


THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "confidence_analysis",
    "operating_threshold.json",
)


OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "deployment_analysis",
)


REPORT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "final_deployment_report.txt",
)


JSON_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "final_deployment_results.json",
)


IMAGE_SIZE = 224
BATCH_SIZE = 32
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
    name: index
    for index, name in enumerate(
        EXPECTED_CLASSES
    )
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
# DATASET
# ============================================================

class FiveClassTestDataset(Dataset):

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

        self._build_samples()

    def _build_samples(self):

        if not os.path.isdir(
            self.root
        ):

            raise FileNotFoundError(
                f"Test directory does not exist:\n"
                f"{self.root}"
            )

        directories = {
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

        missing = [
            name
            for name in EXPECTED_CLASSES
            if name not in directories
        ]

        if missing:

            raise RuntimeError(
                "Test dataset is missing classes:\n"
                f"{missing}\n\n"
                f"Directory:\n{self.root}"
            )

        for class_name in EXPECTED_CLASSES:

            class_directory = os.path.join(
                self.root,
                class_name,
            )

            class_index = CLASS_TO_INDEX[
                class_name
            ]

            for current_root, _dirs, files in os.walk(
                class_directory
            ):

                for filename in sorted(files):

                    extension = os.path.splitext(
                        filename
                    )[1].lower()

                    if extension not in SUPPORTED_EXTENSIONS:
                        continue

                    path = os.path.join(
                        current_root,
                        filename,
                    )

                    self.samples.append(
                        (
                            path,
                            class_index,
                        )
                    )

        if not self.samples:

            raise RuntimeError(
                "No test images found."
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        path, label = self.samples[
            index
        ]

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            label,
            path,
        )


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        raise FileNotFoundError(
            "Operating threshold not found:\n"
            f"{THRESHOLD_PATH}\n\n"
            "Run:\n"
            "python -m ai.operating_threshold"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    threshold = float(
        data["operating_threshold"]
    )

    return threshold


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
        "Loading finalized model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
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

    print(
        "Architecture: "
        f"{checkpoint.get('architecture', 'Unknown')}"
    )

    print(
        "Training type: "
        f"{checkpoint.get('training_type', 'Unknown')}"
    )

    print()

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    threshold,
):

    total = 0
    correct = 0

    accepted = 0
    accepted_correct = 0

    rejected = 0
    rejected_correct = 0

    class_total = Counter()
    class_correct = Counter()
    class_accepted = Counter()
    class_accepted_correct = Counter()

    confusion = Counter()

    review_images = []

    processed = 0

    print(
        "Running final deployment evaluation..."
    )

    print()

    with torch.no_grad():

        for images, labels, paths in loader:

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

            confidences, predictions = probabilities.max(
                dim=1
            )

            for i in range(
                len(labels)
            ):

                label = int(
                    labels[i].item()
                )

                prediction = int(
                    predictions[i].item()
                )

                confidence = float(
                    confidences[i].item()
                )

                path = paths[i]

                total += 1

                class_total[
                    label
                ] += 1

                is_correct = (
                    prediction
                    == label
                )

                if is_correct:

                    correct += 1

                    class_correct[
                        label
                    ] += 1

                if confidence >= threshold:

                    accepted += 1

                    class_accepted[
                        label
                    ] += 1

                    if is_correct:

                        accepted_correct += 1

                        class_accepted_correct[
                            label
                        ] += 1

                else:

                    rejected += 1

                    if is_correct:

                        rejected_correct += 1

                    review_images.append({
                        "path": path,
                        "actual":
                            EXPECTED_CLASSES[
                                label
                            ],
                        "predicted":
                            EXPECTED_CLASSES[
                                prediction
                            ],
                        "confidence":
                            confidence,
                    })

                if not is_correct:

                    confusion[
                        (
                            EXPECTED_CLASSES[
                                label
                            ],
                            EXPECTED_CLASSES[
                                prediction
                            ],
                        )
                    ] += 1

            processed += len(labels)

            if (
                processed % 1000 == 0
                or processed == len(loader.dataset)
            ):

                current_accuracy = (
                    correct
                    / total
                )

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(loader.dataset)}"
                    f" | Accuracy: "
                    f"{current_accuracy:.4f}"
                )

    overall_accuracy = (
        correct / total
    )

    coverage = (
        accepted / total
    )

    accepted_accuracy = (
        accepted_correct / accepted
        if accepted > 0
        else 0.0
    )

    rejected_accuracy = (
        rejected_correct / rejected
        if rejected > 0
        else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": overall_accuracy,
        "threshold": threshold,
        "accepted": accepted,
        "rejected": rejected,
        "coverage": coverage,
        "accepted_correct": accepted_correct,
        "accepted_incorrect":
            accepted - accepted_correct,
        "accepted_accuracy":
            accepted_accuracy,
        "rejected_correct":
            rejected_correct,
        "rejected_accuracy":
            rejected_accuracy,
        "class_total":
            dict(class_total),
        "class_correct":
            dict(class_correct),
        "class_accepted":
            dict(class_accepted),
        "class_accepted_correct":
            dict(class_accepted_correct),
        "confusion":
            {
                f"{a} -> {p}": count
                for (a, p), count
                in confusion.items()
            },
        "review_images":
            review_images,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
):

    threshold = results[
        "threshold"
    ]

    print()

    print("=" * 60)

    print(
        "FINAL DEPLOYMENT EVALUATION"
    )

    print("=" * 60)

    print()

    print(
        f"Test images: "
        f"{results['total']}"
    )

    print(
        f"Correct predictions: "
        f"{results['correct']}"
    )

    print(
        f"Incorrect predictions: "
        f"{results['incorrect']}"
    )

    print(
        f"Test accuracy: "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "LOCKED OPERATING POLICY"
    )

    print(
        f"Threshold: "
        f"{threshold:.3f}"
    )

    print()

    print(
        f"Accepted automatically: "
        f"{results['accepted']}"
    )

    print(
        f"Sent for review: "
        f"{results['rejected']}"
    )

    print(
        f"Coverage: "
        f"{results['coverage'] * 100:.2f}%"
    )

    print(
        f"Accepted accuracy: "
        f"{results['accepted_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Confusion pairs:"
    )

    if results["confusion"]:

        for pair, count in results[
            "confusion"
        ].items():

            print(
                f"  {pair:<25}: "
                f"{count}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "FINAL INTERPRETATION"
    )

    print(
        "High-confidence predictions are "
        "accepted automatically."
    )

    print(
        "Low-confidence predictions are "
        "sent for additional inspection."
    )

    print()

    print(
        "This is the finalized deployment policy."
    )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_data = {
        key: value
        for key, value
        in results.items()
        if key != "review_images"
    }

    json_data[
        "model"
    ] = os.path.basename(
        MODEL_PATH
    )

    json_data[
        "threshold_source"
    ] = THRESHOLD_PATH

    json_data[
        "policy"
    ] = {
        "accepted":
            "confidence >= threshold",
        "review":
            "confidence < threshold",
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "AgriculturalQuadcopter"
    )

    lines.append(
        "FINAL DEPLOYMENT EVALUATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "MODEL"
    )

    lines.append(
        MODEL_PATH
    )

    lines.append("")

    lines.append(
        "OPERATING THRESHOLD"
    )

    lines.append(
        f"{results['threshold']:.3f}"
    )

    lines.append("")

    lines.append(
        "FINAL TEST PERFORMANCE"
    )

    lines.append(
        f"Total images: "
        f"{results['total']}"
    )

    lines.append(
        f"Correct: "
        f"{results['correct']}"
    )

    lines.append(
        f"Incorrect: "
        f"{results['incorrect']}"
    )

    lines.append(
        f"Accuracy: "
        f"{results['accuracy'] * 100:.2f}%"
    )

    lines.append("")

    lines.append(
        "DEPLOYMENT POLICY"
    )

    lines.append(
        f"Accepted: "
        f"{results['accepted']}"
    )

    lines.append(
        f"Review: "
        f"{results['rejected']}"
    )

    lines.append(
        f"Coverage: "
        f"{results['coverage'] * 100:.2f}%"
    )

    lines.append(
        f"Accepted accuracy: "
        f"{results['accepted_accuracy'] * 100:.2f}%"
    )

    lines.append("")

    lines.append(
        "CLASS RESULTS"
    )

    lines.append(
        "-" * 70
    )

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        total = results[
            "class_total"
        ].get(
            index,
            0,
        )

        correct = results[
            "class_correct"
        ].get(
            index,
            0,
        )

        accepted = results[
            "class_accepted"
        ].get(
            index,
            0,
        )

        accuracy = (
            correct / total
            if total
            else 0
        )

        lines.append(
            f"{class_name:<12}"
            f" Images: {total:5d}"
            f" Accuracy: "
            f"{accuracy * 100:7.2f}%"
            f" Accepted: {accepted:5d}"
        )

    lines.append("")

    lines.append(
        "CONFUSION PAIRS"
    )

    lines.append(
        "-" * 70
    )

    for pair, count in results[
        "confusion"
    ].items():

        lines.append(
            f"{pair:<30}"
            f"{count:5d}"
        )

    lines.append("")

    lines.append(
        "REVIEW QUEUE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Images requiring review: "
        f"{len(results['review_images'])}"
    )

    lines.append("")

    for item in results[
        "review_images"
    ][:100]:

        lines.append(
            f"{item['actual']:<10} -> "
            f"{item['predicted']:<10} | "
            f"{item['confidence'] * 100:6.2f}% | "
            f"{item['path']}"
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
        "This evaluation uses the untouched test "
        "dataset exactly once for final deployment "
        "evaluation."
    )

    lines.append("")

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )

    print(
        "Results saved:"
    )

    print(
        f"  {JSON_PATH}"
    )

    print(
        f"  {REPORT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Final Deployment Evaluation"
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

    for index, name in enumerate(
        EXPECTED_CLASSES
    ):

        print(
            f"{index} = {name}"
        )

    print()

    # --------------------------------------------------------
    # Load locked threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    print(
        "Locked operating threshold:"
    )

    print(
        f"{threshold:.3f}"
    )

    print()

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print(
        "Test dataset:"
    )

    print(
        TEST_SEARCH_ROOT
    )

    print()

    dataset = FiveClassTestDataset(
        root=TEST_SEARCH_ROOT,
        transform=create_test_transform(),
    )

    print(
        f"Test images found: "
        f"{len(dataset)}"
    )

    print()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate(
        model,
        loader,
        threshold,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "FINAL DEPLOYMENT EVALUATION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()