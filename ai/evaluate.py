"""
ai/evaluate.py

Evaluate the trained crop disease classifier
on the unseen test dataset.
"""

import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ai.dataset import create_dataset
from ai.augmentation import (
    get_validation_transform,
)
from ai.cnn import (
    create_model,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.path.join(
    "trained_models",
    "crop_disease_mobilenet_blspot_targeted.pth"
)

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model
# ============================================================

def load_trained_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = create_model(
        pretrained=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model, checkpoint


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Crop Disease Model Evaluation"
    )

    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = create_dataset(
        "test",
        transform=get_validation_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = (
        load_trained_model()
    )

    print(
        "Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    print(
        "Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .numpy()
            )

            all_predictions.extend(
                predictions
            )

            all_labels.extend(
                labels
            )

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(
        "=" * 60
    )

    print(
        f"TEST ACCURACY: "
        f"{accuracy:.4f}"
    )

    print(
        f"TEST ACCURACY: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()

    print(
        "PER-CLASS PERFORMANCE"
    )

    print(
        "-" * 60
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "-" * 60
    )

    print(
        "Rows = actual"
    )

    print(
        "Columns = predicted"
    )

    print()

    print(
        "              "
        + " ".join(
            f"{name:>10}"
            for name in CLASS_NAMES
        )
    )

    for index, row in enumerate(
        matrix
    ):

        print(
            f"{CLASS_NAMES[index]:>12} "
            + " ".join(
                f"{value:10d}"
                for value in row
            )
        )

    print()

    print("=" * 60)

    print(
        "MODEL EVALUATION COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()