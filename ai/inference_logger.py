"""
ai/inference_logger.py

AgriculturalQuadcopter
Inference Logging System

Records every AI prediction made by the crop disease model.

Five classes:

    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Decision logic:

    confidence >= 0.95
        -> AUTO-ACCEPTED

    confidence < 0.95
        -> NEEDS-REVIEW
"""

import csv
import os
import sys
from datetime import datetime

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
    "crop_disease_mobilenet_blspot_targeted.pth",
)

LOG_DIRECTORY = "mission_logs"

LOG_FILE = os.path.join(
    LOG_DIRECTORY,
    "inference_log.csv",
)

IMAGE_SIZE = 224

CONFIDENCE_THRESHOLD = 0.95


CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Transform
# ============================================================

def create_transform():

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
# Load model
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Targeted model not found:\n"
            f"{MODEL_PATH}"
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

    return (
        model,
        checkpoint,
    )


# ============================================================
# Predict
# ============================================================

def predict_image(
    model,
    image_path,
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = create_transform()

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    confidence, prediction = (
        probabilities.max(
            dim=1
        )
    )

    predicted_index = (
        prediction.item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence_value = (
        confidence.item()
    )

    probability_values = (
        probabilities[0]
        .cpu()
        .tolist()
    )

    return (
        predicted_class,
        predicted_index,
        confidence_value,
        probability_values,
    )


# ============================================================
# Decision
# ============================================================

def make_decision(
    confidence,
):

    if (
        confidence
        >= CONFIDENCE_THRESHOLD
    ):

        return "AUTO-ACCEPTED"

    return "NEEDS-REVIEW"


# ============================================================
# Create CSV
# ============================================================

def initialize_log():

    os.makedirs(
        LOG_DIRECTORY,
        exist_ok=True
    )

    if os.path.exists(
        LOG_FILE
    ):

        return

    headers = [
        "timestamp",
        "image_path",
        "predicted_class",
        "class_index",
        "confidence",
        "decision",
        "blight_probability",
        "healthy_probability",
        "leafspot_probability",
        "mildew_probability",
        "rust_probability",
    ]

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            headers
        )


# ============================================================
# Save prediction
# ============================================================

def save_prediction(
    image_path,
    predicted_class,
    predicted_index,
    confidence,
    decision,
    probabilities,
):

    initialize_log()

    timestamp = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    row = [

        timestamp,

        image_path,

        predicted_class,

        predicted_index,

        f"{confidence:.6f}",

        decision,

        f"{probabilities[0]:.6f}",

        f"{probabilities[1]:.6f}",

        f"{probabilities[2]:.6f}",

        f"{probabilities[3]:.6f}",

        f"{probabilities[4]:.6f}",
    ]

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            row
        )


# ============================================================
# Print result
# ============================================================

def print_result(
    image_path,
    predicted_class,
    predicted_index,
    confidence,
    decision,
    probabilities,
):

    print()

    print("=" * 60)

    print(
        "AGRICULTURALQUADCOPTER"
    )

    print(
        "INFERENCE RESULT"
    )

    print("=" * 60)

    print()

    print(
        f"Image:"
    )

    print(
        f"  {image_path}"
    )

    print()

    print(
        "Prediction:"
    )

    print(
        f"  Disease: "
        f"{predicted_class}"
    )

    print(
        f"  Class index: "
        f"{predicted_index}"
    )

    print(
        f"  Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print()

    print(
        "Class probabilities:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_name:<10}: "
            f"{probabilities[index] * 100:6.2f}%"
        )

    print()

    print(
        "Confidence threshold:"
    )

    print(
        f"  {CONFIDENCE_THRESHOLD * 100:.2f}%"
    )

    print()

    print(
        "Decision:"
    )

    print(
        f"  {decision}"
    )

    print()

    if decision == "AUTO-ACCEPTED":

        print(
            "AI decision can be used "
            "automatically."
        )

    else:

        print(
            "AI decision requires "
            "additional inspection."
        )

    print()

    print(
        f"Log file:"
    )

    print(
        f"  {LOG_FILE}"
    )

    print()

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
        "Inference Logging System"
    )

    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print()

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python -m ai.inference_logger "
            "\"path_to_image\""
        )

        print()

        return

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading targeted model..."
    )

    model, checkpoint = load_model()

    print(
        "Model loaded successfully."
    )

    print()

    print(
        f"Architecture: "
        f"{checkpoint.get('architecture', 'Unknown')}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint.get('validation_accuracy', 0.0):.4f}"
    )

    print()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "Running inference..."
    )

    (
        predicted_class,
        predicted_index,
        confidence,
        probabilities,
    ) = predict_image(
        model,
        image_path,
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = make_decision(
        confidence
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_prediction(
        image_path,
        predicted_class,
        predicted_index,
        confidence,
        decision,
        probabilities,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_result(
        image_path,
        predicted_class,
        predicted_index,
        confidence,
        decision,
        probabilities,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()