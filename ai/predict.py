"""
ai/predict.py

Single-image crop disease inference
for AgriculturalQuadcopter.

Uses the best fine-tuned MobileNetV3-Small model.

Features:
    - Loads the fine-tuned model
    - Predicts one crop-disease class
    - Calculates confidence
    - Applies a confidence threshold
    - Returns ACCEPT or UNCERTAIN
    - Displays all class probabilities
"""

import os
import sys

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
    "crop_disease_mobilenet_finetuned.pth"
)

IMAGE_SIZE = 224

# ------------------------------------------------------------
# Minimum confidence required to accept a prediction.
#
# Example:
#
#   0.80 = 80%
#
# If confidence is below this value, the system reports:
#
#   UNCERTAIN
#
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.80


# ------------------------------------------------------------
# Fallback class names
#
# Normally these are loaded directly from the checkpoint.
# ------------------------------------------------------------

DEFAULT_CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Image Transformation
# ============================================================

def create_inference_transform():

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
            ]
        ),
    ])


# ============================================================
# Load Model
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Fine-tuned model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run the fine-tuning experiment first."
        )

    print(
        "Loading fine-tuned model..."
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Create the same architecture used during training
    # --------------------------------------------------------

    model = (
        create_transfer_learning_model()
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Move model to selected device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Load class names from checkpoint
    # --------------------------------------------------------

    class_names = checkpoint.get(
        "class_names",
        DEFAULT_CLASS_NAMES
    )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

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

    print(
        f"Validation accuracy: "
        f"{checkpoint.get('validation_accuracy', 0.0):.4f}"
    )

    print(
        f"Number of classes: "
        f"{checkpoint.get('num_classes', len(class_names))}"
    )

    print()

    return (
        model,
        class_names
    )


# ============================================================
# Predict Image
# ============================================================

def predict_image(
    model,
    class_names,
    image_path,
):

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Create preprocessing pipeline
    # --------------------------------------------------------

    transform = (
        create_inference_transform()
    )

    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # [3, 224, 224]
    #
    # becomes
    #
    # [1, 3, 224, 224]
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    probabilities = (
        probabilities[0]
        .cpu()
    )

    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    predicted_index = (
        probabilities.argmax()
        .item()
    )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_class = (
        class_names[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    confidence = (
        probabilities[
            predicted_index
        ].item()
    )

    # --------------------------------------------------------
    # Confidence decision
    #
    # Example:
    #
    # 0.8211 >= 0.80
    #
    # therefore:
    #
    # ACCEPT
    #
    # --------------------------------------------------------

    if (
        confidence
        >= CONFIDENCE_THRESHOLD
    ):

        decision = "ACCEPT"

    else:

        decision = "UNCERTAIN"

    # --------------------------------------------------------
    # Return inference results
    # --------------------------------------------------------

    return (
        predicted_class,
        confidence,
        decision,
        probabilities,
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
        "Crop Disease Prediction"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            "  python -m ai.predict IMAGE_PATH"
        )

        print()

        print(
            "Example:"
        )

        print(
            "  python -m ai.predict "
            "data/sample_field.jpg"
        )

        return

    # --------------------------------------------------------
    # Image path
    # --------------------------------------------------------

    image_path = (
        sys.argv[1]
    )

    print(
        f"Image: {image_path}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, class_names = (
        load_model()
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    (
        predicted_class,
        confidence,
        decision,
        probabilities,
    ) = predict_image(
        model,
        class_names,
        image_path,
    )

    # ========================================================
    # Prediction Result
    # ========================================================

    print("=" * 60)

    print(
        "PREDICTION"
    )

    print("=" * 60)

    print()

    print(
        f"Class: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Threshold: "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    print(
        f"Decision: "
        f"{decision}"
    )

    print()

    # ========================================================
    # Interpretation
    # ========================================================

    if decision == "ACCEPT":

        print(
            "The prediction meets the confidence threshold."
        )

    else:

        print(
            "The prediction is below the confidence threshold."
        )

        print(
            "The system should treat this result as uncertain."
        )

    print()

    # ========================================================
    # Class Probabilities
    # ========================================================

    print(
        "CLASS PROBABILITIES"
    )

    print("-" * 60)

    results = []

    for index, class_name in enumerate(
        class_names
    ):

        probability = (
            probabilities[
                index
            ].item()
        )

        results.append(
            (
                class_name,
                probability
            )
        )

    # --------------------------------------------------------
    # Highest probability first
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for (
        class_name,
        probability
    ) in results:

        print(
            f"{class_name:<12}"
            f": "
            f"{probability * 100:7.2f}%"
        )

    # ========================================================
    # Complete
    # ========================================================

    print()

    print("=" * 60)

    print(
        "PREDICTION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()