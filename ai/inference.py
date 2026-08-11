"""
ai/inference.py

AgriculturalQuadcopter
Single Image Disease Inference

Purpose
-------
Run the trained five-class crop disease classifier on one image.

Classes:
    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Operating policy
----------------
The confidence threshold is NOT hardcoded.

It is loaded from:

    confidence_analysis/operating_threshold.json

The current validated operating threshold is:

    0.650

Decision:

    confidence >= threshold
        -> ACCEPT

    confidence < threshold
        -> NEEDS-REVIEW
"""


# ============================================================
# Imports
# ============================================================

import os
import sys
import json

from PIL import Image

import torch
from torchvision import transforms

from ai.cnn import create_transfer_learning_model
from ai.training import DEVICE


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    )
)


# ------------------------------------------------------------
# Trained model
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "crop_disease_mobilenet_blspot_targeted.pth",
)


# ------------------------------------------------------------
# Operating threshold
# ------------------------------------------------------------

THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "confidence_analysis",
    "operating_threshold.json",
)


# ------------------------------------------------------------
# Image configuration
# ------------------------------------------------------------

IMAGE_SIZE = 224


# ------------------------------------------------------------
# Five-class configuration
# ------------------------------------------------------------

EXPECTED_CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Image transformation
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
            ],
        ),
    ])


# ============================================================
# Load operating threshold
# ============================================================

def load_operating_threshold():

    """
    Load the operating confidence threshold selected
    using the validation dataset.

    IMPORTANT:
    This function does not calculate or tune the threshold.

    It only reads the already-selected threshold.
    """

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        raise FileNotFoundError(
            "\n"
            "Operating threshold file was not found.\n\n"
            f"Expected file:\n"
            f"{THRESHOLD_PATH}\n\n"
            "Run the threshold selection first:\n\n"
            "python -m ai.operating_threshold\n"
        )

    try:

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Operating threshold JSON is invalid:\n"
            f"{THRESHOLD_PATH}\n\n"
            f"Error: {exc}"
        ) from exc

    if "operating_threshold" not in data:

        raise RuntimeError(
            "The operating threshold file does not "
            "contain 'operating_threshold'.\n\n"
            f"File:\n{THRESHOLD_PATH}"
        )

    try:

        threshold = float(
            data["operating_threshold"]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise RuntimeError(
            "Invalid operating threshold value:\n"
            f"{data['operating_threshold']}"
        ) from exc

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise RuntimeError(
            "Operating threshold must be between "
            "0.0 and 1.0.\n\n"
            f"Loaded value: {threshold}"
        )

    return threshold


# ============================================================
# Load trained model
# ============================================================

def load_model():

    """
    Load the trained targeted MobileNetV3-Small model.
    """

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\n"
            "Trained model was not found.\n\n"
            f"Expected model:\n"
            f"{MODEL_PATH}\n\n"
            "Make sure the targeted training lesson "
            "has been completed."
        )

    print(
        "Loading model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = create_transfer_learning_model()

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        raise RuntimeError(
            "The model checkpoint does not contain "
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

    if isinstance(
        checkpoint,
        dict
    ):

        architecture = checkpoint.get(
            "architecture",
            "Unknown",
        )

        training_type = checkpoint.get(
            "training_type",
            "Unknown",
        )

        validation_accuracy = checkpoint.get(
            "validation_accuracy",
            None,
        )

        print(
            f"Architecture: {architecture}"
        )

        print(
            f"Training type: {training_type}"
        )

        if validation_accuracy is not None:

            print(
                "Validation accuracy: "
                f"{float(validation_accuracy):.4f}"
            )

    print()

    return model


# ============================================================
# Validate image path
# ============================================================

def validate_image_path(
    image_path,
):

    if not image_path:

        raise ValueError(
            "No image path was provided."
        )

    image_path = os.path.abspath(
        image_path
    )

    if not os.path.isfile(
        image_path
    ):

        raise FileNotFoundError(
            "\n"
            "Image file was not found:\n"
            f"{image_path}"
        )

    return image_path


# ============================================================
# Load image
# ============================================================

def load_image(
    image_path,
):

    try:

        image = Image.open(
            image_path
        )

        image = image.convert(
            "RGB"
        )

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "Could not open image:\n"
            f"{image_path}\n\n"
            f"Error: {exc}"
        ) from exc

    return image


# ============================================================
# Run prediction
# ============================================================

def predict_image(
    image_path,
    model,
    transform,
    confidence_threshold,
):

    """
    Run inference on one image.

    Returns:
        dictionary containing prediction,
        confidence, probabilities and decision.
    """

    image_path = validate_image_path(
        image_path
    )

    image = load_image(
        image_path
    )

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

        confidence_tensor, prediction_tensor = (
            probabilities.max(
                dim=1
            )
        )

    confidence = float(
        confidence_tensor.item()
    )

    class_index = int(
        prediction_tensor.item()
    )

    if not (
        0
        <= class_index
        < len(EXPECTED_CLASSES)
    ):

        raise RuntimeError(
            "Model returned an invalid class index:\n"
            f"{class_index}"
        )

    disease = EXPECTED_CLASSES[
        class_index
    ]

    # --------------------------------------------------------
    # Deployment decision
    # --------------------------------------------------------

    if confidence >= confidence_threshold:

        decision = "ACCEPT"

    else:

        decision = "NEEDS-REVIEW"

    # --------------------------------------------------------
    # All class probabilities
    # --------------------------------------------------------

    probability_values = (
        probabilities[
            0
        ]
        .cpu()
        .tolist()
    )

    class_probabilities = {}

    for index, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        class_probabilities[
            class_name
        ] = float(
            probability_values[index]
        )

    return {
        "image": image_path,

        "disease": disease,

        "class_index": class_index,

        "confidence": confidence,

        "threshold": confidence_threshold,

        "decision": decision,

        "class_probabilities":
            class_probabilities,
    }


# ============================================================
# Print inference result
# ============================================================

def print_result(
    result,
):

    print()

    print(
        "=" * 60
    )

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "AI IMAGE INFERENCE"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Image:"
    )

    print(
        result["image"]
    )

    print()

    print(
        "Prediction:"
    )

    print(
        f"Disease: "
        f"{result['disease']}"
    )

    print(
        f"Class index: "
        f"{result['class_index']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print()

    print(
        "All class probabilities:"
    )

    for class_name in EXPECTED_CLASSES:

        probability = result[
            "class_probabilities"
        ][
            class_name
        ]

        print(
            f"{class_name:<10}: "
            f"{probability * 100:6.2f}%"
        )

    print()

    print(
        "Decision:"
    )

    print(
        result["decision"]
    )

    print()

    print(
        "Operating confidence threshold:"
    )

    print(
        f"{result['threshold'] * 100:.2f}%"
    )

    print()

    if result["decision"] == "ACCEPT":

        print(
            "The prediction meets the operating "
            "confidence threshold."
        )

        print(
            "Automatic acceptance is permitted."
        )

    else:

        print(
            "The prediction is below the operating "
            "confidence threshold."
        )

        print(
            "Additional inspection is required."
        )

    print()

    print(
        "=" * 60
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check command-line arguments
    # --------------------------------------------------------

    if len(
        sys.argv
    ) != 2:

        print()

        print(
            "Usage:"
        )

        print()

        print(
            'python -m ai.inference '
            '"path\\to\\image.jpg"'
        )

        print()

        print(
            "Example:"
        )

        print()

        print(
            'python -m ai.inference '
            '"datasets\\crop_disease\\test\\'
            'Blight\\test_000995.jpg"'
        )

        print()

        sys.exit(
            1
        )

    image_path = sys.argv[
        1
    ]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"Device: {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Load operating threshold
    # --------------------------------------------------------

    confidence_threshold = (
        load_operating_threshold()
    )

    print(
        "Confidence threshold: "
        f"{confidence_threshold:.3f}"
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
        "Operating threshold source:"
    )

    print(
        THRESHOLD_PATH
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create preprocessing
    # --------------------------------------------------------

    transform = (
        create_inference_transform()
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print(
        "Running inference..."
    )

    result = predict_image(
        image_path=image_path,
        model=model,
        transform=transform,
        confidence_threshold=confidence_threshold,
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print_result(
        result
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()