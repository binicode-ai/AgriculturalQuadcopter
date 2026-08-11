"""
ai/batch_inference.py

AgriculturalQuadcopter
Batch AI Image Inference

Purpose
-------
Run the finalized five-class crop disease classifier on
an entire directory of images.

Deployment policy:
    confidence >= 0.650
        -> ACCEPT

    confidence < 0.650
        -> NEEDS-REVIEW

The operating threshold is loaded from:
    confidence_analysis/operating_threshold.json

Classes:
    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Outputs:
    deployment_analysis/
        batch_predictions.csv
        batch_summary.txt
"""


import os
import sys
import csv
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


CSV_OUTPUT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "batch_predictions.csv",
)


SUMMARY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "batch_summary.txt",
)


IMAGE_SIZE = 224


EXPECTED_CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Image transform
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
# Load operating threshold
# ============================================================

def load_operating_threshold():

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        raise FileNotFoundError(
            "Operating threshold file was not found:\n"
            f"{THRESHOLD_PATH}\n\n"
            "Run:\n"
            "python -m ai.operating_threshold"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    threshold = float(
        data["operating_threshold"]
    )

    print(
        f"Locked operating threshold: "
        f"{threshold:.3f}"
    )

    print()

    return threshold


# ============================================================
# Load model
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Finalized model was not found:\n"
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

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
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

    print()

    return model


# ============================================================
# Find images
# ============================================================

def find_images(
    input_directory,
):

    input_directory = os.path.abspath(
        input_directory
    )

    if not os.path.isdir(
        input_directory
    ):

        raise FileNotFoundError(
            "Input directory does not exist:\n"
            f"{input_directory}"
        )

    images = []

    for current_root, _directories, files in os.walk(
        input_directory
    ):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            image_path = os.path.join(
                current_root,
                filename,
            )

            images.append(
                image_path
            )

    images.sort()

    return images


# ============================================================
# Predict one image
# ============================================================

def predict_image(
    model,
    image_path,
    transform,
):

    try:

        image = Image.open(
            image_path
        ).convert(
            "RGB"
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

            output = model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1,
            )

            confidence, prediction = (
                probabilities.max(
                    dim=1
                )
            )

        prediction_index = int(
            prediction.item()
        )

        confidence_value = float(
            confidence.item()
        )

        class_name = EXPECTED_CLASSES[
            prediction_index
        ]

        probability_values = (
            probabilities[0]
            .cpu()
            .tolist()
        )

        return {
            "success": True,
            "disease": class_name,
            "class_index": prediction_index,
            "confidence": confidence_value,
            "probabilities": probability_values,
            "error": "",
        }

    except Exception as exc:

        return {
            "success": False,
            "disease": "",
            "class_index": "",
            "confidence": "",
            "probabilities": [],
            "error": str(exc),
        }


# ============================================================
# Process directory
# ============================================================

def run_batch_inference(
    model,
    images,
    threshold,
):

    transform = create_transform()

    results = []

    total = len(
        images
    )

    accepted = 0

    needs_review = 0

    failed = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        result = predict_image(
            model,
            image_path,
            transform,
        )

        relative_path = os.path.relpath(
            image_path,
            PROJECT_ROOT,
        )

        if not result["success"]:

            failed += 1

            results.append({
                "image": relative_path,
                "disease": "",
                "class_index": "",
                "confidence": "",
                "decision": "ERROR",
                "Blight": "",
                "Healthy": "",
                "LeafSpot": "",
                "Mildew": "",
                "Rust": "",
                "error": result["error"],
            })

        else:

            confidence = result[
                "confidence"
            ]

            if confidence >= threshold:

                decision = "ACCEPT"

                accepted += 1

            else:

                decision = "NEEDS-REVIEW"

                needs_review += 1

            probabilities = (
                result["probabilities"]
            )

            results.append({
                "image": relative_path,
                "disease": result["disease"],
                "class_index": result["class_index"],
                "confidence": f"{confidence:.6f}",
                "decision": decision,
                "Blight": f"{probabilities[0]:.6f}",
                "Healthy": f"{probabilities[1]:.6f}",
                "LeafSpot": f"{probabilities[2]:.6f}",
                "Mildew": f"{probabilities[3]:.6f}",
                "Rust": f"{probabilities[4]:.6f}",
                "error": "",
            })

        if (
            index % 100 == 0
            or index == total
        ):

            print(
                f"Processed "
                f"{index}/{total}"
                f" | Accepted: {accepted}"
                f" | Review: {needs_review}"
                f" | Errors: {failed}"
            )

    return (
        results,
        accepted,
        needs_review,
        failed,
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    results,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    fieldnames = [
        "image",
        "disease",
        "class_index",
        "confidence",
        "decision",
        "Blight",
        "Healthy",
        "LeafSpot",
        "Mildew",
        "Rust",
        "error",
    ]

    with open(
        CSV_OUTPUT_PATH,
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
            results
        )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    input_directory,
    total,
    accepted,
    needs_review,
    failed,
    threshold,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    successful = (
        total - failed
    )

    coverage = (
        accepted / successful
        if successful > 0
        else 0.0
    )

    lines = []

    lines.append(
        "AgriculturalQuadcopter"
    )

    lines.append(
        "Batch Deployment Inference Summary"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Input directory: {input_directory}"
    )

    lines.append(
        f"Total images: {total}"
    )

    lines.append(
        f"Successful predictions: {successful}"
    )

    lines.append(
        f"Errors: {failed}"
    )

    lines.append("")

    lines.append(
        "LOCKED OPERATING POLICY"
    )

    lines.append(
        f"Confidence threshold: {threshold:.3f}"
    )

    lines.append(
        "Confidence >= threshold -> ACCEPT"
    )

    lines.append(
        "Confidence < threshold -> NEEDS-REVIEW"
    )

    lines.append("")

    lines.append(
        "RESULTS"
    )

    lines.append(
        f"Accepted automatically: {accepted}"
    )

    lines.append(
        f"Sent for review: {needs_review}"
    )

    lines.append(
        f"Automatic coverage: {coverage * 100:.2f}%"
    )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "High-confidence predictions are accepted "
        "automatically."
    )

    lines.append(
        "Low-confidence predictions are sent for "
        "additional inspection."
    )

    lines.append("")

    lines.append(
        "The classifier and operating threshold "
        "are locked."
    )

    lines.append(
        "This batch inference module does not modify "
        "the trained model."
    )

    with open(
        SUMMARY_OUTPUT_PATH,
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
        "BATCH AI IMAGE INFERENCE"
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
    # Input directory
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            'python -m ai.batch_inference '
            '"path\\to\\image_folder"'
        )

        print()

        print(
            "Example:"
        )

        print(
            'python -m ai.batch_inference '
            '"datasets\\crop_disease\\test\\Blight"'
        )

        sys.exit(1)

    input_directory = sys.argv[1]

    input_directory = os.path.abspath(
        input_directory
    )

    print(
        "Input directory:"
    )

    print(
        f"  {input_directory}"
    )

    print()

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_operating_threshold()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images(
        input_directory
    )

    if not images:

        raise RuntimeError(
            "No supported images were found in:\n"
            f"{input_directory}"
        )

    print(
        f"Images found: {len(images)}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print(
        "Running batch inference..."
    )

    print()

    (
        results,
        accepted,
        needs_review,
        failed,
    ) = run_batch_inference(
        model,
        images,
        threshold,
    )

    print()

    total = len(
        images
    )

    successful = (
        total - failed
    )

    coverage = (
        accepted / successful
        if successful > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "BATCH INFERENCE COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        f"Total images: {total}"
    )

    print(
        f"Successful predictions: {successful}"
    )

    print(
        f"Errors: {failed}"
    )

    print()

    print(
        "LOCKED OPERATING POLICY"
    )

    print(
        f"Threshold: {threshold:.3f}"
    )

    print()

    print(
        f"Accepted automatically: {accepted}"
    )

    print(
        f"Sent for review: {needs_review}"
    )

    print(
        f"Coverage: {coverage * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_csv(
        results
    )

    save_summary(
        input_directory,
        total,
        accepted,
        needs_review,
        failed,
        threshold,
    )

    print(
        "Results saved:"
    )

    print(
        f"  {CSV_OUTPUT_PATH}"
    )

    print()

    print(
        "Summary saved:"
    )

    print(
        f"  {SUMMARY_OUTPUT_PATH}"
    )

    print()

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()