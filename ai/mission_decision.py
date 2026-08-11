"""
ai/mission_decision.py

AgriculturalQuadcopter
Mission-Level Disease Decision System

Converts individual AI predictions into a
mission-level agricultural observation.

Five classes:

    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

Decision strategy:

    1. Ignore low-confidence predictions.
    2. Count accepted disease observations.
    3. Calculate average confidence per class.
    4. Determine dominant class.
    5. Require sufficient evidence before
       declaring a mission-level disease.
"""

import os
from collections import defaultdict

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

IMAGE_SIZE = 224

CONFIDENCE_THRESHOLD = 0.95

MIN_ACCEPTED_OBSERVATIONS = 3

DOMINANCE_RATIO = 0.60


CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Image transformation
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
# Model
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

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
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# Single-image prediction
# ============================================================

def predict_image(
    model,
    image_path,
    transform,
):

    image = Image.open(
        image_path
    ).convert("RGB")

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

    class_index = (
        prediction.item()
    )

    confidence_value = (
        confidence.item()
    )

    predicted_class = CLASS_NAMES[
        class_index
    ]

    return (
        predicted_class,
        confidence_value,
    )


# ============================================================
# Mission analyzer
# ============================================================

class MissionAnalyzer:

    def __init__(self):

        self.observations = []

        self.class_counts = defaultdict(
            int
        )

        self.class_confidences = defaultdict(
            list
        )

        self.review_count = 0

    # --------------------------------------------------------
    # Add observation
    # --------------------------------------------------------

    def add_observation(
        self,
        image_path,
        predicted_class,
        confidence,
    ):

        accepted = (
            confidence
            >= CONFIDENCE_THRESHOLD
        )

        observation = {

            "image_path":
                image_path,

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "accepted":
                accepted,
        }

        self.observations.append(
            observation
        )

        if accepted:

            self.class_counts[
                predicted_class
            ] += 1

            self.class_confidences[
                predicted_class
            ].append(
                confidence
            )

        else:

            self.review_count += 1

    # --------------------------------------------------------
    # Total observations
    # --------------------------------------------------------

    def total_observations(self):

        return len(
            self.observations
        )

    # --------------------------------------------------------
    # Accepted observations
    # --------------------------------------------------------

    def accepted_observations(self):

        return sum(
            self.class_counts.values()
        )

    # --------------------------------------------------------
    # Determine mission decision
    # --------------------------------------------------------

    def decide(self):

        accepted = (
            self.accepted_observations()
        )

        if accepted < MIN_ACCEPTED_OBSERVATIONS:

            return {

                "decision":
                    "INSUFFICIENT-EVIDENCE",

                "dominant_class":
                    None,

                "dominance_ratio":
                    0.0,

                "average_confidence":
                    0.0,
            }

        dominant_class = max(
            self.class_counts,
            key=self.class_counts.get,
        )

        dominant_count = (
            self.class_counts[
                dominant_class
            ]
        )

        dominance_ratio = (
            dominant_count
            / accepted
        )

        confidences = (
            self.class_confidences[
                dominant_class
            ]
        )

        average_confidence = (
            sum(confidences)
            / len(confidences)
        )

        if (
            dominance_ratio
            >= DOMINANCE_RATIO
        ):

            decision = (
                "DISEASE-CONFIRMED"
            )

        else:

            decision = (
                "MIXED-OBSERVATION"
            )

        return {

            "decision":
                decision,

            "dominant_class":
                dominant_class,

            "dominance_ratio":
                dominance_ratio,

            "average_confidence":
                average_confidence,
        }

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    def print_summary(self):

        result = self.decide()

        print()

        print(
            "=" * 70
        )

        print(
            "MISSION-LEVEL AI DECISION"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Total observations: "
            f"{self.total_observations()}"
        )

        print(
            f"Accepted observations: "
            f"{self.accepted_observations()}"
        )

        print(
            f"Needs review: "
            f"{self.review_count}"
        )

        print()

        print(
            "Disease observation counts:"
        )

        for class_name in CLASS_NAMES:

            count = (
                self.class_counts[
                    class_name
                ]
            )

            print(
                f"  {class_name:<10}: "
                f"{count}"
            )

        print()

        print(
            f"Mission decision:"
        )

        print(
            f"  {result['decision']}"
        )

        if (
            result["dominant_class"]
            is not None
        ):

            print()

            print(
                f"Dominant disease:"
            )

            print(
                f"  {result['dominant_class']}"
            )

            print()

            print(
                f"Dominance:"
            )

            print(
                f"  "
                f"{result['dominance_ratio'] * 100:.2f}%"
            )

            print()

            print(
                f"Average confidence:"
            )

            print(
                f"  "
                f"{result['average_confidence'] * 100:.2f}%"
            )

        print()

        print(
            "=" * 70
        )


# ============================================================
# Find images
# ============================================================

def collect_images(
    directory
):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )

    images = []

    for root, _, files in os.walk(
        directory
    ):

        for filename in files:

            if filename.lower().endswith(
                extensions
            ):

                images.append(
                    os.path.join(
                        root,
                        filename,
                    )
                )

    images.sort()

    return images


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Mission-Level Decision System"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Minimum accepted observations: "
        f"{MIN_ACCEPTED_OBSERVATIONS}"
    )

    print(
        f"Required dominance ratio: "
        f"{DOMINANCE_RATIO:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Directory argument
    # --------------------------------------------------------

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python -m ai.mission_decision "
            "\"datasets/crop_disease/test/Blight\""
        )

        print()

        return

    image_directory = sys.argv[1]

    if not os.path.isdir(
        image_directory
    ):

        raise FileNotFoundError(
            "Image directory not found:\n"
            f"{image_directory}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading targeted model..."
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )

    print()

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = collect_images(
        image_directory
    )

    if not images:

        raise RuntimeError(
            "No supported images found."
        )

    print(
        f"Images found: "
        f"{len(images)}"
    )

    print()

    # --------------------------------------------------------
    # Analyzer
    # --------------------------------------------------------

    analyzer = MissionAnalyzer()

    transform = create_transform()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    print(
        "Running mission analysis..."
    )

    print()

    for index, image_path in enumerate(
        images
    ):

        (
            predicted_class,
            confidence,
        ) = predict_image(
            model,
            image_path,
            transform,
        )

        analyzer.add_observation(
            image_path,
            predicted_class,
            confidence,
        )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(images)}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    analyzer.print_summary()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()