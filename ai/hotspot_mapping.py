"""
ai/hotspot_mapping.py

AgriculturalQuadcopter
Disease Hotspot Mapping

Converts AI disease predictions plus GPS coordinates
into field-level disease hotspots.

Five classes:

    0 = Blight
    1 = Healthy
    2 = LeafSpot
    3 = Mildew
    4 = Rust

This lesson uses simulated GPS coordinates.

Later these coordinates will come directly from
the drone flight controller / GPS subsystem.
"""

import csv
import math
import os
import random
from collections import defaultdict
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

OUTPUT_DIRECTORY = "hotspot_analysis"

OBSERVATION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "field_observations.csv",
)

HOTSPOT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "disease_hotspots.csv",
)

IMAGE_SIZE = 224

CONFIDENCE_THRESHOLD = 0.95

# Approximate hotspot radius in meters.
#
# Observations within this distance are considered
# part of the same local field area.

HOTSPOT_RADIUS_METERS = 15.0

# Minimum number of accepted disease observations
# required to call an area a hotspot.

MIN_HOTSPOT_OBSERVATIONS = 3

# Minimum fraction of observations that must belong
# to the dominant disease.

HOTSPOT_DOMINANCE_RATIO = 0.60


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
# Load model
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
# Predict image
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

    class_index = prediction.item()

    predicted_class = CLASS_NAMES[
        class_index
    ]

    confidence_value = confidence.item()

    return (
        predicted_class,
        confidence_value,
    )


# ============================================================
# Geographic distance
# ============================================================

def distance_meters(
    latitude1,
    longitude1,
    latitude2,
    longitude2,
):

    """
    Approximate distance between two GPS points.

    Accurate enough for small agricultural field areas.
    """

    earth_radius = 6371000.0

    lat1 = math.radians(
        latitude1
    )

    lat2 = math.radians(
        latitude2
    )

    delta_lat = math.radians(
        latitude2 - latitude1
    )

    delta_lon = math.radians(
        longitude2 - longitude1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return (
        earth_radius * c
    )


# ============================================================
# Observation
# ============================================================

class FieldObservation:

    def __init__(
        self,
        timestamp,
        image_path,
        latitude,
        longitude,
        altitude,
        disease,
        confidence,
    ):

        self.timestamp = timestamp

        self.image_path = image_path

        self.latitude = latitude

        self.longitude = longitude

        self.altitude = altitude

        self.disease = disease

        self.confidence = confidence

        self.accepted = (
            confidence
            >= CONFIDENCE_THRESHOLD
        )


# ============================================================
# Hotspot
# ============================================================

class DiseaseHotspot:

    def __init__(self):

        self.observations = []

    # --------------------------------------------------------
    # Add observation
    # --------------------------------------------------------

    def add(
        self,
        observation,
    ):

        self.observations.append(
            observation
        )

    # --------------------------------------------------------
    # Number of observations
    # --------------------------------------------------------

    def count(self):

        return len(
            self.observations
        )

    # --------------------------------------------------------
    # Center latitude
    # --------------------------------------------------------

    def center_latitude(self):

        if not self.observations:

            return 0.0

        return (
            sum(
                observation.latitude
                for observation
                in self.observations
            )
            / self.count()
        )

    # --------------------------------------------------------
    # Center longitude
    # --------------------------------------------------------

    def center_longitude(self):

        if not self.observations:

            return 0.0

        return (
            sum(
                observation.longitude
                for observation
                in self.observations
            )
            / self.count()
        )

    # --------------------------------------------------------
    # Disease counts
    # --------------------------------------------------------

    def disease_counts(self):

        counts = defaultdict(
            int
        )

        for observation in (
            self.observations
        ):

            counts[
                observation.disease
            ] += 1

        return counts

    # --------------------------------------------------------
    # Dominant disease
    # --------------------------------------------------------

    def dominant_disease(self):

        counts = self.disease_counts()

        if not counts:

            return None

        return max(
            counts,
            key=counts.get,
        )

    # --------------------------------------------------------
    # Dominance ratio
    # --------------------------------------------------------

    def dominance_ratio(self):

        if not self.observations:

            return 0.0

        dominant = (
            self.dominant_disease()
        )

        counts = (
            self.disease_counts()
        )

        return (
            counts[dominant]
            / self.count()
        )

    # --------------------------------------------------------
    # Average confidence
    # --------------------------------------------------------

    def average_confidence(self):

        if not self.observations:

            return 0.0

        return (
            sum(
                observation.confidence
                for observation
                in self.observations
            )
            / self.count()
        )


# ============================================================
# Group observations into hotspots
# ============================================================

def build_hotspots(
    observations,
):

    hotspots = []

    for observation in observations:

        # ----------------------------------------------------
        # Ignore low-confidence predictions.
        # ----------------------------------------------------

        if not observation.accepted:

            continue

        assigned = False

        # ----------------------------------------------------
        # Find nearby existing hotspot.
        # ----------------------------------------------------

        for hotspot in hotspots:

            center_lat = (
                hotspot.center_latitude()
            )

            center_lon = (
                hotspot.center_longitude()
            )

            distance = distance_meters(
                observation.latitude,
                observation.longitude,
                center_lat,
                center_lon,
            )

            if (
                distance
                <= HOTSPOT_RADIUS_METERS
            ):

                hotspot.add(
                    observation
                )

                assigned = True

                break

        # ----------------------------------------------------
        # Create new hotspot.
        # ----------------------------------------------------

        if not assigned:

            hotspot = (
                DiseaseHotspot()
            )

            hotspot.add(
                observation
            )

            hotspots.append(
                hotspot
            )

    return hotspots


# ============================================================
# Filter actual disease hotspots
# ============================================================

def identify_disease_hotspots(
    hotspots,
):

    disease_hotspots = []

    for hotspot in hotspots:

        if (
            hotspot.count()
            < MIN_HOTSPOT_OBSERVATIONS
        ):

            continue

        dominant = (
            hotspot.dominant_disease()
        )

        if dominant == "Healthy":

            continue

        if (
            hotspot.dominance_ratio()
            < HOTSPOT_DOMINANCE_RATIO
        ):

            continue

        disease_hotspots.append(
            hotspot
        )

    return disease_hotspots


# ============================================================
# Save observations
# ============================================================

def save_observations(
    observations,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    with open(
        OBSERVATION_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "timestamp",
            "image_path",
            "latitude",
            "longitude",
            "altitude_m",
            "disease",
            "confidence",
            "accepted",
        ])

        for observation in (
            observations
        ):

            writer.writerow([

                observation.timestamp,

                observation.image_path,

                f"{observation.latitude:.8f}",

                f"{observation.longitude:.8f}",

                f"{observation.altitude:.2f}",

                observation.disease,

                f"{observation.confidence:.6f}",

                observation.accepted,
            ])


# ============================================================
# Save hotspots
# ============================================================

def save_hotspots(
    hotspots,
):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    with open(
        HOTSPOT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([

            "hotspot_id",

            "latitude",

            "longitude",

            "observation_count",

            "dominant_disease",

            "dominance_ratio",

            "average_confidence",
        ])

        for index, hotspot in enumerate(
            hotspots,
            start=1,
        ):

            writer.writerow([

                index,

                f"{hotspot.center_latitude():.8f}",

                f"{hotspot.center_longitude():.8f}",

                hotspot.count(),

                hotspot.dominant_disease(),

                f"{hotspot.dominance_ratio():.4f}",

                f"{hotspot.average_confidence():.4f}",
            ])


# ============================================================
# Print hotspot report
# ============================================================

def print_hotspot_report(
    hotspots,
):

    print()

    print(
        "=" * 70
    )

    print(
        "DISEASE HOTSPOT REPORT"
    )

    print(
        "=" * 70
    )

    print()

    if not hotspots:

        print(
            "No disease hotspots detected."
        )

        print()

        return

    print(
        f"Disease hotspots detected: "
        f"{len(hotspots)}"
    )

    print()

    for index, hotspot in enumerate(
        hotspots,
        start=1,
    ):

        print(
            f"Hotspot #{index}"
        )

        print(
            "-" * 40
        )

        print(
            f"  Latitude: "
            f"{hotspot.center_latitude():.8f}"
        )

        print(
            f"  Longitude: "
            f"{hotspot.center_longitude():.8f}"
        )

        print(
            f"  Observations: "
            f"{hotspot.count()}"
        )

        print(
            f"  Dominant disease: "
            f"{hotspot.dominant_disease()}"
        )

        print(
            f"  Dominance: "
            f"{hotspot.dominance_ratio() * 100:.2f}%"
        )

        print(
            f"  Average confidence: "
            f"{hotspot.average_confidence() * 100:.2f}%"
        )

        print()


# ============================================================
# Generate simulated GPS coordinates
# ============================================================

def generate_simulated_position(
    index,
):

    """
    Temporary GPS simulator.

    This will eventually be replaced with
    real GPS telemetry.

    The simulated flight moves gradually
    through a field.
    """

    base_latitude = 40.000000

    base_longitude = -74.000000

    row = index // 20

    column = index % 20

    latitude_offset = (
        row * 0.00010
    )

    longitude_offset = (
        column * 0.00010
    )

    return (
        base_latitude
        + latitude_offset,

        base_longitude
        + longitude_offset,
    )


# ============================================================
# Main
# ============================================================

def main():

    import sys

    print(
        "=" * 70
    )

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Disease Hotspot Mapping"
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
        f"Hotspot radius: "
        f"{HOTSPOT_RADIUS_METERS:.1f} meters"
    )

    print(
        f"Minimum hotspot observations: "
        f"{MIN_HOTSPOT_OBSERVATIONS}"
    )

    print()

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python -m ai.hotspot_mapping "
            "\"datasets/crop_disease/test\""
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
    # Find images
    # --------------------------------------------------------

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )

    image_paths = []

    for root, _, files in os.walk(
        image_directory
    ):

        for filename in files:

            if filename.lower().endswith(
                extensions
            ):

                image_paths.append(
                    os.path.join(
                        root,
                        filename,
                    )
                )

    image_paths.sort()

    if not image_paths:

        raise RuntimeError(
            "No images found."
        )

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    print()

    transform = create_transform()

    observations = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    print(
        "Running AI + GPS observation simulation..."
    )

    print()

    for index, image_path in enumerate(
        image_paths
    ):

        (
            disease,
            confidence,
        ) = predict_image(
            model,
            image_path,
            transform,
        )

        (
            latitude,
            longitude,
        ) = generate_simulated_position(
            index
        )

        altitude = 10.0

        timestamp = (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        )

        observation = (
            FieldObservation(
                timestamp,
                image_path,
                latitude,
                longitude,
                altitude,
                disease,
                confidence,
            )
        )

        observations.append(
            observation
        )

        if (
            index + 1
        ) % 100 == 0:

            accepted = sum(
                observation.accepted
                for observation
                in observations
            )

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(image_paths)}"
                f" | Accepted: "
                f"{accepted}"
            )

    # --------------------------------------------------------
    # Save raw observations
    # --------------------------------------------------------

    save_observations(
        observations
    )

    # --------------------------------------------------------
    # Build hotspots
    # --------------------------------------------------------

    print()

    print(
        "Building geographic hotspots..."
    )

    all_hotspots = build_hotspots(
        observations
    )

    disease_hotspots = (
        identify_disease_hotspots(
            all_hotspots
        )
    )

    # --------------------------------------------------------
    # Save hotspot data
    # --------------------------------------------------------

    save_hotspots(
        disease_hotspots
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_hotspot_report(
        disease_hotspots
    )

    print(
        "Output files:"
    )

    print(
        f"  {OBSERVATION_FILE}"
    )

    print(
        f"  {HOTSPOT_FILE}"
    )

    print()

    print(
        "=" * 70
    )

    print(
        "HOTSPOT MAPPING COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()