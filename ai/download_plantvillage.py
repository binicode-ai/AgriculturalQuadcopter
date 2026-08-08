"""
ai/download_plantvillage.py

Build the AgriculturalQuadcopter five-class
crop disease dataset from PlantVillage.

Target classes:
    Blight
    Healthy
    LeafSpot
    Mildew
    Rust
"""

from pathlib import Path

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "GVJahnavi/PlantVillage_dataset"

OUTPUT_DIR = Path("datasets/crop_disease")


# ============================================================
# PlantVillage -> AgriculturalQuadcopter mapping
# ============================================================

CLASS_MAPPING = {

    # -------------------------
    # Blight
    # -------------------------

    "Potato___Early_blight": "Blight",

    "Potato___Late_blight": "Blight",

    "Corn_(maize)___Northern_Leaf_Blight": "Blight",

    "Tomato___Early_blight": "Blight",

    "Tomato___Late_blight": "Blight",


    # -------------------------
    # Healthy
    # -------------------------

    "Apple___healthy": "Healthy",

    "Blueberry___healthy": "Healthy",

    "Cherry_(including_sour)___healthy": "Healthy",

    "Corn_(maize)___healthy": "Healthy",

    "Grape___healthy": "Healthy",

    "Peach___healthy": "Healthy",

    "Pepper,_bell___healthy": "Healthy",

    "Potato___healthy": "Healthy",

    "Raspberry___healthy": "Healthy",

    "Soybean___healthy": "Healthy",

    "Strawberry___healthy": "Healthy",

    "Tomato___healthy": "Healthy",


    # -------------------------
    # Leaf Spot
    # -------------------------

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "LeafSpot",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "LeafSpot",

    "Tomato___Septoria_leaf_spot":
        "LeafSpot",


    # -------------------------
    # Mildew
    # -------------------------

    "Cherry_(including_sour)___Powdery_mildew":
        "Mildew",

    "Squash___Powdery_mildew":
        "Mildew",


    # -------------------------
    # Rust
    # -------------------------

    "Apple___Cedar_apple_rust":
        "Rust",

    "Corn_(maize)___Common_rust_":
        "Rust",
}


TARGET_CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Directory creation
# ============================================================

def create_directories():

    for split in ["train", "validation", "test"]:

        for class_name in TARGET_CLASSES:

            directory = (
                OUTPUT_DIR
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# Save images
# ============================================================

def process_split(
    dataset_split,
    split_name
):

    label_feature = (
        dataset_split.features["label"]
    )

    counters = {
        class_name: 0
        for class_name in TARGET_CLASSES
    }

    print()
    print("=" * 60)
    print(f"Processing {split_name}")
    print("=" * 60)

    for index, sample in enumerate(
        dataset_split
    ):

        label_id = sample["label"]

        original_label = (
            label_feature.int2str(label_id)
        )

        target_class = CLASS_MAPPING.get(
            original_label
        )

        # Ignore classes that are not part
        # of our five-class problem.
        if target_class is None:
            continue

        image = sample["image"]

        counter = counters[target_class]

        filename = (
            f"{split_name}_"
            f"{counter:06d}.jpg"
        )

        output_path = (

            OUTPUT_DIR
            / split_name
            / target_class
            / filename
        )

        if output_path.exists():

            counters[target_class] += 1

            continue

        image.convert("RGB").save(
            output_path,
            "JPEG",
            quality=95
        )

        counters[target_class] += 1

        if (
            (counter + 1) % 500 == 0
        ):

            print(
                f"{target_class:10s}: "
                f"{counter + 1}"
            )

    print()
    print("Split summary")

    for class_name in TARGET_CLASSES:

        print(
            f"{class_name:10s}: "
            f"{counters[class_name]}"
        )

    return counters


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        " AGRICULTURAL QUADCOPTER"
    )

    print(
        " PlantVillage Dataset Builder"
    )

    print("=" * 60)

    print()

    print(
        "Target classes:"
    )

    for class_name in TARGET_CLASSES:

        print(
            f"  - {class_name}"
        )

    print()

    print(
        "Creating directories..."
    )

    create_directories()

    print()

    print(
        "Loading PlantVillage..."
    )

    dataset = load_dataset(
        DATASET_NAME
    )

    print()

    print(dataset)

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_counts = process_split(
        dataset["train"],
        "train"
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    test_counts = process_split(
        dataset["test"],
        "test"
    )

    print()
    print("=" * 60)
    print(" DATASET BUILD COMPLETE")
    print("=" * 60)

    print()

    print("TRAIN")

    for class_name in TARGET_CLASSES:

        print(
            f"{class_name:10s}: "
            f"{train_counts[class_name]}"
        )

    print()

    print("TEST")

    for class_name in TARGET_CLASSES:

        print(
            f"{class_name:10s}: "
            f"{test_counts[class_name]}"
        )

    print()
    print(
        "Note: validation will be created "
        "in the next step."
    )


if __name__ == "__main__":

    main()