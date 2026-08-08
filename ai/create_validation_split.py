"""
ai/create_validation_split.py

Create a validation set from the existing training set.

The official test set is NOT modified.

Validation ratio:
    15%

Random seed:
    42
"""

from pathlib import Path
import random
import shutil


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    "datasets/crop_disease"
)

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]

VALIDATION_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Get image files
# ============================================================

def get_images(directory):

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


# ============================================================
# Create validation directories
# ============================================================

def create_directories():

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for class_name in CLASSES:

        (
            VALIDATION_DIR
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# Check whether validation already exists
# ============================================================

def validation_already_created():

    total = 0

    for class_name in CLASSES:

        directory = (
            VALIDATION_DIR
            / class_name
        )

        if directory.exists():

            total += len(
                get_images(directory)
            )

    return total > 0


# ============================================================
# Split one class
# ============================================================

def split_class(
    class_name,
    rng
):

    train_class_dir = (
        TRAIN_DIR
        / class_name
    )

    validation_class_dir = (
        VALIDATION_DIR
        / class_name
    )

    images = get_images(
        train_class_dir
    )

    if len(images) == 0:

        print(
            f"WARNING: "
            f"{class_name} has no images."
        )

        return 0

    shuffled = images.copy()

    rng.shuffle(shuffled)

    validation_count = int(
        len(shuffled)
        * VALIDATION_RATIO
    )

    validation_images = (
        shuffled[:validation_count]
    )

    print()
    print(
        f"{class_name}"
    )

    print(
        f"  Training images before split : "
        f"{len(images)}"
    )

    print(
        f"  Validation images            : "
        f"{len(validation_images)}"
    )

    print(
        f"  Training images after split  : "
        f"{len(images) - len(validation_images)}"
    )

    for image_path in validation_images:

        destination = (
            validation_class_dir
            / image_path.name
        )

        shutil.move(
            str(image_path),
            str(destination)
        )

    return len(validation_images)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(" AgriculturalQuadcopter")
    print(" Validation Dataset Builder")
    print("=" * 60)

    print()

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"Training directory not found: "
            f"{TRAIN_DIR}"
        )

    create_directories()

    if validation_already_created():

        print(
            "Validation data already exists."
        )

        print(
            "No changes were made."
        )

        print()
        print(
            "If you want to recreate it, "
            "restore the validation images "
            "to train first."
        )

        return

    rng = random.Random(
        RANDOM_SEED
    )

    total_validation = 0

    for class_name in CLASSES:

        count = split_class(
            class_name,
            rng
        )

        total_validation += count

    print()
    print("=" * 60)

    print(
        "Validation split complete."
    )

    print(
        f"Total validation images: "
        f"{total_validation}"
    )

    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO:.0%}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()