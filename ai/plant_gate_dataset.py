"""
ai/plant_gate_dataset.py

Verification utilities for the Plant/NonPlant
binary dataset used by AgriculturalQuadcopter.
"""

import os


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = os.path.join(
    "datasets",
    "plant_gate"
)

SPLITS = [
    "train",
    "validation",
    "test",
]

CLASSES = [
    "Plant",
    "NonPlant",
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ============================================================
# Count Images
# ============================================================

def count_images(
    directory
):

    if not os.path.exists(
        directory
    ):

        return 0

    count = 0

    for filename in os.listdir(
        directory
    ):

        path = os.path.join(
            directory,
            filename
        )

        if (
            os.path.isfile(path)
            and filename.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ):

            count += 1

    return count


# ============================================================
# Verify Dataset
# ============================================================

def verify_dataset():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Plant Gate Dataset Verification"
    )

    print("=" * 60)

    print()

    print(
        f"Dataset root:"
    )

    print(
        f"  {os.path.abspath(DATASET_ROOT)}"
    )

    print()

    total_images = 0

    errors = []

    # --------------------------------------------------------
    # Check splits
    # --------------------------------------------------------

    for split in SPLITS:

        print(
            split.upper()
        )

        print(
            "-" * 60
        )

        split_total = 0

        split_directory = os.path.join(
            DATASET_ROOT,
            split
        )

        if not os.path.isdir(
            split_directory
        ):

            errors.append(
                f"Missing split: {split}"
            )

            print(
                "  MISSING"
            )

            print()

            continue

        # ----------------------------------------------------
        # Check classes
        # ----------------------------------------------------

        for class_name in CLASSES:

            class_directory = os.path.join(
                split_directory,
                class_name
            )

            if not os.path.isdir(
                class_directory
            ):

                errors.append(
                    f"Missing directory: "
                    f"{split}/{class_name}"
                )

                count = 0

            else:

                count = count_images(
                    class_directory
                )

            print(
                f"{class_name:<12}: "
                f"{count:6d}"
            )

            split_total += count

        print(
            "-" * 60
        )

        print(
            f"{'TOTAL':<12}: "
            f"{split_total:6d}"
        )

        print()

        total_images += split_total

    # ========================================================
    # Final result
    # ========================================================

    print("=" * 60)

    print(
        f"TOTAL DATASET IMAGES: "
        f"{total_images}"
    )

    print()

    if errors:

        print(
            "DATASET VERIFICATION FAILED"
        )

        print()

        for error in errors:

            print(
                f"ERROR: {error}"
            )

        return False

    print(
        "DATASET STRUCTURE VERIFIED"
    )

    print("=" * 60)

    return True


# ============================================================
# Entry Point
# ============================================================

def main():

    verify_dataset()


if __name__ == "__main__":

    main()