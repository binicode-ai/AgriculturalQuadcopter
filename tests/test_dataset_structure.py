"""
tests/test_dataset_structure.py

Verify the final crop-disease dataset.
"""

from pathlib import Path


DATASET_DIR = Path(
    "datasets/crop_disease"
)

SPLITS = [
    "train",
    "validation",
    "test",
]

CLASSES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_images(directory):

    return sum(
        1
        for file in directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def main():

    print("=" * 60)
    print(" AgriculturalQuadcopter")
    print(" Dataset Structure Verification")
    print("=" * 60)

    grand_total = 0

    for split in SPLITS:

        print()
        print(split.upper())

        print("-" * 60)

        split_total = 0

        for class_name in CLASSES:

            directory = (
                DATASET_DIR
                / split
                / class_name
            )

            if not directory.exists():

                raise FileNotFoundError(
                    f"Missing directory: "
                    f"{directory}"
                )

            count = count_images(
                directory
            )

            split_total += count

            print(
                f"{class_name:10s}: "
                f"{count:6d}"
            )

        print("-" * 60)

        print(
            f"{'TOTAL':10s}: "
            f"{split_total:6d}"
        )

        grand_total += split_total

    print()
    print("=" * 60)

    print(
        f"TOTAL DATASET IMAGES: "
        f"{grand_total}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()