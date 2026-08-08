"""
ai/dataset.py

AgriculturalQuadcopter crop disease dataset loader.

Five classes:

    0 -> Blight
    1 -> Healthy
    2 -> LeafSpot
    3 -> Mildew
    4 -> Rust
"""

from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "crop_disease"
)


# ============================================================
# Classes
# ============================================================

CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# Supported image formats
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Dataset
# ============================================================

class CropDiseaseDataset(Dataset):
    """
    Explicit five-class crop disease dataset.

    We deliberately do not use ImageFolder here because
    we want complete control over class discovery.
    """

    def __init__(
        self,
        split,
        transform=None
    ):

        valid_splits = {
            "train",
            "validation",
            "test",
        }

        if split not in valid_splits:

            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of: "
                f"{sorted(valid_splits)}"
            )

        self.split = split

        self.transform = transform

        self.root = (
            DATASET_ROOT
            / split
        )

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset split does not exist:\n"
                f"{self.root}"
            )

        if not self.root.is_dir():

            raise NotADirectoryError(
                f"Dataset split is not a directory:\n"
                f"{self.root}"
            )

        self.classes = CLASS_NAMES.copy()

        self.class_to_idx = (
            CLASS_TO_INDEX.copy()
        )

        self.samples = []

        self._load_samples()

    # ========================================================
    # Load image paths
    # ========================================================

    def _load_samples(self):

        print(
            f"Scanning {self.split}: "
            f"{self.root}"
        )

        for class_name in CLASS_NAMES:

            class_directory = (
                self.root
                / class_name
            )

            if not class_directory.exists():

                raise FileNotFoundError(
                    f"Missing class directory:\n"
                    f"{class_directory}"
                )

            if not class_directory.is_dir():

                raise NotADirectoryError(
                    f"Class path is not a directory:\n"
                    f"{class_directory}"
                )

            class_index = (
                self.class_to_idx[
                    class_name
                ]
            )

            image_files = sorted(
                [
                    path
                    for path in
                    class_directory.iterdir()
                    if (
                        path.is_file()
                        and
                        path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                ]
            )

            for image_path in image_files:

                self.samples.append(
                    (
                        image_path,
                        class_index
                    )
                )

            print(
                f"  {class_name:10s}: "
                f"{len(image_files):6d}"
            )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.root}"
            )

    # ========================================================
    # Number of samples
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Get one sample
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            ) from error

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# Factory function
# ============================================================

def create_dataset(
    split,
    transform=None
):
    """
    Create a CropDiseaseDataset.
    """

    return CropDiseaseDataset(
        split=split,
        transform=transform
    )


# ============================================================
# Verify class mapping
# ============================================================

def verify_classes(dataset):

    expected = CLASS_NAMES

    actual = dataset.classes

    if actual != expected:

        raise RuntimeError(
            "\nUnexpected class ordering.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    expected_mapping = {
        "Blight": 0,
        "Healthy": 1,
        "LeafSpot": 2,
        "Mildew": 3,
        "Rust": 4,
    }

    if dataset.class_to_idx != expected_mapping:

        raise RuntimeError(
            "\nUnexpected class mapping.\n"
            f"Expected: {expected_mapping}\n"
            f"Actual:   {dataset.class_to_idx}"
        )

    return True


# ============================================================
# Count classes
# ============================================================

def get_class_counts(dataset):

    counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    for _, label in dataset.samples:

        class_name = (
            CLASS_NAMES[label]
        )

        counts[class_name] += 1

    return counts


# ============================================================
# Print summary
# ============================================================

def print_dataset_summary(
    split,
    dataset
):

    counts = get_class_counts(
        dataset
    )

    print()

    print(
        split.upper()
    )

    print("-" * 60)

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:10s}: "
            f"{counts[class_name]:6d}"
        )

    print("-" * 60)

    print(
        f"{'TOTAL':10s}: "
        f"{len(dataset):6d}"
    )


# ============================================================
# Test one image
# ============================================================

def test_sample(dataset):

    image, label = dataset[0]

    print()

    print(
        "Sample test:"
    )

    print(
        f"  Image type : "
        f"{type(image)}"
    )

    if isinstance(image, torch.Tensor):

        print(
            f"  Image shape: "
            f"{tuple(image.shape)}"
        )

    print(
        f"  Label      : "
        f"{label}"
    )

    print(
        f"  Class      : "
        f"{CLASS_NAMES[label]}"
    )


# ============================================================
# Main diagnostic
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter Dataset"
    )

    print("=" * 60)

    print()

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print()

    print(
        f"Dataset root:\n"
        f"{DATASET_ROOT}"
    )

    print()

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset root does not exist:\n"
            f"{DATASET_ROOT}"
        )

    print(
        "Class mapping:"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index} -> {name}"
        )

    total = 0

    # --------------------------------------------------------
    # Check all splits
    # --------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test",
    ]:

        dataset = create_dataset(
            split
        )

        verify_classes(
            dataset
        )

        print_dataset_summary(
            split,
            dataset
        )

        total += len(dataset)

        # Test one sample
        test_sample(
            dataset
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        f"TOTAL DATASET IMAGES: "
        f"{total}"
    )

    print(
        f"NUMBER OF CLASSES: "
        f"{NUM_CLASSES}"
    )

    print()

    print(
        "Dataset verification PASSED"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()