"""
tests/test_training_dataset.py

Verify that the AgriculturalQuadcopter
dataset works correctly with the training
and validation transforms.
"""

import torch

from ai.dataset import (
    create_dataset,
    CLASS_NAMES,
)

from ai.augmentation import (
    get_train_transform,
    get_validation_transform,
)


def test_dataset(
    split,
    transform
):

    print()
    print("=" * 60)

    print(
        f"Testing {split.upper()} dataset"
    )

    print("=" * 60)

    dataset = create_dataset(
        split,
        transform=transform
    )

    print()

    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Get first image
    # --------------------------------------------------------

    image, label = dataset[0]

    print()

    print(
        f"Tensor type: "
        f"{type(image)}"
    )

    print(
        f"Tensor shape: "
        f"{tuple(image.shape)}"
    )

    print(
        f"Tensor dtype: "
        f"{image.dtype}"
    )

    print(
        f"Label: "
        f"{label}"
    )

    print(
        f"Class: "
        f"{CLASS_NAMES[label]}"
    )

    # --------------------------------------------------------
    # Verify tensor
    # --------------------------------------------------------

    assert isinstance(
        image,
        torch.Tensor
    )

    assert image.shape == (
        3,
        224,
        224
    )

    assert image.dtype == torch.float32

    # --------------------------------------------------------
    # Verify label
    # --------------------------------------------------------

    assert (
        0 <= label < len(CLASS_NAMES)
    )

    print()

    print(
        "Tensor verification PASSED"
    )


def main():

    print()
    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Training Dataset Integration Test"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    test_dataset(
        "train",
        get_train_transform()
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    test_dataset(
        "validation",
        get_validation_transform()
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "TRAINING DATASET INTEGRATION PASSED"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()