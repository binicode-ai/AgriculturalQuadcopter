"""
ai/augmentation.py

Image preprocessing and augmentation pipeline
for the AgriculturalQuadcopter crop disease classifier.
"""

from torchvision import transforms


# ============================================================
# Image configuration
# ============================================================

IMAGE_SIZE = 224


# ImageNet normalization.
# This is appropriate for the pretrained CNN models
# we will use later.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# Training transforms
# ============================================================

def get_train_transform():

    transform = transforms.Compose([

        # Resize the shortest side.
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        # Randomly flip leaves horizontally.
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        # Small rotations simulate
        # different camera orientations.
        transforms.RandomRotation(
            degrees=15
        ),

        # Simulate natural lighting/color
        # differences in agricultural images.
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # Convert PIL image -> Tensor.
        transforms.ToTensor(),

        # Normalize using ImageNet statistics.
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    return transform


# ============================================================
# Validation transforms
# ============================================================

def get_validation_transform():

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    return transform


# ============================================================
# Test transforms
# ============================================================

def get_test_transform():

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    return transform


# ============================================================
# Diagnostic
# ============================================================

def main():

    print("=" * 60)
    print("AgriculturalQuadcopter")
    print("Image Transformation Pipeline")
    print("=" * 60)

    print()

    print(
        f"Image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print()

    print("Training transformations:")
    print(
        "  - Resize"
    )
    print(
        "  - Random horizontal flip"
    )
    print(
        "  - Random rotation"
    )
    print(
        "  - Color jitter"
    )
    print(
        "  - ToTensor"
    )
    print(
        "  - ImageNet normalization"
    )

    print()

    print("Validation transformations:")
    print(
        "  - Resize"
    )
    print(
        "  - ToTensor"
    )
    print(
        "  - ImageNet normalization"
    )

    print()

    print("Test transformations:")
    print(
        "  - Resize"
    )
    print(
        "  - ToTensor"
    )
    print(
        "  - ImageNet normalization"
    )

    print()

    # Actually construct each pipeline.
    train_transform = (
        get_train_transform()
    )

    validation_transform = (
        get_validation_transform()
    )

    test_transform = (
        get_test_transform()
    )

    print(
        "Training pipeline:"
    )
    print(train_transform)

    print()

    print(
        "Validation pipeline:"
    )
    print(validation_transform)

    print()

    print(
        "Test pipeline:"
    )
    print(test_transform)

    print()

    print("=" * 60)
    print("Augmentation pipeline verification PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()