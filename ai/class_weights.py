"""
ai/class_weights.py

Calculate class weights for the
AgriculturalQuadcopter disease classifier.
"""

import torch

from ai.dataset import (
    create_dataset,
    CLASS_NAMES,
)


def calculate_class_weights(dataset):
    """
    Calculate balanced class weights.

    Weight is proportional to:

        total_samples
        ----------------
        number_of_classes * class_count
    """

    counts = [
        0
        for _ in CLASS_NAMES
    ]

    for _, label in dataset.samples:

        counts[label] += 1

    total_samples = sum(counts)

    num_classes = len(
        CLASS_NAMES
    )

    weights = []

    for count in counts:

        weight = (
            total_samples
            /
            (
                num_classes
                * count
            )
        )

        weights.append(weight)

    return (
        counts,
        torch.tensor(
            weights,
            dtype=torch.float32
        )
    )


def main():

    print("=" * 60)
    print("AgriculturalQuadcopter")
    print("Class Weight Analysis")
    print("=" * 60)

    print()

    dataset = create_dataset(
        "train"
    )

    counts, weights = (
        calculate_class_weights(
            dataset
        )
    )

    print(
        "Training class distribution:"
    )

    print()

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{index}  "
            f"{name:10s} "
            f"count={counts[index]:6d} "
            f"weight={weights[index]:.4f}"
        )

    print()

    print(
        "Class weights:"
    )

    print(weights)

    print()

    print("=" * 60)
    print("Class weight calculation PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()