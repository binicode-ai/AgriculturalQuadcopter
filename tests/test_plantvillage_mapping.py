"""
tests/test_plantvillage_mapping.py

Count PlantVillage samples that belong to our
five AgriculturalQuadcopter disease classes.
"""

from collections import Counter

from datasets import load_dataset


TARGET_CLASSES = {
    "Blight": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
    ],

    "Healthy": [
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___healthy",
        "Grape___healthy",
        "Peach___healthy",
        "Pepper,_bell___healthy",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Strawberry___healthy",
        "Tomato___healthy",
    ],

    "LeafSpot": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Tomato___Septoria_leaf_spot",
    ],

    "Mildew": [
        "Cherry_(including_sour)___Powdery_mildew",
        "Squash___Powdery_mildew",
    ],

    "Rust": [
        "Apple___Cedar_apple_rust",
        "Corn_(maize)___Common_rust_",
    ],
}


def build_reverse_mapping():

    mapping = {}

    for target_class, labels in TARGET_CLASSES.items():

        for label in labels:

            mapping[label] = target_class

    return mapping


def main():

    print("=" * 60)
    print(" PlantVillage Five-Class Mapping Analysis")
    print("=" * 60)

    dataset = load_dataset(
        "GVJahnavi/PlantVillage_dataset",
        split="train"
    )

    label_feature = dataset.features["label"]

    reverse_mapping = build_reverse_mapping()

    counts = Counter()

    excluded = 0

    for label_id in dataset["label"]:

        original_label = (
            label_feature.int2str(label_id)
        )

        target_class = reverse_mapping.get(
            original_label
        )

        if target_class is None:

            excluded += 1

        else:

            counts[target_class] += 1

    print()

    print("Selected classes")
    print("-" * 60)

    total_selected = 0

    for target_class in [
        "Blight",
        "Healthy",
        "LeafSpot",
        "Mildew",
        "Rust",
    ]:

        count = counts[target_class]

        total_selected += count

        print(
            f"{target_class:10s}: {count:6d}"
        )

    print("-" * 60)

    print(
        f"{'TOTAL':10s}: {total_selected:6d}"
    )

    print()

    print(
        f"Excluded images: {excluded}"
    )

    print()

    print("=" * 60)
    print("Mapping analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()