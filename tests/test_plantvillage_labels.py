"""
tests/test_plantvillage_labels.py

Inspect the original PlantVillage labels before
creating our five-class dataset.
"""

from datasets import load_dataset


print("=" * 60)
print(" PlantVillage Label Inspection")
print("=" * 60)

dataset = load_dataset(
    "GVJahnavi/PlantVillage_dataset",
    split="train"
)

print()
print(f"Number of training images: {len(dataset)}")

print()
print("Dataset columns:")
print(dataset.column_names)

print()
print("Available labels:")
print("=" * 60)

label_feature = dataset.features["label"]

for index, label in enumerate(label_feature.names):
    print(f"{index:3d} : {label}")

print()
print("=" * 60)
print(f"Total classes: {len(label_feature.names)}")
print("=" * 60)