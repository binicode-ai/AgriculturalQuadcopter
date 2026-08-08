"""
tests/test_plantvillage_connection.py

Test access to an open PlantVillage dataset mirror.
"""

from datasets import load_dataset


print("=" * 60)
print("Testing PlantVillage dataset connection")
print("=" * 60)

dataset = load_dataset(
    "GVJahnavi/PlantVillage_dataset",
    split="train[:5]"
)

print()
print("Dataset loaded successfully!")

print()
print("Number of samples:")
print(len(dataset))

print()
print("Columns:")
print(dataset.column_names)

print()
print("First sample:")

sample = dataset[0]

print(sample)

print()
print("=" * 60)
print("PlantVillage connection test PASSED")
print("=" * 60)