"""
ai/inspect_errors.py

Inspect the most important errors from the
five-class crop disease classifier.

Focus:
    Blight <-> LeafSpot

This script creates a readable report containing:
    - image path
    - actual class
    - predicted class
    - confidence
"""

import os
import csv
from collections import Counter


# ============================================================
# Configuration
# ============================================================

CSV_PATH = os.path.join(
    "error_analysis",
    "misclassifications.csv",
)

OUTPUT_PATH = os.path.join(
    "error_analysis",
    "error_summary.txt",
)


# ============================================================
# Load errors
# ============================================================

def load_errors():

    if not os.path.exists(
        CSV_PATH
    ):

        raise FileNotFoundError(
            "Error-analysis CSV not found:\n"
            f"{CSV_PATH}\n\n"
            "Run first:\n"
            "python -m ai.error_analysis"
        )

    errors = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            row["confidence"] = float(
                row["confidence"]
            )

            errors.append(
                row
            )

    return errors


# ============================================================
# Analyze confusion pairs
# ============================================================

def analyze_pairs(
    errors
):

    pairs = Counter()

    for error in errors:

        pair = (
            error["actual"],
            error["predicted"],
        )

        pairs[pair] += 1

    return pairs


# ============================================================
# Print important errors
# ============================================================

def print_blght_leafspot_errors(
    errors
):

    important = []

    for error in errors:

        actual = error["actual"]

        predicted = error["predicted"]

        if (
            actual == "Blight"
            and predicted == "LeafSpot"
        ) or (
            actual == "LeafSpot"
            and predicted == "Blight"
        ):

            important.append(
                error
            )

    # --------------------------------------------------------
    # Lowest confidence first
    # --------------------------------------------------------

    important.sort(
        key=lambda item: item["confidence"]
    )

    print(
        "BLIGHT ↔ LEAFSPOT ERRORS"
    )

    print("-" * 80)

    print(
        f"Total: {len(important)}"
    )

    print()

    for index, error in enumerate(
        important,
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"Actual: "
            f"{error['actual']:<10} "
            f"| Predicted: "
            f"{error['predicted']:<10} "
            f"| Confidence: "
            f"{error['confidence'] * 100:6.2f}%"
        )

        print(
            f"    {error['image_path']}"
        )

    print()

    return important


# ============================================================
# Print all confusion pairs
# ============================================================

def print_all_pairs(
    pairs
):

    print(
        "ALL CONFUSION PAIRS"
    )

    print("-" * 80)

    for (
        actual,
        predicted,
    ), count in pairs.most_common():

        print(
            f"{actual:<12}"
            f" -> "
            f"{predicted:<12}"
            f": "
            f"{count:4d}"
        )

    print()


# ============================================================
# Save summary
# ============================================================

def save_summary(
    errors,
    pairs,
    important,
):

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "AgriculturalQuadcopter\n"
        )

        file.write(
            "Five-Class Error Inspection\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Total errors: "
            f"{len(errors)}\n\n"
        )

        file.write(
            "CONFUSION PAIRS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for (
            actual,
            predicted,
        ), count in pairs.most_common():

            file.write(
                f"{actual:<12}"
                f" -> "
                f"{predicted:<12}"
                f": "
                f"{count}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "BLIGHT ↔ LEAFSPOT ERRORS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for index, error in enumerate(
            important,
            start=1,
        ):

            file.write(
                f"{index:02d}. "
                f"Actual={error['actual']} "
                f"| Predicted={error['predicted']} "
                f"| Confidence="
                f"{error['confidence']:.4f}\n"
            )

            file.write(
                f"    {error['image_path']}\n"
            )

    print(
        f"Summary saved:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Visual Error Inspection"
    )

    print("=" * 60)

    print()

    errors = load_errors()

    print(
        f"Total misclassified images: "
        f"{len(errors)}"
    )

    print()

    pairs = analyze_pairs(
        errors
    )

    print_all_pairs(
        pairs
    )

    important = (
        print_blght_leafspot_errors(
            errors
        )
    )

    save_summary(
        errors,
        pairs,
        important,
    )

    print()

    print("=" * 60)

    print(
        "ERROR INSPECTION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()