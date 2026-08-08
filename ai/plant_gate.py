"""
ai/plant_gate.py

Plant/non-plant decision layer
for AgriculturalQuadcopter.

This module provides a simple interface that can later
be replaced by a dedicated plant detector or segmentation
model.
"""


# ============================================================
# Configuration
# ============================================================

PLANT_GATE_THRESHOLD = 0.50


# ============================================================
# Plant Gate
# ============================================================

def check_plant_presence(
    plant_score
):
    """
    Decide whether an image should be passed
    to the disease classifier.

    Parameters
    ----------
    plant_score : float
        Estimated probability that the image
        contains a plant.

    Returns
    -------
    bool
        True  -> send to disease classifier
        False -> reject/ignore
    """

    return (
        plant_score
        >= PLANT_GATE_THRESHOLD
    )


# ============================================================
# Decision
# ============================================================

def plant_gate_decision(
    plant_score
):

    if (
        plant_score
        >= PLANT_GATE_THRESHOLD
    ):

        return "PLANT"

    return "NON_PLANT"


# ============================================================
# Test
# ============================================================

def main():

    print("=" * 60)

    print(
        "AgriculturalQuadcopter"
    )

    print(
        "Plant Gate"
    )

    print("=" * 60)

    print()

    test_scores = [
        0.95,
        0.82,
        0.61,
        0.42,
        0.15,
    ]

    for score in test_scores:

        decision = plant_gate_decision(
            score
        )

        print(
            f"Plant score: "
            f"{score:.2f}"
            f" -> "
            f"{decision}"
        )

    print()

    print("=" * 60)

    print(
        "PLANT GATE TEST COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()