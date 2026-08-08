"""
ai/cnn.py

CNN disease classifier for the AgriculturalQuadcopter project.

Uses MobileNetV3-Small as the feature extractor and replaces
the final classifier with a 5-class agricultural disease head.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# Project configuration
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Blight",
    "Healthy",
    "LeafSpot",
    "Mildew",
    "Rust",
]


# ============================================================
# Model builder
# ============================================================

def create_model(
    num_classes=NUM_CLASSES,
    pretrained=False,
):
    """
    Create the AgriculturalQuadcopter disease classifier.

    Parameters
    ----------
    num_classes : int
        Number of output disease classes.

    pretrained : bool
        If True, load ImageNet pretrained weights.

    Returns
    -------
    torch.nn.Module
        MobileNetV3-Small classifier.
    """

    if pretrained:

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
        )

    else:

        weights = None

    model = mobilenet_v3_small(
        weights=weights
    )

    # --------------------------------------------------------
    # Replace the original ImageNet classifier.
    #
    # MobileNetV3-Small normally produces 1000 classes.
    # Our project has only 5.
    # --------------------------------------------------------

    input_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    return model


# ============================================================
# Parameter information
# ============================================================

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# Freeze feature extractor
# ============================================================

def freeze_backbone(model):
    """
    Freeze the MobileNet feature extractor.

    The new classifier remains trainable.
    """

    for parameter in model.features.parameters():

        parameter.requires_grad = False

    return model


# ============================================================
# Model summary
# ============================================================

def print_model_info(model):

    total, trainable = (
        count_parameters(model)
    )

    print()
    print(
        "Total parameters:     "
        f"{total:,}"
    )

    print(
        "Trainable parameters: "
        f"{trainable:,}"
    )

    print()

    print(
        "Output classes:"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index} : {name}"
        )


# ============================================================
# Forward-pass verification
# ============================================================

def verify_forward_pass(model):

    print()
    print(
        "Running forward-pass test..."
    )

    # One RGB image.
    #
    # Shape:
    # batch = 1
    # channels = 3
    # height = 224
    # width = 224

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    model.eval()

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        "Input shape:  "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        "Output shape: "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # Verify expected output.
    # --------------------------------------------------------

    assert output.shape == (
        1,
        NUM_CLASSES
    )

    assert torch.isfinite(
        output
    ).all()

    print()
    print(
        "Forward-pass verification PASSED"
    )

    return output


# ============================================================
# Main diagnostic
# ============================================================

def main():

    print("=" * 60)
    print(
        "AgriculturalQuadcopter"
    )
    print(
        "CNN Disease Classifier"
    )
    print("=" * 60)

    print()

    print(
        "Architecture: MobileNetV3-Small"
    )

    print(
        "Input: 3 × 224 × 224"
    )

    print(
        "Output classes: 5"
    )

    print(
        "Pretrained weights: No"
    )

    print()

    # --------------------------------------------------------
    # Create model without downloading weights.
    #
    # This first test checks the architecture itself.
    # --------------------------------------------------------

    model = create_model(
        pretrained=False
    )

    print_model_info(
        model
    )

    # --------------------------------------------------------
    # Verify forward pass.
    # --------------------------------------------------------

    verify_forward_pass(
        model
    )

    print()

    print("=" * 60)
    print(
        "CNN MODEL VERIFICATION PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()

def create_transfer_learning_model():
    """
    Create a MobileNetV3-Small model using
    ImageNet pretrained weights.

    The feature extractor is frozen.
    Only the 5-class classifier is trainable.
    """

    model = create_model(
        pretrained=True
    )

    model = freeze_backbone(
        model
    )

    return model    