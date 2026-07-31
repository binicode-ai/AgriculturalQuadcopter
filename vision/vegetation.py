"""
vision/vegetation.py

Vegetation detection using the Excess Green (ExG) index.

Author: Biniyam Samuel
"""

import cv2
import numpy as np


class VegetationDetector:

    def __init__(self):
        pass

    # -------------------------------------------------
    # Compute Excess Green Index
    # -------------------------------------------------

    def compute_exg(self, image):

        if image is None:
            raise ValueError(
                "Input image is None. Check the image path."
            )

        image = image.astype(np.float32)

        # OpenCV uses BGR order
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        total = R + G + B + 1e-6

        r = R / total
        g = G / total
        b = B / total

        exg = 2 * g - r - b

        return exg

    # -------------------------------------------------
    # Segment vegetation
    # -------------------------------------------------

    def segment(self, image, threshold=0.05):

        exg = self.compute_exg(image)

        mask = (exg > threshold).astype(np.uint8) * 255

        return mask

    # -------------------------------------------------
    # Morphological filtering
    # -------------------------------------------------

    def clean_mask(
        self,
        mask,
        kernel_size=5
    ):
        """
        Remove small noise and fill small holes.
        """

        kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8
        )

        # Remove isolated pixels
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Fill holes
        cleaned = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel
        )

        return cleaned

    # -------------------------------------------------
    # Vegetation percentage
    # -------------------------------------------------

    def vegetation_percentage(self, mask):

        vegetation_pixels = np.sum(mask > 0)

        total_pixels = mask.size

        return 100.0 * vegetation_pixels / total_pixels