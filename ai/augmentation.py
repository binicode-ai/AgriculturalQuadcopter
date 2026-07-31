"""
ai/augmentation.py

Image augmentation for agricultural datasets.

Author: Biniyam Samuel
"""

import cv2
import numpy as np


class ImageAugmentor:

    def __init__(self):
        pass

    # ------------------------------------------------

    def rotate(self, image, angle):

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (w, h)
        )

    # ------------------------------------------------

    def flip_horizontal(self, image):

        return cv2.flip(image, 1)

    # ------------------------------------------------

    def flip_vertical(self, image):

        return cv2.flip(image, 0)

    # ------------------------------------------------

    def adjust_brightness(
        self,
        image,
        factor
    ):

        image = image.astype(np.float32)

        image *= factor

        image = np.clip(
            image,
            0,
            255
        )

        return image.astype(np.uint8)

    # ------------------------------------------------

    def add_gaussian_noise(
        self,
        image,
        sigma=10
    ):

        noise = np.random.normal(
            0,
            sigma,
            image.shape
        )

        noisy = image.astype(np.float32)

        noisy += noise

        noisy = np.clip(
            noisy,
            0,
            255
        )

        return noisy.astype(np.uint8)

    # ------------------------------------------------

    def blur(
        self,
        image,
        kernel=5
    ):

        return cv2.GaussianBlur(
            image,
            (kernel, kernel),
            0
        )