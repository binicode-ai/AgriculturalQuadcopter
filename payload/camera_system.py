"""
payload/camera_system.py

Virtual RGB camera system.

Author: Biniyam Samuel
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ImageRecord:

    image_id: int

    timestamp: float

    position: np.ndarray

    attitude: np.ndarray


class CameraSystem:

    def __init__(self):

        self.images = []

        self.image_counter = 0

    def capture(
        self,
        time,
        position,
        attitude
    ):

        image = ImageRecord(

            image_id=self.image_counter,

            timestamp=time,

            position=position.copy(),

            attitude=attitude.copy()

        )

        self.images.append(image)

        self.image_counter += 1

        return image

    def number_of_images(self):

        return len(self.images)

    def get_images(self):

        return self.images