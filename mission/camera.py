"""
mission/camera.py

Camera geometry model for agricultural UAV missions.

Author: Biniyam Samuel
"""

import numpy as np


class Camera:

    def __init__(
        self,
        sensor_width,
        sensor_height,
        focal_length,
        image_width,
        image_height
    ):

        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.focal_length = focal_length

        self.image_width = image_width
        self.image_height = image_height

    # -------------------------------------------------
    # Field of View
    # -------------------------------------------------

    def horizontal_fov(self):

        return 2 * np.arctan(
            self.sensor_width /
            (2 * self.focal_length)
        )

    def vertical_fov(self):

        return 2 * np.arctan(
            self.sensor_height /
            (2 * self.focal_length)
        )

    # -------------------------------------------------
    # Ground Footprint
    # -------------------------------------------------

    def footprint(self, altitude):

        width = (
            2
            * altitude
            * np.tan(self.horizontal_fov() / 2)
        )

        height = (
            2
            * altitude
            * np.tan(self.vertical_fov() / 2)
        )

        return width, height

    # -------------------------------------------------
    # Distance Between Flight Lines
    # -------------------------------------------------

    def flight_spacing(
        self,
        altitude,
        side_overlap
    ):

        width, _ = self.footprint(altitude)

        return width * (1 - side_overlap)

    # -------------------------------------------------
    # Distance Between Photos
    # -------------------------------------------------

    def photo_interval(
        self,
        altitude,
        forward_overlap
    ):

        _, height = self.footprint(altitude)

        return height * (1 - forward_overlap)