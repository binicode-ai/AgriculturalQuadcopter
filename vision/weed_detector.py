"""
vision/weed_detector.py

Simple weed detector based on distance to crop rows.
"""

import cv2
import numpy as np


class WeedDetector:

    def __init__(self, row_distance=40):
        self.row_distance = row_distance

    # -------------------------------------------------
    # Distance from point to line segment
    # -------------------------------------------------

    def point_to_line_distance(
        self,
        point,
        line
    ):

        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line

        px, py = point

        A = np.array([x1, y1], dtype=float)
        B = np.array([x2, y2], dtype=float)
        P = np.array([px, py], dtype=float)

        AB = B - A

        if np.linalg.norm(AB) < 1e-6:
            return np.linalg.norm(P - A)

        t = np.dot(P - A, AB) / np.dot(AB, AB)

        t = np.clip(t, 0, 1)

        projection = A + t * AB

        return np.linalg.norm(P - projection)

    # -------------------------------------------------

    def classify(
        self,
        plant_centers,
        crop_rows
    ):

        crops = []
        weeds = []

        if crop_rows is None or len(crop_rows) == 0:
            weeds.extend(plant_centers)
            return crops, weeds

        for center in plant_centers:

            minimum = 1e9

            for row in crop_rows:

                d = self.point_to_line_distance(
                    center,
                    row
                )

                minimum = min(minimum, d)

            if minimum < self.row_distance:
                crops.append(center)
            else:
                weeds.append(center)

        return crops, weeds

    # -------------------------------------------------

    def draw(
        self,
        image,
        crops,
        weeds
    ):

        output = image.copy()

        for x, y in crops:

            cv2.circle(
                output,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

        for x, y in weeds:

            cv2.circle(
                output,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

        return output