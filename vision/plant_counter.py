"""
vision/plant_counter.py

Plant counting using connected-component analysis.

Author: Biniyam Samuel
"""

import cv2
import numpy as np


class PlantCounter:

    def __init__(self, min_area=50):

        self.min_area = min_area

    # ---------------------------------------------
    # Detect plants
    # ---------------------------------------------

    def detect(self, mask):

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        plants = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            plants.append(contour)

        return plants

    # ---------------------------------------------
    # Count plants
    # ---------------------------------------------

    def count(self, mask):

        plants = self.detect(mask)

        return len(plants)

    # ---------------------------------------------
    # Draw detections
    # ---------------------------------------------

    def draw(self, image, plants):

        output = image.copy()

        for contour in plants:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(

                output,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2

            )

            cv2.circle(

                output,

                (x + w // 2, y + h // 2),

                3,

                (0, 0, 255),

                -1

            )

        return output

    # ---------------------------------------------
    # Plant centers
    # ---------------------------------------------

    def centers(self, plants):

        centers = []

        for contour in plants:

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))

        return centers