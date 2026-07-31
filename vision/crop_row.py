"""
vision/crop_rows.py

Crop row detection using the Hough Transform.

Author: Biniyam Samuel
"""

import cv2
import numpy as np


class CropRowDetector:
    """
    Detect crop rows from a binary vegetation mask.
    """

    def __init__(
        self,
        threshold=100,
        min_line_length=100,
        max_line_gap=20
    ):

        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    # -------------------------------------------------
    # Detect crop rows
    # -------------------------------------------------

    def detect(self, mask):

        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        return lines

    # -------------------------------------------------
    # Filter by orientation
    # -------------------------------------------------

    def filter_by_angle(
        self,
        lines,
        min_angle=70,
        max_angle=110
    ):
        """
        Keep only lines whose angle lies within
        the specified range.
        """

        if lines is None:
            return []

        filtered = []

        for line in lines:

            # Handle both OpenCV formats
            if len(line) == 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line

            angle = abs(
                np.degrees(
                    np.arctan2(
                        y2 - y1,
                        x2 - x1
                    )
                )
            )

            if min_angle <= angle <= max_angle:
                filtered.append(line)

        return filtered

    # -------------------------------------------------
    # Draw detected rows
    # -------------------------------------------------

    def draw(self, image, lines):

        output = image.copy()

        if lines is None or len(lines) == 0:
            return output

        for line in lines:

            # Support both OpenCV output formats
            if len(line) == 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line

            cv2.line(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2
            )

        return output

    # -------------------------------------------------
    # Compute average row angle
    # -------------------------------------------------

    def average_angle(self, lines):
        """
        Returns the average crop-row angle (degrees).
        """

        if lines is None or len(lines) == 0:
            return None

        angles = []

        for line in lines:

            if len(line) == 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            angles.append(angle)

        return np.mean(angles)

    # -------------------------------------------------
    # Count detected rows
    # -------------------------------------------------

    def number_of_rows(self, lines):

        if lines is None:
            return 0

        return len(lines)