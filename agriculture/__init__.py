"""
agriculture/target_selector.py

Selects whether a detected plant
should be sprayed.

Author: Biniyam Samuel
"""


class TargetSelector:

    def __init__(

        self,

        confidence_threshold=0.85

    ):

        self.confidence_threshold = confidence_threshold

    # -----------------------------------------

    def should_spray(

        self,

        disease,

        confidence

    ):

        if confidence < self.confidence_threshold:

            return False

        if disease == "Healthy":

            return False

        return True