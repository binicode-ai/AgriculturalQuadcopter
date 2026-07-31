"""
AI Disease Detector Framework

Author: Biniyam Samuel
"""

import cv2
import numpy as np


class DiseaseDetector:

    def __init__(self):

        self.model_loaded = False
        self.class_names = [
            "Healthy",
            "Leaf Spot",
            "Rust",
            "Blight",
            "Mildew"
        ]

    # ---------------------------------

    def load_model(self, path):

        """
        Placeholder for future AI model.
        """

        print(f"Loading model: {path}")

        self.model_loaded = True

    # ---------------------------------

    def preprocess(self, image):

        image = cv2.resize(
            image,
            (224,224)
        )

        image = image.astype(np.float32)

        image /= 255.0

        return image

    # ---------------------------------

    def predict(self, image):

        """
        Temporary placeholder.

        Later this function will call
        TensorFlow or PyTorch.
        """

        if not self.model_loaded:

            raise RuntimeError(
                "Model not loaded."
            )

        image = self.preprocess(image)

        label = self.class_names[0]

        confidence = 1.0

        return label, confidence