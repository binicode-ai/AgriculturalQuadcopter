import cv2

from ai.disease_detector import DiseaseDetector

image = cv2.imread(
    "data/sample_field.jpg"
)

if image is None:
    raise FileNotFoundError(
        "Image not found."
    )

detector = DiseaseDetector()

detector.load_model(
    "trained_models/crop_model.pt"
)

label, confidence = detector.predict(image)

print()

print("Disease")

print(label)

print()

print("Confidence")

print(confidence)