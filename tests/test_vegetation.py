import cv2
import os

from vision.vegetation import VegetationDetector

image_path = "data/sample_field.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        os.path.abspath(image_path)
    )

detector = VegetationDetector()

mask = detector.segment(image)

cleaned = detector.clean_mask(mask)

coverage = detector.vegetation_percentage(cleaned)

print()

print("Vegetation Coverage")

print(f"{coverage:.2f}%")

cv2.imshow("Original", image)

cv2.imshow("Raw Mask", mask)

cv2.imshow("Cleaned Mask", cleaned)

cv2.waitKey(0)

cv2.destroyAllWindows()