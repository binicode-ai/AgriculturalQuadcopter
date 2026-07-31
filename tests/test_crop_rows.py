import cv2

from vision.vegetation import VegetationDetector
from vision.crop_row import CropRowDetector

image = cv2.imread("data/sample_field.jpg")

if image is None:
    raise FileNotFoundError("sample_field.jpg not found")

vegetation = VegetationDetector()

mask = vegetation.segment(image)

mask = vegetation.clean_mask(mask)

detector = CropRowDetector()

lines = detector.detect(mask)

lines = detector.filter_by_angle(lines)

result = detector.draw(image, lines)

print("Detected Rows :", detector.number_of_rows(lines))
print("Average Angle :", detector.average_angle(lines))

cv2.imshow("Original", image)
cv2.imshow("Vegetation", mask)
cv2.imshow("Crop Rows", result)

cv2.waitKey(0)
cv2.destroyAllWindows()