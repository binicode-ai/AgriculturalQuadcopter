import cv2

from vision.vegetation import VegetationDetector
from vision.plant_counter import PlantCounter

image = cv2.imread("data/sample_field.jpg")

if image is None:
    raise FileNotFoundError(
        "sample_field.jpg not found"
    )

vegetation = VegetationDetector()

mask = vegetation.segment(image)

mask = vegetation.clean_mask(mask)

counter = PlantCounter(
    min_area=80
)

plants = counter.detect(mask)

print()

print("Plants Detected:", len(plants))

result = counter.draw(
    image,
    plants
)

cv2.imshow("Original", image)

cv2.imshow("Mask", mask)

cv2.imshow("Plants", result)

cv2.waitKey(0)

cv2.destroyAllWindows()