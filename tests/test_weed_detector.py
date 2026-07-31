import cv2

from vision.vegetation import VegetationDetector
from vision.crop_row import CropRowDetector
from vision.plant_counter import PlantCounter
from vision.weed_detector import WeedDetector

image = cv2.imread("data/sample_field.jpg")

if image is None:
    raise FileNotFoundError("Image not found")

vegetation = VegetationDetector()

mask = vegetation.segment(image)
mask = vegetation.clean_mask(mask)

rows = CropRowDetector()

lines = rows.detect(mask)
lines = rows.filter_by_angle(lines)

counter = PlantCounter()

plants = counter.detect(mask)

centers = counter.centers(plants)

weed_detector = WeedDetector()

crops, weeds = weed_detector.classify(
    centers,
    lines
)

result = weed_detector.draw(
    image,
    crops,
    weeds
)

print()
print("Crop Plants :", len(crops))
print("Weeds       :", len(weeds))

cv2.imshow("Classification", result)

cv2.waitKey(0)

cv2.destroyAllWindows()