import cv2

from ai.augmentation import ImageAugmentor

image = cv2.imread(
    "data/sample_field.jpg"
)

if image is None:
    raise FileNotFoundError(
        "Image not found."
    )

augmentor = ImageAugmentor()

rotated = augmentor.rotate(image, 20)

flipped = augmentor.flip_horizontal(image)

bright = augmentor.adjust_brightness(
    image,
    1.4
)

noise = augmentor.add_gaussian_noise(image)

blur = augmentor.blur(image)

cv2.imshow("Original", image)

cv2.imshow("Rotated", rotated)

cv2.imshow("Flipped", flipped)

cv2.imshow("Bright", bright)

cv2.imshow("Noise", noise)

cv2.imshow("Blur", blur)

cv2.waitKey(0)

cv2.destroyAllWindows()