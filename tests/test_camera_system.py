import numpy as np

from payload.camera_system import CameraSystem

camera = CameraSystem()

for i in range(5):

    camera.capture(

        time=i,

        position=np.array([
            i,
            2*i,
            20
        ]),

        attitude=np.array([
            0.0,
            0.0,
            0.1*i
        ])

    )

print()

print("Images Captured")

print(camera.number_of_images())

print()

for image in camera.get_images():

    print(image)