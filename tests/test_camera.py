from mission.camera import Camera

camera = Camera(

    sensor_width=13.2,

    sensor_height=8.8,

    focal_length=8.8,

    image_width=5472,

    image_height=3648

)

altitude = 50


print(camera.horizontal_fov())

print(camera.vertical_fov())

print(camera.footprint(40))

print(camera.flight_spacing(
    40,
    0.70
))

print(camera.photo_interval(
    40,
    0.80
))