from hardware.simulated_hardware import SimulatedHardware

hardware = SimulatedHardware()

hardware.arm()

hardware.set_motor_speeds(

    [400, 400, 400, 400]

)

print(hardware.read_gps())

print(hardware.read_imu())

image = hardware.capture_image()

print(image)

hardware.spray()

hardware.disarm()