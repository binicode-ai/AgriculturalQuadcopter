from controllers.pid import PID

pid = PID(
    kp=2,
    ki=0.5,
    kd=0.1
)

measurement = 0

target = 10

dt = 0.01

for i in range(500):

    control = pid.update(
        target,
        measurement,
        dt
    )

    measurement += control * 0.005

    print(
        i,
        measurement
    )