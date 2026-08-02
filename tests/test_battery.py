from hardware.battery import Battery

battery = Battery()

print()

print("Initial")

print(f"{battery.percentage():.1f}%")

for i in range(60):

    battery.consume(

        power_watts=300,

        dt=10

    )

    if i % 10 == 0:

        print(

            f"{battery.percentage():.1f}%"

        )

print()

print(

    "Estimated Time:",

    battery.estimated_time(300),

    "minutes"

)

print()

print(

    "Low:",

    battery.is_low()

)

print(

    "Critical:",

    battery.is_critical()

)