from autopilot.return_home import ReturnHome

rth = ReturnHome(arrival_radius=1.5)

rth.set_home(

    0,

    0,

    2

)

rth.activate()

positions = [

    (15, 12, 2),

    (8, 5, 2),

    (3, 2, 2),

    (1, 0.5, 2),

    (0.3, 0.2, 2)

]

for position in positions:

    x, y, z = position

    distance = rth.distance_to_home(

        x,

        y,

        z

    )

    print()

    print("Current:", position)

    print(f"Distance: {distance:.2f} m")

    print(

        "Reached:",

        rth.reached_home(

            x,

            y,

            z

        )

    )