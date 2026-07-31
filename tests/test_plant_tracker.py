from agriculture.plant_tracker import PlantTracker

tracker = PlantTracker(

    distance_threshold=1.5

)

plants = [

    (0,0),

    (0.3,0.2),

    (5,5),

    (5.6,5.3),

    (12,4),

    (0.8,0.5)

]

print()

print("===== PLANT TRACKER =====")

for plant in plants:

    x, y = plant

    if tracker.already_sprayed(

        x,

        y

    ):

        print(

            plant,

            "Already sprayed"

        )

    else:

        print(

            plant,

            "NEW -> Spray"

        )

        tracker.add(

            x,

            y

        )

print()

print(

    "Stored Plants:",

    tracker.count()

)