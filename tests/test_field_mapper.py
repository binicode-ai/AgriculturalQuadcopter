from agriculture.field_mapper import FieldMapper

mapper = FieldMapper()

observations = [

    (2, 3, "Healthy"),

    (5, 6, "Rust"),

    (8, 4, "Blight"),

    (3, 8, "Healthy"),

    (6, 2, "LeafSpot"),

    (9, 7, "Mildew"),

    (12, 4, "Healthy"),

    (13, 6, "Rust")

]

for observation in observations:

    mapper.add_observation(*observation)

mapper.plot()