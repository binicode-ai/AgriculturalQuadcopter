from agriculture.target_selector import TargetSelector

selector = TargetSelector()

tests = [

    ("Healthy", 0.99),

    ("Rust", 0.95),

    ("Rust", 0.60),

    ("Blight", 0.93),

    ("LeafSpot", 0.82),

    ("Mildew", 0.91)

]

print()

print("====== TARGET SELECTION ======")

for disease, confidence in tests:

    spray = selector.should_spray(

        disease,

        confidence

    )

    print(

        f"{disease:10s}"

        f"{confidence:.2f}"

        f" -> "

        f"{spray}"

    )