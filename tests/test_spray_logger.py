from agriculture.spray_logger import SprayLogger

logger = SprayLogger()

events = [

    (10.2, 5.1, "Rust", 0.95, "Fungicide-A", 1.5, 0.05),

    (12.8, 7.4, "Blight", 0.98, "Copper Fungicide", 2.0, 0.07),

    (18.3, 9.2, "Mildew", 0.91, "Sulfur Spray", 1.0, 0.04)

]

for event in events:

    logger.log(*event)

print()

print("Spray log created successfully.")