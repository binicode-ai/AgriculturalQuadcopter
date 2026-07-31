from agriculture.spray_controller import SprayController

import time

sprayer = SprayController(

    tank_capacity=2.0,

    flow_rate=0.10,

    spray_duration=2.0,

    cooldown=1.0

)

print()

print("Initial Tank")

print(f"{sprayer.level_percent():.1f}%")

for i in range(5):

    sprayer.spray()

    time.sleep(1.2)

print()

print("Final Tank")

print(f"{sprayer.level_percent():.1f}%")