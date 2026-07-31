from navigation.coverage import CoveragePlanner

planner = CoveragePlanner(
    field_width=40,
    field_length=30,
    line_spacing=10,
    altitude=15
)

waypoints = planner.generate()

print("Coverage Waypoints\n")

for i, wp in enumerate(waypoints):

    print(f"{i+1:2d}: {wp}")