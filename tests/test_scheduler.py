from mission.scheduler import MissionScheduler


def takeoff():
    print("Taking off...")


def climb():
    print("Climbing to 10 m...")


def navigate():
    print("Flying to field...")


def inspect():
    print("Inspecting crops...")


def spray():
    print("Spraying diseased plants...")


def return_home():
    print("Returning home...")


def land():
    print("Landing...")


scheduler = MissionScheduler()

scheduler.add_task("Takeoff", takeoff)
scheduler.add_task("Climb", climb)
scheduler.add_task("Navigate", navigate)
scheduler.add_task("Inspection", inspect)
scheduler.add_task("Spraying", spray)
scheduler.add_task("Return Home", return_home)
scheduler.add_task("Landing", land)

scheduler.execute()