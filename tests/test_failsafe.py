from autopilot.failsafe import FailsafeManager

manager = FailsafeManager()

tests = [

    (True, True, True, False, False),

    (True, True, False, False, False),

    (False, True, True, False, False),

    (True, True, True, True, False),

    (True, True, True, False, True),

    (True, False, True, False, False)

]

for case in tests:

    manager.update(*case)

    manager.print_status()