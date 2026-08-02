"""
mission/scheduler.py

Mission scheduler and task executor.

Author: Biniyam Samuel
"""


class MissionScheduler:

    def __init__(self):

        self.tasks = []

        self.current = 0

    # -------------------------------------

    def add_task(self, name, callback):

        self.tasks.append({

            "name": name,

            "callback": callback

        })

    # -------------------------------------

    def execute(self):

        while self.current < len(self.tasks):

            task = self.tasks[self.current]

            print(f"\nExecuting: {task['name']}")

            task["callback"]()

            self.current += 1

        print("\nMission Completed.")

    # -------------------------------------

    def reset(self):

        self.current = 0