"""Scoring-only route checks. These measurements never enter policy inputs."""

from dataclasses import dataclass, field


@dataclass
class CourseProgress:
    # Scoring spans enclosing each obstacle and the rough patch.
    checkpoints: tuple = (
        (1.98, 2.02, 0.65),
        (4.58, 5.02, 0.65),
        (7.23, 7.37, 0.65),
        (7.915, 9.275, 0.55),
    )
    completed: int = 0
    entered: bool = False
    previous_feet: list | None = field(default=None, repr=False)

    def update(self, feet):
        """Both feet must pass each challenge in order, within its width.

        This verifies route coverage, not airborne clearance. A player may
        return to the front of a missed checkpoint and try again.
        """
        if self.completed == len(self.checkpoints):
            return
        front, back, half_width = self.checkpoints[self.completed]
        inside = all(abs(foot[1]) <= half_width for foot in feet)
        if not inside:
            self.entered = False
        elif self.previous_feet is not None:
            before = all(foot[0] < front for foot in self.previous_feet)
            if before and all(
                abs(foot[1]) <= half_width for foot in self.previous_feet
            ):
                self.entered = True
            if self.entered and all(foot[0] > back for foot in feet):
                self.completed += 1
                self.entered = False
        self.previous_feet = feet

    @property
    def complete(self):
        return self.completed == len(self.checkpoints)
