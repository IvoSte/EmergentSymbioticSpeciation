import random
import numpy as np
from math import pi, sin, cos, atan2, degrees, radians, sqrt, copysign
from shared_components.logger import log


def degree_to_radian(degrees):
    degrees = degrees % 360
    return degrees * pi / 180


class SpaceCalculator:
    def __init__(self, size_x, size_y, toroidal=True):
        self.min_x = 0
        self.min_y = 0
        self.max_x = size_x
        self.max_y = size_y
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

        self.center = np.array(
            ((self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2)
        )
        self.size = np.array((self.width, self.height))
        self.torus = toroidal

    def get_random_position(self) -> tuple:
        x = random.randint(0, self.max_x)
        y = random.randint(0, self.max_y)
        return (x, y)

    def get_pos_with_delta(self, pos, dx, dy) -> tuple:
        x = (pos[0] + dx) % self.max_x
        y = (pos[1] + dy) % self.max_y
        return (x, y)

    def get_pos_with_heading_distance(self, pos, heading, distance) -> tuple:
        rad = radians(heading)
        dx = distance * cos(rad)
        dy = distance * sin(rad)
        x = (pos[0] + dx) % self.max_x
        y = (pos[1] + dy) % self.max_y
        return (x, y)

    def get_position_delta(self, pos_1, pos_2) -> tuple:
        dx = pos_2[0] - pos_1[0]
        dy = pos_2[1] - pos_1[1]
        # Get the closest dx and dy, with the correct sign.
        if self.torus:
            dx = (
                dx
                if abs(dx) < abs(self.width - abs(dx))
                else -1.0 * copysign(self.width - abs(dx), dx)
            )
            dy = (
                dy
                if abs(dy) < abs(self.height - abs(dy))
                else -1.0 * copysign(self.height - abs(dy), dy)
            )
        return (dx, dy)

    def get_heading_to_pos(self, pos_1, pos_2) -> float:
        dx, dy = self.get_position_delta(pos_1, pos_2)
        angle = atan2(dy, dx)
        heading = degrees(angle)
        return heading

    def get_distance_to_pos(self, pos_1, pos_2) -> float:
        dx = abs(pos_2[0] - pos_1[0])
        dy = abs(pos_2[1] - pos_1[1])
        if self.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        distance = np.sqrt(dx * dx + dy * dy)
        return distance

    def get_average_position_delta(
        self, pos_1, other_positions: list[tuple]
    ) -> tuple[float]:
        position_deltas = [
            self.get_position_delta(pos_1, pos_2) for pos_2 in other_positions
        ]
        dx, dy = self.get_average_position(position_deltas)
        return (dx, dy)

    def get_average_position(self, positions: list[tuple]) -> tuple[float]:
        x = sum(pos[0] for pos in positions) / len(positions)
        y = sum(pos[1] for pos in positions) / len(positions)
        return (x, y)

    def get_distance_weighted_average_position_delta(
        self, pos_1: tuple, other_positions: list[tuple]
    ) -> tuple[float]:
        log.warning("Unimplemented function called.")
        return (0, 0)

    def get_distance(self, dx, dy):
        return sqrt(dx**2 + dy**2)
