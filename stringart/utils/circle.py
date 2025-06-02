from typing import List

import numpy as np
from stringart.line.bresenham import Bresenham
from stringart.utils.types import Point


def compute_pegs(number_of_pegs: int, radius: int, center_point: Point = Point(0, 0)) -> List[Point]:
    return [
        Point(
            int(radius * np.cos(2 * np.pi * k / number_of_pegs) + center_point.x),
            int(radius * np.sin(2 * np.pi * k / number_of_pegs) + center_point.y),
        )
        for k in range(number_of_pegs)
    ]


def compute_line_lengths(number_of_pegs: int, radius: int, center_point: Point) -> List[float]:
    pegs: List[Point] = compute_pegs(number_of_pegs, radius, center_point)
    distances: List[float] = []

    for i in range(len(pegs)):
        for j in range(i + 1, len(pegs)):
            line = Bresenham.compute_line(pegs[i], pegs[j])
            distances.append(len(line))

    return distances
