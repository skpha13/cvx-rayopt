from typing import List

import numpy as np
from stringart.utils.types import Point


def compute_pegs(number_of_pegs: int, radius: int, center_point: Point = Point(0, 0)) -> List[Point]:
    return [
        Point(
            int(radius * np.cos(2 * np.pi * k / number_of_pegs) + center_point.x),
            int(radius * np.sin(2 * np.pi * k / number_of_pegs) + center_point.y),
        )
        for k in range(number_of_pegs)
    ]
