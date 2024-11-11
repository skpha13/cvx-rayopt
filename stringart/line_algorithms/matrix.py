from typing import List

from stringart.utils.types import Point


class MatrixGenerator:
    def __init__(self, lines: List[Point], pegs: List[Point]):
        self.lines: List[Point] = lines
        self.pegs: List[Point] = pegs

    # TODO: maybe replace default values
    def compute_matrix(self, number_of_pegs: int = 200, radius: int | None = None):
        pass
